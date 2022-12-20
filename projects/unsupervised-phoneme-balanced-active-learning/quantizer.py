import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.models.wav2vec2 import Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
    _compute_mask_indices,
    _sample_negative_indices,
)
from transformers.utils import ModelOutput


@dataclass
class MyWav2Vec2ForPreTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    projected_states: torch.FloatTensor = None
    projected_quantized_states: torch.FloatTensor = None
    codevector_perplexity: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    contrastive_loss: Optional[torch.FloatTensor] = None
    diversity_loss: Optional[torch.FloatTensor] = None
    quantized_indices: Optional[torch.LongTensor] = None


class MyWav2Vec2GumbelVectorQuantizer(torch.nn.Module):
    """
    Vector quantization using gumbel softmax. See `[CATEGORICAL REPARAMETERIZATION WITH
    GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf) for more information.
    """

    def __init__(self, config):
        super().__init__()
        # config.codevector_dim: d
        # 潜在表現の分割数（G）
        self.num_groups = config.num_codevector_groups
        # 各分割ごとのコードブックの大きさ (V)
        self.num_vars = config.num_codevectors_per_group

        if config.codevector_dim % self.num_groups != 0:
            raise ValueError(
                f"`config.codevector_dim {config.codevector_dim} must be divisible "
                f"by `config.num_codevector_groups` {self.num_groups} for concatenation"
            )

        # storage for codebook variables (codewords)
        # (1, G * V, d / G)
        self.codevectors = torch.nn.Parameter(
            torch.FloatTensor(1, self.num_groups * self.num_vars, config.codevector_dim // self.num_groups)
        )
        # from transformer output to G * V
        self.weight_proj = torch.nn.Linear(config.conv_dim[-1], self.num_groups * self.num_vars)

        # can be decayed for training
        self.temperature = 2

    @staticmethod
    def _compute_perplexity(probs, mask=None):
        if mask is not None:
            mask_extended = mask.flatten()[:, None, None].expand(probs.shape)
            probs = torch.where(mask_extended, probs, torch.zeros_like(probs))
            marginal_probs = probs.sum(dim=0) / mask.sum()
        else:
            marginal_probs = probs.mean(dim=0)

        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)).sum()
        return perplexity

    def forward(self, hidden_states, mask_time_indices=None):
        # hidden_states: (B, T, d)
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # project to codevector dim: (B, T, G * V)
        hidden_states = self.weight_proj(hidden_states)
        # (B * T * G, V)
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        if self.training:
            # sample code vector probs via gumbel in differentiateable way
            # compute code vector probs: (B * T * G, V)
            # 各バッチの各時刻の各グループごとにV個の量子化表現のうち、どれにするかを確率で表現
            # 近いのに量子化しているわけではなく、選択過程も学習される
            codevector_probs = torch.nn.functional.gumbel_softmax(
                hidden_states.float(), tau=self.temperature, hard=True
            ).type_as(hidden_states)

            # compute perplexity
            codevector_soft_dist = torch.softmax(
                hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1
            )
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
        else:
            # take argmax in non-differentiable way
            # comptute hard codevector distribution (one hot)
            codevector_idx = hidden_states.argmax(dim=-1)
            codevector_probs = hidden_states.new_zeros(hidden_states.shape).scatter_(
                -1, codevector_idx.view(-1, 1), 1.0
            )
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)

            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)
        # (B * T, G * V)
        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
        # use probs to retrieve codevectors
        # (B * T, G * V, 1) @ (1, G * V, d / G) -> (B * T, G * V, d / G)
        # ブロードキャストについて:
        # 値が1のものは繰り返しによって、(B * T, G * V, d / G)になる。つまり、量子化表現の各要素に同じ確率が掛けられる
        # また、コードベクターは、(1, G * V, d / G)になっているので、(B * T, G * V, d / G)になる。つまり、同じコードベクターが各バッチ、各時刻に割り当てられる
        # これらのことから各バッチ、各時刻における各グループの各量子化ベクトルに確率を掛けたものが得られる
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors
        # (B * T, G, V, d / G)
        codevectors = codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        # 各グループについてV方向の和を計算することで、各グループの量子化表現を得る
        # どうして？ -> 学習時には各コードブック内の量子化表現の重み付き和で計算しており、推論時にはArgmaxでone-hot確率を計算し、最も確率が高い量子化表現を選択しているため
        codevectors = codevectors.sum(-2).view(batch_size, sequence_length, -1)
        # w2v-BERT用のインデックス配列
        codevector_idx = hidden_states.argmax(dim=-1).view(batch_size, sequence_length, -1)
        # codevectors: (B, T, d)
        # codevector_idx: (B, T, G)
        return codevectors, codevector_idx, perplexity


class MyWav2Vec2ForPreTraining(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout_features = nn.Dropout(config.feat_quantizer_dropout)

        self.quantizer = MyWav2Vec2GumbelVectorQuantizer(config)

        # Initialize weights and apply final processing
        self.post_init()

        # make sure that project_hid & project_q are initialized like normal linear layers
        self.project_hid = nn.Linear(config.hidden_size, config.proj_codevector_dim)
        self.project_q = nn.Linear(config.codevector_dim, config.proj_codevector_dim)

    def set_gumbel_temperature(self, temperature: int):
        """
        Set the Gumbel softmax temperature to a given value. Only necessary for training
        """
        self.quantizer.temperature = temperature

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5."
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    @staticmethod
    def compute_contrastive_logits(
        target_features: torch.FloatTensor,
        negative_features: torch.FloatTensor,
        predicted_features: torch.FloatTensor,
        temperature: int = 0.1,
    ):
        """
        Compute logits for contrastive loss based using cosine similarity as the distance measure between
        `[positive_feature, negative_features]` and `[predicted_features]`. Additionally, temperature can be applied.
        """
        target_features = torch.cat([target_features, negative_features], dim=0)

        logits = torch.cosine_similarity(predicted_features.float(), target_features.float(), dim=-1).type_as(
            target_features
        )

        # apply temperature
        logits = logits / temperature
        return logits

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.BoolTensor] = None,
        sampled_negative_indices: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MyWav2Vec2ForPreTrainingOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask_time_indices=mask_time_indices,
            return_dict=return_dict,
        )

        # 1. project all transformed features (including masked) to final vq dim
        transformer_features = self.project_hid(outputs[0])

        # 2. quantize all (unmasked) extracted features and project to final vq dim
        extract_features = self.dropout_features(outputs[1])

        if attention_mask is not None:
            # compute reduced attention_mask correponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        quantized_features, quantized_indices, codevector_perplexity = self.quantizer(
            extract_features, mask_time_indices=mask_time_indices
        )
        quantized_features = self.project_q(quantized_features)

        loss = contrastive_loss = diversity_loss = None
        if sampled_negative_indices is not None:
            batch_size, sequence_length, hidden_size = quantized_features.shape

            # for training, we sample negatives
            # 3. sample K negatives (distractors) quantized states for contrastive loss
            # if attention_mask is passed, make sure that padded feature vectors cannot be sampled
            # sample negative quantized vectors BTC => (BxT)C
            negative_quantized_features = quantized_features.view(-1, hidden_size)[
                sampled_negative_indices.long().view(-1)
            ]
            negative_quantized_features = negative_quantized_features.view(
                batch_size, sequence_length, -1, hidden_size
            ).permute(2, 0, 1, 3)

            # 4. compute logits, corresponding to `logs = sim(c_t, [q_t, \sim{q}_t]) / \kappa`
            # of equation (3) in https://arxiv.org/pdf/2006.11477.pdf
            logits = self.compute_contrastive_logits(
                quantized_features[None, :],
                negative_quantized_features,
                transformer_features,
                self.config.contrastive_logits_temperature,
            )

            # 5. if a negative vector is identical to the positive (i.e. when codebook utilization is low),
            # its cosine similarity will be masked
            neg_is_pos = (quantized_features == negative_quantized_features).all(-1)

            if neg_is_pos.any():
                logits[1:][neg_is_pos] = float("-inf")

            # 6. compute contrastive loss \mathbf{L}_m = cross_entropy(logs) =
            # -log(exp(sim(c_t, q_t)/\kappa) / \sum_{\sim{q}} exp(sim(c_t, \sim{q})/\kappa))
            logits = logits.transpose(0, 2).reshape(-1, logits.size(0))
            target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()

            contrastive_loss = nn.functional.cross_entropy(logits.float(), target, reduction="sum")
            # 7. compute diversity loss: \mathbf{L}_d
            num_codevectors = self.config.num_codevectors_per_group * self.config.num_codevector_groups
            diversity_loss = ((num_codevectors - codevector_perplexity) / num_codevectors) * mask_time_indices.sum()

            # 8. \mathbf{L} = \mathbf{L}_m + \alpha * \mathbf{L}_d
            loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss

        if not return_dict:
            if loss is not None:
                return (loss, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
            return (transformer_features, quantized_features, codevector_perplexity) + outputs[2:]

        return MyWav2Vec2ForPreTrainingOutput(
            loss=loss,
            projected_states=transformer_features,
            projected_quantized_states=quantized_features,
            codevector_perplexity=codevector_perplexity,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            contrastive_loss=contrastive_loss,
            diversity_loss=diversity_loss,
            quantized_indices=quantized_indices,
        )


class Quantizer:
    def __init__(self, device):
        self.model = MyWav2Vec2ForPreTraining.from_pretrained(
            "facebook/wav2vec2-large-lv60",
            cache_dir="/home/shibutani/fs/.cache/huggingface/transformers").to(device)
        self.num_codevector_groups = self.model.config.num_codevector_groups
        self.num_codevectors_per_group = self.model.config.num_codevectors_per_group
        self.idx_pair_to_idx_mapper = {}
        for i in range(self.num_codevectors_per_group):
            for j in range(self.num_codevectors_per_group):
                self.idx_pair_to_idx_mapper[(i, j)] = i * self.num_codevectors_per_group + j

    @torch.no_grad()
    def quantize(self, bx):
        # bx: (batch_size, seq_len)
        batch_size = bx.shape[0]
        # output length from conv layer
        sequence_length = self.model._get_feat_extract_output_lengths(bx.shape[1]).item()
        mask_time_indices = _compute_mask_indices(
            shape=(batch_size, sequence_length), mask_prob=0.2, mask_length=2
        )
        sampled_negative_indices = _sample_negative_indices(
            features_shape=(batch_size, sequence_length),
            num_negatives=self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        mask_time_indices = torch.tensor(mask_time_indices, device=bx.device, dtype=torch.long)
        sampled_negative_indices = torch.tensor(
            data=sampled_negative_indices, device=bx.device, dtype=torch.long
        )
        self.model.eval()
        outputs = self.model(
            bx, mask_time_indices=mask_time_indices, sampled_negative_indices=sampled_negative_indices
        )
        bquantized_indices = outputs.quantized_indices
        # quantized_indices: (batch_size, seq_len, 2) -> (batch_size, seq_len)
        bquantized_indices = torch.tensor(
            [[self.idx_pair_to_idx_mapper[tuple(idx_pair.tolist())] for idx_pair in quantized_indices]
                for quantized_indices in bquantized_indices],
            device=bx.device, dtype=torch.long
        )

        return bquantized_indices
