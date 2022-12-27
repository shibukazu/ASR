import torch
from modules.preprocessing.spec_aug import SpecAug
from omegaconf import DictConfig
from torchaudio.pipelines import WAV2VEC2_BASE


class Model(torch.nn.Module):
    def __init__(self, nlabel, cfg: DictConfig):
        super(Model, self).__init__()
        bundle = WAV2VEC2_BASE
        self.nlabel = nlabel
        self.wav2vec2 = bundle.get_model()
        # 1. Convレイヤー
        self.feature_extractor = self.wav2vec2.feature_extractor
        # Convレイヤーは学習しない
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        # 2. SpecAug
        self.spec_aug = SpecAug(
            freq_mask_max_length=cfg.model.spec_aug.freq_mask_max_length,
            time_mask_max_length=cfg.model.spec_aug.time_mask_max_length,
            num_freq_mask=cfg.model.spec_aug.num_freq_mask,
            num_time_mask=cfg.model.spec_aug.num_time_mask,
        )
        # 3. Transformerレイヤー
        self.encoder = self.wav2vec2.encoder
        self.hidden_size = 768

        self.fc = torch.nn.Linear(self.hidden_size, self.nlabel, bias=True)
        self.log_softmax = torch.nn.functional.log_softmax

    def forward(self, x, x_lengths):
        # args:
        #   x: [B, T]
        #   x_lengths: [B]
        #       padding前のシーケンス長
        # return:
        #   log_prob: [B, T, nlabel]
        #   y_lengths: [B]
        #       非パディング部分のシーケンス長
        # TODO: with torch.no_grad()とすることで計算グラフの構築を抑制する
        x, x_lengths = self.feature_extractor(x, x_lengths)  # [B, T', D]
        if self.training:
            for i in range(x.shape[0]):
                x[i] = self.spec_aug(x[i])
        xs = self.encoder.extract_features(x, x_lengths)  # (L, B, T', D)
        last_hidden_states = xs[-1]  # (B, T', D)
        y = self.fc(last_hidden_states)  # [B, T', nlabel]
        log_probs = self.log_softmax(y, dim=-1)  # [B, T', nlabel]
        return log_probs, x_lengths
