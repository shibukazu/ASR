import torch
from modules.preprocessing.subsampling import Conv2DSubSampling
from modules.transformers.encoder import TransformerEncoder
from omegaconf import DictConfig


class Model(torch.nn.Module):
    def __init__(self, nlabel, cfg: DictConfig):
        print("Model: Transformer + CTC Moel")
        super(Model, self).__init__()
        self.nlabel = nlabel
        # out_sizeがSelf-Attentionの入力次元
        self.conv2d_sub_sampling = Conv2DSubSampling(
            in_size=cfg.model.input_feature_size,
            out_size=cfg.model.subsampling.output_feature_size,
            kernel_size1=cfg.model.subsampling.kernel1_size,
            kernel_size2=cfg.model.subsampling.kernel2_size,
            stride1=cfg.model.subsampling.num_stride1,
            stride2=cfg.model.subsampling.num_stride2,
        )
        # in_hidden_sizeがFFの次元
        self.transformer_encoder = TransformerEncoder(
            in_size=cfg.model.subsampling.output_feature_size,
            nlayer=cfg.model.transformer.num_layer,
            nhead=cfg.model.transformer.num_head,
            in_hidden_size=cfg.model.transformer.hidden_feature_size,
            dropout=cfg.model.transformer.dropout,
            norm_first=True,
        )
        self.fc = torch.nn.Linear(cfg.model.subsampling.output_feature_size, nlabel, bias=True)
        self.log_softmax = torch.nn.functional.log_softmax

    def forward(self, x, x_lengths):
        # args:
        #   x: [B, T, in_size]
        #   x_lengths: [B]
        #       padding前のシーケンス長
        # return:
        #   log_prob: [B, T, nlabel]
        #   y_lengths: [B]
        #       非パディング部分のシーケンス長
        subsampled_x, subsampled_x_length = self.conv2d_sub_sampling(x, x_lengths)
        encoded, encoded_inner = self.transformer_encoder(
            subsampled_x, subsampled_x_length
        )  # [B, T', subsampled_in_size]
        y = self.fc(encoded)  # [B, T', nlabel]
        y_lengths = subsampled_x_length
        log_probs = self.log_softmax(y, dim=-1)  # [B, T', nlabel]
        return log_probs, y_lengths
