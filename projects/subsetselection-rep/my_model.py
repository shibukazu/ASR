import torch
from torchaudio import pipelines


class Model(torch.nn.Module):
    def __init__(self, nlabel):
        super(Model, self).__init__()
        bundle = pipelines.WAV2VEC2_BASE
        self.model_sample_rate = bundle.sample_rate
        self.in_size = bundle._params["encoder_embed_dim"]
        self.nlabel = nlabel
        self.wav2vec_encoder = bundle.get_model()
        self.fc = torch.nn.Linear(self.in_size, self.nlabel, bias=True)
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
        encoded, y_lengths = self.wav2vec_encoder.extract_features(x, x_lengths)  # encoded: [L, B, T, in_size]

        y = self.fc(encoded[-1])  # [B, T', nlabel]

        log_probs = self.log_softmax(y, dim=2)  # [B, T', nlabel]
        return log_probs, y_lengths
