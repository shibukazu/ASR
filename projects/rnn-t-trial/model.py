from typing import List

import torch
from modules.encoder import CausalConformerEncoder, LSTMEncoder
from modules.jointnet import JointNet
from modules.predictor import Predictor


class LSTMModel(torch.nn.Module):
    def __init__(
        self,
        encoder_input_size,
        encoder_hidden_size,
        encoder_num_layers,
        encoder_dropout,
        vocab_size,
        embedding_size,
        predictor_hidden_size,
        predictor_num_layers,
        jointnet_hidden_size,
        blank_idx,
    ):
        super().__init__()

        self.blank_idx = blank_idx

        self.encoder = LSTMEncoder(
            input_size=encoder_input_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_num_layers,
            dropout=encoder_dropout,
        )

        self.predictor = Predictor(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            hidden_size=predictor_hidden_size,
            num_layers=predictor_num_layers,
            blank_idx=blank_idx,
        )

        self.jointnet = JointNet(
            enc_hidden_size=encoder_hidden_size,
            pred_hidden_size=predictor_hidden_size,
            hidden_size=jointnet_hidden_size,
            vocab_size=vocab_size,
        )

    def forward(
        self,
        padded_enc_input,
        padded_pred_input,
        enc_input_lengths,
        pred_input_lengths,
    ):
        padded_enc_output, subsampled_enc_input_lengths = self.encoder(padded_enc_input, enc_input_lengths)
        padded_pred_output, _ = self.predictor(padded_pred_input, pred_input_lengths)
        padded_output = self.jointnet(padded_enc_output, padded_pred_output)

        return padded_output, subsampled_enc_input_lengths

    @torch.no_grad()
    def greedy_inference(self, enc_inputs, enc_input_lengths) -> List[List[int]]:
        # enc_inputs: 3D tensor (batch, seq_len, n_mel)
        # enc_input_lengths: 1D tensor (batch)
        # output: 2D List (batch, hyp_len)
        batch_hyp_tokens = []
        for i, (enc_input, enc_input_length) in enumerate(zip(enc_inputs, enc_input_lengths)):
            if enc_input.size(0) > enc_input_length:
                enc_input = enc_input[:enc_input_length, :]

            enc_output, _ = self.encoder(
                enc_input.unsqueeze(0), torch.tensor([enc_input.size(0)])
            )  # [1, subsampled_enc_input_length, output_size]
            pred_input = torch.tensor([[self.blank_idx]], dtype=torch.int32).to(enc_output.device)
            pred_output, hidden = self.predictor.forward_wo_prepend(pred_input, torch.tensor([1]), hidden=None)
            # [1, 1, output_size]
            timestamp = 0
            hyp_tokens = []
            while timestamp < enc_output.size(1):
                enc_output_at_timestamp = enc_output[0, timestamp]
                logits = self.jointnet(enc_output_at_timestamp.view(1, 1, -1), pred_output)
                pred_token = logits.argmax(dim=-1)
                if pred_token != self.blank_idx:
                    hyp_tokens.append(pred_token.item())
                    pred_input = torch.tensor([[pred_token]], dtype=torch.int32).to(enc_output.device)
                    pred_output, hidden = self.predictor.forward_wo_prepend(
                        pred_input, torch.tensor([1]), hidden=hidden
                    )
                else:
                    timestamp += 1

                if len(hyp_tokens) >= 400:
                    break
            batch_hyp_tokens.append(hyp_tokens)
        return batch_hyp_tokens


class CausalConformerModel(torch.nn.Module):
    def __init__(
        self,
        encoder_input_size,
        encoder_subsampled_input_size,
        encoder_num_conformer_blocks,
        encoder_ff_hidden_size,
        encoder_conv_hidden_size,
        encoder_conv_kernel_size,
        encoder_mha_num_heads,
        encoder_dropout,
        vocab_size,
        embedding_size,
        predictor_hidden_size,
        predictor_num_layers,
        jointnet_hidden_size,
        blank_idx,
    ):
        super().__init__()

        self.blank_idx = blank_idx

        self.encoder = CausalConformerEncoder(
            input_size=encoder_input_size,
            subsampled_input_size=encoder_subsampled_input_size,
            num_conformer_blocks=encoder_num_conformer_blocks,
            ff_hidden_size=encoder_ff_hidden_size,
            conv_hidden_size=encoder_conv_hidden_size,
            conv_kernel_size=encoder_conv_kernel_size,
            mha_num_heads=encoder_mha_num_heads,
            dropout=encoder_dropout,
        )  # [B, subsampled_T, subsampled_D]

        self.predictor = Predictor(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            hidden_size=predictor_hidden_size,
            num_layers=predictor_num_layers,
            blank_idx=blank_idx,
        )

        self.jointnet = JointNet(
            enc_hidden_size=encoder_subsampled_input_size,
            pred_hidden_size=predictor_hidden_size,
            hidden_size=jointnet_hidden_size,
            vocab_size=vocab_size,
        )

    def forward(
        self,
        padded_enc_input,
        padded_pred_input,
        enc_input_lengths,
        pred_input_lengths,
    ):
        padded_enc_output, subsampled_enc_input_lengths = self.encoder(padded_enc_input, enc_input_lengths)
        padded_pred_output, _ = self.predictor(padded_pred_input, pred_input_lengths)
        padded_output = self.jointnet(padded_enc_output, padded_pred_output)

        return padded_output, subsampled_enc_input_lengths

    @torch.no_grad()
    def greedy_inference(self, enc_inputs, enc_input_lengths) -> List[List[int]]:
        # enc_inputs: 3D tensor (batch, seq_len, n_mel)
        # enc_input_lengths: 1D tensor (batch)
        # output: 2D List (batch, hyp_len)
        batch_hyp_tokens = []
        for i, (enc_input, enc_input_length) in enumerate(zip(enc_inputs, enc_input_lengths)):
            if enc_input.size(0) > enc_input_length:
                enc_input = enc_input[:enc_input_length, :]

            enc_output, _ = self.encoder(
                enc_input.unsqueeze(0), torch.tensor([enc_input.size(0)])
            )  # [1, subsampled_enc_input_length, output_size]
            pred_input = torch.tensor([[self.blank_idx]], dtype=torch.int32).to(enc_output.device)
            pred_output, hidden = self.predictor.forward_wo_prepend(pred_input, torch.tensor([1]), hidden=None)
            # [1, 1, output_size]
            timestamp = 0
            hyp_tokens = []
            while timestamp < enc_output.size(1):
                enc_output_at_timestamp = enc_output[0, timestamp]
                logits = self.jointnet(enc_output_at_timestamp.view(1, 1, -1), pred_output)
                pred_token = logits.argmax(dim=-1)
                if pred_token != self.blank_idx:
                    hyp_tokens.append(pred_token.item())
                    pred_input = torch.tensor([[pred_token]], dtype=torch.int32).to(enc_output.device)
                    pred_output, hidden = self.predictor.forward_wo_prepend(
                        pred_input, torch.tensor([1]), hidden=hidden
                    )
                else:
                    timestamp += 1

                if len(hyp_tokens) >= 400:
                    break
            batch_hyp_tokens.append(hyp_tokens)
        return batch_hyp_tokens
