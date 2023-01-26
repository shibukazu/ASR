import torch
from modules.encoder import Encoder
from modules.jointnet import JointNet
from modules.predictor import Predictor


class Model(torch.nn.Module):
    def __init__(
        self,
        encoder_input_size,
        encoder_subsampled_input_size,
        encoder_hidden_size,
        encoder_num_layers,
        encoder_output_size,
        vocab_size,
        embedding_size,
        predictor_hidden_size,
        predictor_num_layers,
        predictor_output_size,
        jointnet_hidden_size,
        blank_idx,
    ):
        super().__init__()
        self.encoder = Encoder(
            input_size=encoder_input_size,
            subsampled_input_size=encoder_subsampled_input_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_num_layers,
            output_size=encoder_output_size,
        )

        self.predictor = Predictor(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            hidden_size=predictor_hidden_size,
            num_layers=predictor_num_layers,
            output_size=predictor_output_size,
            blank_idx=blank_idx,
        )

        self.jointnet = JointNet(
            enc_out_size=encoder_output_size,
            pred_out_size=predictor_output_size,
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
    def greedy_inference(self, enc_inputs, enc_input_lengths):
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
            pred_input = torch.tensor([[0]], dtype=torch.int32).to(enc_output.device)
            pred_output, hidden = self.predictor.forward_wo_prepend(pred_input, torch.tensor([1]), hidden=None)
            # [1, 1, output_size]
            timestamp = 0
            hyp_tokens = []
            while timestamp < enc_output.size(1):
                enc_output_at_timestamp = enc_output[0, timestamp]
                logits = self.jointnet(enc_output_at_timestamp.view(1, 1, -1), pred_output)
                pred_token = logits.argmax(dim=-1)
                if pred_token != 0:
                    hyp_tokens.append(pred_token.item())
                    pred_input = torch.tensor([[pred_token]], dtype=torch.int32).to(enc_output.device)
                    pred_output, hidden = self.predictor.forward_wo_prepend(
                        pred_input, torch.tensor([1]), hidden=hidden
                    )
                else:
                    timestamp += 1

                if len(hyp_tokens) >= 100:
                    break
            batch_hyp_tokens.append(hyp_tokens)
        return batch_hyp_tokens