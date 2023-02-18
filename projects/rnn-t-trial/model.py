from typing import List

import torch
import torchaudio
from modules.encoder import CausalConformerEncoder, LSTMEncoder, TorchAudioConformerEncoder
from modules.jointnet import JointNet
from modules.predictor import Predictor
from tqdm import tqdm


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

        self.rnnt_loss = torchaudio.transforms.RNNTLoss(blank=blank_idx, reduction="sum")
        self.ctc_loss = torch.nn.CTCLoss(reduction="sum", blank=self.blank_idx)

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
        encoder_subsampling_kernel_size1,
        encoder_subsampling_stride1,
        encoder_subsampling_kernel_size2,
        encoder_subsampling_stride2,
        encoder_num_previous_frames,
        vocab_size,
        embedding_size,
        predictor_hidden_size,
        predictor_num_layers,
        jointnet_hidden_size,
        blank_idx,
        decoder_buffer_size,
    ):
        super().__init__()

        self.blank_idx = blank_idx
        self.decoder_buffer_size = decoder_buffer_size
        self.encoder_num_previous_frames = encoder_num_previous_frames

        self.encoder = CausalConformerEncoder(
            input_size=encoder_input_size,
            subsampled_input_size=encoder_subsampled_input_size,
            num_conformer_blocks=encoder_num_conformer_blocks,
            ff_hidden_size=encoder_ff_hidden_size,
            conv_hidden_size=encoder_conv_hidden_size,
            conv_kernel_size=encoder_conv_kernel_size,
            mha_num_heads=encoder_mha_num_heads,
            dropout=encoder_dropout,
            subsampling_kernel_size1=encoder_subsampling_kernel_size1,
            subsampling_stride1=encoder_subsampling_stride1,
            subsampling_kernel_size2=encoder_subsampling_kernel_size2,
            subsampling_stride2=encoder_subsampling_stride2,
            num_previous_frames=encoder_num_previous_frames,
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

        self.ctc_ff = torch.nn.Linear(encoder_subsampled_input_size, vocab_size)

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

        # For CTC loss
        padded_ctc_logits = self.ctc_ff(padded_enc_output)
        padded_ctc_log_probs = torch.nn.functional.log_softmax(padded_ctc_logits, dim=-1)

        return padded_output, padded_ctc_log_probs, subsampled_enc_input_lengths

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

    @torch.no_grad()
    def streaming_greedy_inference(self, enc_inputs, enc_input_lengths):
        # enc_inputs: 3D tensor (batch, seq_len, n_mel)
        # enc_input_lengths: 1D tensor (batch)
        # output: 2D List (batch, hyp_len)
        batch_hyp_token_indices = []
        BUFFER_SIZE = self.decoder_buffer_size
        NUM_PREVIOUS_FRAMES = self.encoder_num_previous_frames
        for enc_input, enc_input_length in zip(enc_inputs, enc_input_lengths):
            if enc_input.size(0) > enc_input_length:
                enc_input = enc_input[:enc_input_length, :]
            hyp_token_indices = []
            buffer = []
            pred_input = torch.tensor([[self.blank_idx]], dtype=torch.int32).to(enc_input.device)
            pred_output, hidden = self.predictor.forward_wo_prepend(pred_input, torch.tensor([1]), hidden=None)
            for i in tqdm(range(5, enc_input.shape[0])):
                if NUM_PREVIOUS_FRAMES == "all":
                    buffer.append(enc_input[: i + 1])
                else:
                    buffer.append(enc_input[max(i + 1 - NUM_PREVIOUS_FRAMES, 0) : i + 1])
                if len(buffer) == BUFFER_SIZE:
                    batch = torch.nn.utils.rnn.pad_sequence(buffer, batch_first=True, padding_value=0)
                    batch_lengths = torch.tensor([len(x) for x in buffer])
                    buffer = []
                    batch_enc_output, batch_subsampled_length = self.encoder(batch, batch_lengths)
                    for j in range(len(batch_enc_output)):
                        subsampled_length = batch_subsampled_length[j]
                        # NOTE: JointNetは線形層を通しているだけであり、時刻に関して独立->現在のenc_outだけで十分
                        enc_output = batch_enc_output[j][subsampled_length - 1].view(1, 1, -1)
                        num_token_indices = 0
                        while True:
                            logits = self.jointnet(enc_output, pred_output)[0]
                            pred_token_idx = torch.argmax(logits, dim=-1)
                            if pred_token_idx == self.blank_idx:
                                break
                            else:
                                num_token_indices += 1
                                hyp_token_indices.append(pred_token_idx.item())
                                pred_input = torch.tensor([[pred_token_idx]], dtype=torch.int32).to(enc_input.device)
                                pred_output, hidden = self.predictor.forward_wo_prepend(
                                    pred_input, torch.tensor([1]), hidden=hidden
                                )

                            if num_token_indices >= 5:
                                break
            batch_hyp_token_indices.append(hyp_token_indices)
        return batch_hyp_token_indices


class TorchAudioConformerModel(torch.nn.Module):
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
        encoder_subsampling_kernel_size1,
        encoder_subsampling_stride1,
        encoder_subsampling_kernel_size2,
        encoder_subsampling_stride2,
        encoder_num_previous_frames,
        vocab_size,
        embedding_size,
        predictor_hidden_size,
        predictor_num_layers,
        jointnet_hidden_size,
        blank_idx,
        decoder_buffer_size,
    ):
        super().__init__()

        self.blank_idx = blank_idx
        self.decoder_buffer_size = decoder_buffer_size
        self.encoder_num_previous_frames = encoder_num_previous_frames

        self.encoder = TorchAudioConformerEncoder(
            input_size=encoder_input_size,
            subsampled_input_size=encoder_subsampled_input_size,
            num_conformer_blocks=encoder_num_conformer_blocks,
            ff_hidden_size=encoder_ff_hidden_size,
            conv_hidden_size=encoder_conv_hidden_size,
            conv_kernel_size=encoder_conv_kernel_size,
            mha_num_heads=encoder_mha_num_heads,
            dropout=encoder_dropout,
            subsampling_kernel_size1=encoder_subsampling_kernel_size1,
            subsampling_stride1=encoder_subsampling_stride1,
            subsampling_kernel_size2=encoder_subsampling_kernel_size2,
            subsampling_stride2=encoder_subsampling_stride2,
            num_previous_frames=encoder_num_previous_frames,
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

        self.ctc_ff = torch.nn.Linear(encoder_subsampled_input_size, vocab_size)

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

        # For CTC loss
        padded_ctc_logits = self.ctc_ff(padded_enc_output)
        padded_ctc_log_probs = torch.nn.functional.log_softmax(padded_ctc_logits, dim=-1)

        return padded_output, padded_ctc_log_probs, subsampled_enc_input_lengths

    @torch.no_grad()
    def greedy_inference(self, enc_inputs, enc_input_lengths) -> List[List[int]]:
        # enc_inputs: 3D tensor (batch, seq_len, n_mel)
        # enc_input_lengths: 1D tensor (batch)
        # output: 2D List (batch, hyp_len)
        batch_hyp_tokens = []
        enc_outputs, subsampled_input_lengths = self.encoder(enc_inputs, enc_input_lengths)
        for i, (enc_input, enc_input_length) in enumerate(zip(enc_inputs, enc_input_lengths)):
            enc_output = enc_outputs[i, : subsampled_input_lengths[i], :]
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

    @torch.no_grad()
    def streaming_greedy_inference(self, enc_inputs, enc_input_lengths):
        # enc_inputs: 3D tensor (batch, seq_len, n_mel)
        # enc_input_lengths: 1D tensor (batch)
        # output: 2D List (batch, hyp_len)
        batch_hyp_token_indices = []
        BUFFER_SIZE = self.decoder_buffer_size
        NUM_PREVIOUS_FRAMES = self.encoder_num_previous_frames
        for enc_input, enc_input_length in zip(enc_inputs, enc_input_lengths):
            if enc_input.size(0) > enc_input_length:
                enc_input = enc_input[:enc_input_length, :]
            hyp_token_indices = []
            buffer = []
            pred_input = torch.tensor([[self.blank_idx]], dtype=torch.int32).to(enc_input.device)
            pred_output, hidden = self.predictor.forward_wo_prepend(pred_input, torch.tensor([1]), hidden=None)
            for i in tqdm(range(5, enc_input.shape[0])):
                if NUM_PREVIOUS_FRAMES == "all":
                    buffer.append(enc_input[: i + 1])
                else:
                    buffer.append(enc_input[max(i + 1 - NUM_PREVIOUS_FRAMES, 0) : i + 1])
                if len(buffer) == BUFFER_SIZE:
                    batch = torch.nn.utils.rnn.pad_sequence(buffer, batch_first=True, padding_value=0)
                    batch_lengths = torch.tensor([len(x) for x in buffer])
                    buffer = []
                    batch_enc_output, batch_subsampled_length = self.encoder(batch, batch_lengths)
                    for j in range(len(batch_enc_output)):
                        subsampled_length = batch_subsampled_length[j]
                        # NOTE: JointNetは線形層を通しているだけであり、時刻に関して独立->現在のenc_outだけで十分
                        enc_output = batch_enc_output[j][subsampled_length - 1].view(1, 1, -1)
                        num_token_indices = 0
                        while True:
                            logits = self.jointnet(enc_output, pred_output)[0]
                            pred_token_idx = torch.argmax(logits, dim=-1)
                            if pred_token_idx == self.blank_idx:
                                break
                            else:
                                num_token_indices += 1
                                hyp_token_indices.append(pred_token_idx.item())
                                pred_input = torch.tensor([[pred_token_idx]], dtype=torch.int32).to(enc_input.device)
                                pred_output, hidden = self.predictor.forward_wo_prepend(
                                    pred_input, torch.tensor([1]), hidden=hidden
                                )

                            if num_token_indices >= 5:
                                break
            batch_hyp_token_indices.append(hyp_token_indices)
        return batch_hyp_token_indices
