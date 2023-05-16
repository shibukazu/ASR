import torch
from modules.encoder import CausalConformerAdapterEncoder, CausalConformerEncoder
from tqdm import tqdm


class CausalConformerMultitaskCTCModel(torch.nn.Module):
    def __init__(
        self,
        input_size,
        subsampled_input_size,
        num_conformer_blocks,
        ff_hidden_size,
        conv_hidden_size,
        conv_kernel_size,
        mha_num_heads,
        dropout,
        subsampling_kernel_size1,
        subsampling_stride1,
        subsampling_kernel_size2,
        subsampling_stride2,
        num_previous_frames,
        is_timewise_ln,
        vocab_size,
        blank_idx,
    ):
        super().__init__()

        self.encoder = CausalConformerEncoder(
            input_size=input_size,
            subsampled_input_size=subsampled_input_size,
            num_conformer_blocks=num_conformer_blocks,
            ff_hidden_size=ff_hidden_size,
            conv_hidden_size=conv_hidden_size,
            conv_kernel_size=conv_kernel_size,
            mha_num_heads=mha_num_heads,
            dropout=dropout,
            subsampling_kernel_size1=subsampling_kernel_size1,
            subsampling_stride1=subsampling_stride1,
            subsampling_kernel_size2=subsampling_kernel_size2,
            subsampling_stride2=subsampling_stride2,
            num_previous_frames=num_previous_frames,
            is_timewise_ln=is_timewise_ln,
        )

        self.ctc_ff = torch.nn.Linear(subsampled_input_size, vocab_size)
        self.ctc_log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.vad_ff = torch.nn.Linear(subsampled_input_size, 1)
        self.vad_sigmoid = torch.nn.Sigmoid()

        self.blank_idx = blank_idx
        self.num_previous_frames = num_previous_frames

    def forward(self, bx, bx_len):
        bx, bsubsampled_x_len = self.encoder(bx, bx_len)
        bctc_logits = self.ctc_ff(bx)
        bctc_log_probs = self.ctc_log_softmax(bctc_logits)
        bsubsampled_vad_logits = self.vad_ff(bx)
        bsubsampled_vad_probs = self.vad_sigmoid(bsubsampled_vad_logits)

        return bctc_log_probs, bsubsampled_x_len, bsubsampled_vad_probs

    def greedy_inference(self, bx, bx_len):
        bctc_log_probs, bsubsampled_x_len, _ = self.forward(bx, bx_len)
        batch_hyp_token_idxs = []
        for batch_idx in range(bx.shape[0]):
            hyp_token_idxs = []
            prev_token_idx = -1
            for i in range(bsubsampled_x_len[batch_idx]):
                hyp_ctc_token_idx = torch.argmax(bctc_log_probs[batch_idx, i, :], dim=-1)
                if hyp_ctc_token_idx == self.blank_idx:
                    continue
                elif hyp_ctc_token_idx == prev_token_idx:
                    continue
                else:
                    hyp_token_idxs.append(hyp_ctc_token_idx.item())
                    prev_token_idx = hyp_ctc_token_idx
            batch_hyp_token_idxs.append(hyp_token_idxs)
        return batch_hyp_token_idxs

    def streaming_greedy_inference(self, bx, bx_len, num_previous_frames):
        batch_hyp_token_idxs = []
        BUFFER_SIZE = 200  # BUFFER_SIZEフレームごとに推論する
        NUM_PREVIOUS_FRAMES = num_previous_frames

        for batch_idx in range(bx.shape[0]):
            hyp_token_idxs = []
            prev_token_idx = -1
            x_len = bx_len[batch_idx]
            x = bx[batch_idx, :x_len, :]  # (T, D)
            buffer = []
            for i in tqdm(range(5, x_len)):
                if NUM_PREVIOUS_FRAMES == "all":
                    buffer.append(x[: i + 1])
                else:
                    buffer.append(x[max(i - NUM_PREVIOUS_FRAMES, 0) : i + 1])
                if len(buffer) == BUFFER_SIZE or i == x_len - 1:
                    buffer_bx = torch.nn.utils.rnn.pad_sequence(buffer, batch_first=True, padding_value=0.0).to(
                        bx.device
                    )
                    buffer_bx_len = torch.tensor([len(b) for b in buffer])
                    buffer_bctc_log_probs, buffer_bsubsampled_x_len, _ = self.forward(buffer_bx, buffer_bx_len)
                    for j in range(len(buffer)):
                        hyp_ctc_token_idx = torch.argmax(
                            buffer_bctc_log_probs[j, buffer_bsubsampled_x_len[j] - 1, :], dim=-1
                        )  # 現在の時刻の推論結果を知りたいだけなので、最後の時刻のみを見る
                        if hyp_ctc_token_idx == prev_token_idx:
                            # このときはスキップ
                            continue
                        elif hyp_ctc_token_idx == self.blank_idx:
                            # このときはprev_token_idxのみ更新
                            # これによって、同一トークンの連続が許容される
                            prev_token_idx = hyp_ctc_token_idx
                            continue
                        else:
                            hyp_token_idxs.append(hyp_ctc_token_idx.item())
                            prev_token_idx = hyp_ctc_token_idx
                    buffer = []
            batch_hyp_token_idxs.append(hyp_token_idxs)
        return batch_hyp_token_idxs


class CausalConformerMultitaskCTCAdapterModel(torch.nn.Module):
    def __init__(
        self,
        input_size,
        subsampled_input_size,
        num_conformer_blocks,
        ff_hidden_size,
        conv_hidden_size,
        conv_kernel_size,
        mha_num_heads,
        dropout,
        subsampling_kernel_size1,
        subsampling_stride1,
        subsampling_kernel_size2,
        subsampling_stride2,
        num_previous_frames,
        is_timewise_ln,
        vocab_size,
        blank_idx,
        adapter_hidden_size,
        num_adapter_blocks,
    ):
        super().__init__()

        self.encoder = CausalConformerAdapterEncoder(
            input_size=input_size,
            subsampled_input_size=subsampled_input_size,
            num_conformer_blocks=num_conformer_blocks,
            ff_hidden_size=ff_hidden_size,
            conv_hidden_size=conv_hidden_size,
            conv_kernel_size=conv_kernel_size,
            mha_num_heads=mha_num_heads,
            dropout=dropout,
            subsampling_kernel_size1=subsampling_kernel_size1,
            subsampling_stride1=subsampling_stride1,
            subsampling_kernel_size2=subsampling_kernel_size2,
            subsampling_stride2=subsampling_stride2,
            num_previous_frames=num_previous_frames,
            is_timewise_ln=is_timewise_ln,
            adapter_hidden_size=adapter_hidden_size,
            num_adapter_blocks=num_adapter_blocks,
        )

        self.ctc_ff = torch.nn.Linear(subsampled_input_size, vocab_size)
        self.ctc_log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.vad_ff = torch.nn.Linear(subsampled_input_size, 1)
        self.vad_sigmoid = torch.nn.Sigmoid()

        self.blank_idx = blank_idx
        self.num_previous_frames = num_previous_frames

    def forward(self, bx, bx_len):
        bx, bsubsampled_x_len = self.encoder(bx, bx_len)
        bctc_logits = self.ctc_ff(bx)
        bctc_log_probs = self.ctc_log_softmax(bctc_logits)
        bsubsampled_vad_logits = self.vad_ff(bx)
        bsubsampled_vad_probs = self.vad_sigmoid(bsubsampled_vad_logits)

        return bctc_log_probs, bsubsampled_x_len, bsubsampled_vad_probs

    def greedy_inference(self, bx, bx_len):
        bctc_log_probs, bsubsampled_x_len, _ = self.forward(bx, bx_len)
        batch_hyp_token_idxs = []
        for batch_idx in range(bx.shape[0]):
            hyp_token_idxs = []
            prev_token_idx = -1
            for i in range(bsubsampled_x_len[batch_idx]):
                hyp_ctc_token_idx = torch.argmax(bctc_log_probs[batch_idx, i, :], dim=-1)
                if hyp_ctc_token_idx == self.blank_idx:
                    continue
                elif hyp_ctc_token_idx == prev_token_idx:
                    continue
                else:
                    hyp_token_idxs.append(hyp_ctc_token_idx.item())
                    prev_token_idx = hyp_ctc_token_idx
            batch_hyp_token_idxs.append(hyp_token_idxs)
        return batch_hyp_token_idxs

    def streaming_greedy_inference(self, bx, bx_len, num_previous_frames):
        batch_hyp_token_idxs = []
        BUFFER_SIZE = 500  # BUFFER_SIZEフレームごとに推論する
        NUM_PREVIOUS_FRAMES = num_previous_frames

        for batch_idx in range(bx.shape[0]):
            hyp_token_idxs = []
            prev_token_idx = -1
            x_len = bx_len[batch_idx]
            x = bx[batch_idx, :x_len, :]  # (T, D)
            buffer = []
            for i in tqdm(range(5, x_len)):
                if NUM_PREVIOUS_FRAMES == "all":
                    buffer.append(x[: i + 1])
                else:
                    buffer.append(x[max(i - NUM_PREVIOUS_FRAMES, 0) : i + 1])
                if len(buffer) == BUFFER_SIZE or i == x_len - 1:
                    buffer_bx = torch.nn.utils.rnn.pad_sequence(buffer, batch_first=True, padding_value=0.0).to(
                        bx.device
                    )
                    buffer_bx_len = torch.tensor([len(b) for b in buffer])
                    buffer_bctc_log_probs, buffer_bsubsampled_x_len, _ = self.forward(buffer_bx, buffer_bx_len)
                    for j in range(len(buffer)):
                        hyp_ctc_token_idx = torch.argmax(
                            buffer_bctc_log_probs[j, buffer_bsubsampled_x_len[j] - 1, :], dim=-1
                        )  # 現在の時刻の推論結果を知りたいだけなので、最後の時刻のみを見る
                        if hyp_ctc_token_idx == prev_token_idx:
                            # このときはスキップ
                            continue
                        elif hyp_ctc_token_idx == self.blank_idx:
                            # このときはprev_token_idxのみ更新
                            # これによって、同一トークンの連続が許容される
                            prev_token_idx = hyp_ctc_token_idx
                            continue
                        else:
                            hyp_token_idxs.append(hyp_ctc_token_idx.item())
                            prev_token_idx = hyp_ctc_token_idx
                    buffer = []
            batch_hyp_token_idxs.append(hyp_token_idxs)
        return batch_hyp_token_idxs
