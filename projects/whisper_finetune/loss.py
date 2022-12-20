import torch


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor, target_len: torch.Tensor, NUM_CLASSES: int):
        """
        return sum of cross entropy loss
        x: (batch, SOS_len + seq_len, num_classes)
        target: (batch, seq_len)
        target_len: (batch)
        """
        padded_target_len = target.size(1)
        # p(X) = 1の計算
        one_hot_target = torch.nn.functional.one_hot(target, num_classes=NUM_CLASSES).float()  # [B, T, C]
        # log q(x)の計算
        log_q_x = torch.nn.functional.log_softmax(x[:, -padded_target_len:, :], dim=-1)  # [B, T, C]
        # log q(x) * p(x)の計算
        log_q_x_p_x = log_q_x * one_hot_target  # [B, T, C]
        # padding部分のマスクケイン
        mask = torch.arange(padded_target_len).expand(
            len(target_len), padded_target_len).to(
            target_len.device
            ) < target_len.unsqueeze(1)  # [B, T]
        mask = mask.unsqueeze(-1)  # [B, T, 1]
        # log q(x) * p(x)のpadding部分を0マスクすることでロス和の計算に寄与させない
        log_q_x_p_x = log_q_x_p_x.masked_fill(~mask, 0)  # [B, T, C]
        # ロス和の計算
        loss = -log_q_x_p_x.sum()

        return loss
