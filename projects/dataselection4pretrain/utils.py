import torch


def indices2indices(batch_indices, codebook_vocab_size):
    """
    得られたquantized_indicesを言語モデルへ入力できるように変換する
    (コードブックごとに語彙を共有しないことから、異なる語彙として扱えるようにインデックスを変換する)
    """
    V = codebook_vocab_size
    converted_indices = torch.zeros_like(batch_indices)
    assert converted_indices.dim() == 3
    for b_idx in range(converted_indices.size(0)):
        for t_idx in range(converted_indices.size(1)):
            for p_idx in range(converted_indices.size(2)):
                converted_indices[b_idx][t_idx][p_idx] = batch_indices[b_idx][t_idx][p_idx] + (p_idx * V)

    return converted_indices
