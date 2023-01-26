import torch

def greedy_decode(
    hypotheses: torch.tensor,
    blank_idx: int,
    pad_idx: int):
    

