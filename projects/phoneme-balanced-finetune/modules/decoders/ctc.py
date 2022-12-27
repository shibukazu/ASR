from typing import Dict, List


def greedy_decoder(hypotheses_idxs, vocab: Dict[str, int], pad: str, separator: str, blank: str) -> List[str]:
    """
    greedy decoder for ctc
    """
    # hypothesis_idxs: tensor(batch, time)
    # vocab: key=char|word, value=index
    index_to_vocab = {v: k for k, v in vocab.items()}
    hypotheses_idxs = hypotheses_idxs.cpu().numpy()
    hypotheses = []
    pad_idx = vocab[pad]
    blank_idx = vocab[blank]
    separator_idx = vocab[separator]

    for hypothesis_idxs in hypotheses_idxs:
        hypothesis = []
        prev_idx = -1
        for idx in hypothesis_idxs:
            if idx == blank_idx:
                continue
            elif idx == prev_idx:
                continue
            elif idx == pad_idx:
                continue
            elif idx == separator_idx:
                hypothesis.append(" ")
                prev_idx = idx
            else:
                hypothesis.append(index_to_vocab[idx])
                prev_idx = idx
        hypotheses.append("".join(hypothesis))
    return hypotheses
