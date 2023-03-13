from typing import List

import torch
from tokenizer import SentencePieceTokenizer


class LMHypotheis:
    def __init__(self, hyp: List[int], next_input: torch.Tensor, hidden, score):
        self.hyp = hyp
        self.next_input = next_input
        self.hidden = hidden
        self.score = score


class LMBeamSearch:
    def __init__(
        self,
        beam_size: int,
        max_length: int,
        tokenizer: SentencePieceTokenizer,
        scorer,
    ):
        self.beam_size = beam_size
        self.max_length = max_length
        self.scorer = scorer
        self.tokenizer = tokenizer

    def forward(
        self,
        prompt: torch.Tensor,  # [T]
    ):
        initial_hypothesis = LMHypotheis(prompt.tolist(), prompt, None, 0)
        hypotheses = [initial_hypothesis]
        next_hypotheses = []
        length = prompt.shape[0]
        ended_hypotheses = []
        while length < self.max_length:
            for hypothesis in hypotheses:
                hyp, next_input, hidden, score = (
                    hypothesis.hyp,
                    hypothesis.next_input,
                    hypothesis.hidden,
                    hypothesis.score,
                )
                output, hidden = self.scorer.score(next_input, hidden)  # [1, T, num_tokens]
                output = output[0, -1, :]
                topk = torch.topk(output, self.beam_size)
                for i in range(self.beam_size):
                    new_next_input = topk.indices[i]
                    new_hyp = hyp + [new_next_input.item()]
                    new_score = score + topk.values[i].item()
                    new_hypothesis = LMHypotheis(new_hyp, new_next_input, hidden, new_score)
                    next_hypotheses.append(new_hypothesis)
            next_hypotheses = sorted(next_hypotheses, key=lambda x: x.score, reverse=True)[
                : min(self.beam_size, len(next_hypotheses))
            ]
            if len(next_hypotheses) == 0:
                break
            print(f"length: {length}, {self.tokenizer.token_ids_to_text(next_hypotheses[0].hyp)}", end="\r")
            next_hypotheses, ended_hypotheses = self.post_process(next_hypotheses, ended_hypotheses)
            hypotheses = next_hypotheses
            next_hypotheses = []
            length += 1

        nbest_hypotheses = sorted(ended_hypotheses, key=lambda x: x.score, reverse=True)[
            : min(self.beam_size, len(ended_hypotheses))
        ]
        return nbest_hypotheses

    def post_process(self, next_hypotheses, ended_hypotheses):
        remained_next_hypotheses = []
        for hypothesis in next_hypotheses:
            if hypothesis.next_input == self.tokenizer.eos_token_id:
                ended_hypotheses.append(hypothesis)
            else:
                remained_next_hypotheses.append(hypothesis)
        return remained_next_hypotheses, ended_hypotheses

class RNNTHypotheis:
    def __init__(self, hyp: List[int], next_input: torch.Tensor, hidden, score):
        self.hyp = hyp
        self.next_input = next_input
        self.hidden = hidden
        self.score = score

