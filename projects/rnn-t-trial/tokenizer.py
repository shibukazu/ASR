import os
from typing import List

import sentencepiece as spm


class SentencePieceTokenizer:
    def __init__(self, model_file_path: str):
        # if char tokenizer, use model_file_path="*_char.model"
        # if bpe tokenizer, use model_file_path="*_bpe.model"
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"{model_file_path} does not exist")
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_file_path)

    @property
    def num_tokens(self):
        # this includes idx for <s>, </s>, <unk>, <blank>, <pad>
        return self.sp.GetPieceSize()

    @property
    def blank_token_id(self):
        return self.sp.PieceToId("<blank>")

    @property
    def eos_token_id(self):
        return self.sp.PieceToId("</s>")

    @property
    def pad_token_id(self):
        return self.sp.PieceToId("<pad>")

    def _normalize_text(self, text: str):
        return text.lower()

    def token_to_token_id(self, token: str):
        return self.sp.PieceToId(token)

    def token_id_to_token(self, token_id: int):
        return self.sp.IdToPiece(token_id)

    def text_to_token_ids(self, text: str):
        # text: [T]
        return self.sp.EncodeAsIds(self._normalize_text(text))

    def batch_text_to_token_ids(self, texts: List[str]):
        # texts: [B, T]
        return [self.text_to_token_ids(text) for text in texts]

    def token_ids_to_text(self, token_ids: List[int]):
        # token_ids: [T]
        return self.sp.DecodeIds(token_ids)

    def batch_token_ids_to_text(self, token_ids: List[List[int]]):
        # token_ids: [B, T]
        return [self.token_ids_to_text(token_ids) for token_ids in token_ids]

    @staticmethod
    def create_model(
        transcription_file_path: str,
        model_prefix: str,
        num_tokens: int,
        model_type: str,
        character_coverage: float = 1.0,
    ):
        spm.SentencePieceTrainer.Train(
            f"--input={transcription_file_path} --model_prefix={model_prefix}"
            + f" --vocab_size={num_tokens} --character_coverage={character_coverage}"
            + f" --model_type={model_type} --control_symbols=<blank> --unk_id=1 --bos_id=2 --eos_id=3 --pad_id=4"
        )
