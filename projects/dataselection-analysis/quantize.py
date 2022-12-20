import pickle

import torch
from data import LibriAdaptUS, get_dataloader
from model import MyWav2Vec2ConformerForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices

from utils import indices2indices


@torch.no_grad()
def quantize(
        bx: torch.Tensor,
        model: MyWav2Vec2ConformerForPreTraining,
        num_groups: int,
        num_codevectors_per_group: int
        ):
    batch_size = bx.shape[0]
    # output length from conv layer
    sequence_length = model._get_feat_extract_output_lengths(bx.shape[1]).item()
    mask_time_indices = _compute_mask_indices(
        shape=(batch_size, sequence_length), mask_prob=0.2, mask_length=2
    )
    sampled_negative_indices = _sample_negative_indices(
        features_shape=(batch_size, sequence_length),
        num_negatives=model.config.num_negatives,
        mask_time_indices=mask_time_indices,
    )
    mask_time_indices = torch.tensor(mask_time_indices, device=bx.device, dtype=torch.long)
    sampled_negative_indices = torch.tensor(
        data=sampled_negative_indices, device=bx.device, dtype=torch.long
    )
    model.eval()
    outputs = model(
        bx, mask_time_indices=mask_time_indices, sampled_negative_indices=sampled_negative_indices
    )
    quantized_indices = indices2indices(outputs.quntized_indices, num_codevectors_per_group)
    # LMにするときは.astype(str)が必須
    quantized_indices = quantized_indices.to("cpu").numpy().reshape(
        quantized_indices.shape[0], quantized_indices.shape[1] * quantized_indices.shape[2])
    return quantized_indices


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = MyWav2Vec2ConformerForPreTraining.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large").to(DEVICE)
    G = model.config.num_codevector_groups
    V = model.config.num_codevectors_per_group

    mic_names = ["matrix", "nexus6", "pseye", "respeaker", "shure", "usb"]
    for mic_name in mic_names:
        quantized_result = {}
        print(f"mic_name: {mic_name}")
        train_dataset = LibriAdaptUS(f"{mic_name}-train")
        train_dataloader = get_dataloader(train_dataset)

        for i, (bfile_name, bx, bx_len, by, by_len) in enumerate(train_dataloader):
            if (i + 1) % 100 == 0:
                print(f"i: {i + 1}")
            bx = bx.to(DEVICE)
            quantized_indices = quantize(bx, model, G, V)
            for file_name, indices in zip(bfile_name, quantized_indices):
                quantized_result[file_name] = indices
            del bx, quantized_indices
            torch.cuda.empty_cache()

        with open(f"{mic_name}_quantized_indices.pkl", "wb") as f:
            pickle.dump(quantized_result, f)


if __name__ == "__main__":
    main()
