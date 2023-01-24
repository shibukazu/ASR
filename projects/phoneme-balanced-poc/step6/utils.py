import numpy as np
import torch


def trigram_phoneme_indices2unique_idx(trigram_phoneme_indices, num_phonemes):
    """Converts a trigram of phoneme indices to a unique index.
    Args:
        trigram_phoneme_indices (list): A list of 3 phoneme indices.
        num_phonemes (int): Number of phonemes.
    Returns:
        unique_index (int): A unique index.
    """
    return (
        trigram_phoneme_indices[0] * num_phonemes**2
        + trigram_phoneme_indices[1] * num_phonemes
        + trigram_phoneme_indices[2]
    )


def calculate_tf_idf_over_ds(dataset: torch.utils.data.Dataset):
    target_phones = set(
        [
            "aa",
            "ae",
            "ah",
            "aw",
            "ay",
            "b",
            "ch",
            "d",
            "dh",
            "dx",
            "eh",
            "axr",
            "ey",
            "f",
            "g",
            "bcl",
            "hh",
            "ih",
            "iy",
            "jh",
            "k",
            "el",
            "em",
            "en",
            "eng",
            "ow",
            "oy",
            "p",
            "r",
            "s",
            "sh",
            "t",
            "th",
            "uh",
            "uw",
            "v",
            "w",
            "y",
            "z",
        ]
    )
    phone_to_idx = {phone: idx for idx, phone in enumerate(target_phones)}
    df = np.zeros(39, dtype=np.float32)
    tf = np.zeros(39, dtype=np.float32)

    for idx in range(len(dataset)):
        phones = dataset[idx][-1]
        unique_phones = set(phones)
        for target_phone in list(target_phones):
            if target_phone in unique_phones:
                df[phone_to_idx[target_phone]] += 1
        for phone in phones:
            if phone is not None:
                tf[phone_to_idx[phone]] += 1

    tf = tf / tf.sum()

    df = df / len(dataset)
    df += 1e-8
    idf = np.log(1 / df)

    return tf * idf


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
