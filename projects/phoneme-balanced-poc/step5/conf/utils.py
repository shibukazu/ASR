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
