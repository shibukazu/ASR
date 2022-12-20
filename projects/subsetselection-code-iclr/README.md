

# Towards Representative Subset Selection for Self-Supervised Speech Recognition

This repository is the official implementation of [Towards Representative Subset Selection for Self-Supervised Speech Recognition](https://iclr.cc). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

📋  The datasets (TIMIT, Librispeech, LJSpeech) are used via HuggingFace datasets library and will be downloaded automatically during training/evaluation.

## Training

To compute the training word error rates (WER) which can be used for pruning later on, run this command:

```train
python compute_training_wer.py
```

📋  Three different datasets are supported currently: TIMIT, LJSpeech and Librispeech 10h. To change the dataset, modify the `selected_dataset` in `compute_training_wer.py`.

## Pruning and Evaluation

To prune data through a particular subset selection strategy and fine-tune wav2vec on data subset, run this commmand:

```eval
python subset_selection.py
```

📋  Three different datasets are supported currently: TIMIT, LJSpeech and Librispeech 10h. To change the dataset, modify the `selected_dataset` in `subset_selection.py`. In addition to this, `EPOCH` (WER selection epoch) and `SELECTED_STRATEGY` (pruning strategy) can also be changed in the same file.

## Contributing

📋  All contributions welcome! All content in this repository is licensed under the MIT license.

