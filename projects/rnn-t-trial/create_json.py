import glob
import json
import os

import torchaudio
from tqdm import tqdm


def create_yesno_json(split: str):
    all_wav_file_paths = glob.glob("./datasets/waves_yesno/" + "*.wav")
    sorted(all_wav_file_paths)
    if split == "test":
        wav_file_paths = all_wav_file_paths[:2]
        json_file_path = "./json/yesno_test.json"
    elif split == "dev":
        wav_file_paths = all_wav_file_paths[2:4]
        json_file_path = "./json/yesno_dev.json"
    elif split == "train":
        wav_file_paths = all_wav_file_paths[4:]
        json_file_path = "./json/yesno_train.json"
    else:
        raise ValueError("Invalid split.")

    json_data = {}
    for i, wav_file_path in tqdm(enumerate(wav_file_paths)):
        audio, sampling_rate = torchaudio.load(wav_file_path)
        audio = audio.flatten()
        audio_sec = len(audio) / sampling_rate
        file_name = os.path.splitext(os.path.basename(wav_file_path))[0]
        text = ""
        for c in file_name:
            if c == "1":
                text += "yes"
            elif c == "0":
                text += "no"
            elif c == "_":
                text += " "
            else:
                raise ValueError("Invalid Identifier.")
        json_data[i] = {
            "wav_file_path": wav_file_path,
            "sampling_rate": sampling_rate,
            "audio_sec": audio_sec,
            "raw_transcript": text,
        }
    with open(json_file_path, "w") as f:
        json.dump(json_data, f, indent=2)


def create_librispeech_json(split: str):
    base_dir = "./datasets/librispeech"
    split_to_dataset = {}
    if split == "train-clean-100":
        dataset = torchaudio.datasets.LIBRISPEECH(base_dir, url="train-clean-100")
        split_to_dataset[split] = dataset
    elif split == "train-clean-360":
        dataset = torchaudio.datasets.LIBRISPEECH(base_dir, url="train-clean-360")
        split_to_dataset[split] = dataset
    elif split == "train-other-500":
        dataset = torchaudio.datasets.LIBRISPEECH(base_dir, url="train-other-500")
        split_to_dataset[split] = dataset
    elif split == "train-all-960":
        dataset_train_clean_100 = torchaudio.datasets.LIBRISPEECH(base_dir, url="train-clean-100")
        dataset_train_clean_360 = torchaudio.datasets.LIBRISPEECH(base_dir, url="train-clean-360")
        dataset_train_other_500 = torchaudio.datasets.LIBRISPEECH(base_dir, url="train-other-500")
        split_to_dataset["train-clean-100"] = dataset_train_clean_100
        split_to_dataset["train-clean-360"] = dataset_train_clean_360
        split_to_dataset["train-other-500"] = dataset_train_other_500
    elif split == "dev-clean":
        dataset = torchaudio.datasets.LIBRISPEECH(base_dir, url="dev-clean")
        split_to_dataset[split] = dataset
    elif split == "dev-other":
        dataset = torchaudio.datasets.LIBRISPEECH(base_dir, url="dev-other")
        split_to_dataset[split] = dataset
    elif split == "dev-all":
        dataset_dev_clean = torchaudio.datasets.LIBRISPEECH(base_dir, url="dev-clean")
        dataset_dev_other = torchaudio.datasets.LIBRISPEECH(base_dir, url="dev-other")
        split_to_dataset["dev-clean"] = dataset_dev_clean
        split_to_dataset["dev-other"] = dataset_dev_other
    elif split == "test-clean":
        dataset = torchaudio.datasets.LIBRISPEECH(base_dir, url="test-clean")
        split_to_dataset[split] = dataset
    elif split == "test-other":
        dataset = torchaudio.datasets.LIBRISPEECH(base_dir, url="test-other")
        split_to_dataset[split] = dataset
    elif split == "test-all":
        dataset_test_clean = torchaudio.datasets.LIBRISPEECH(base_dir, url="test-clean")
        dataset_test_other = torchaudio.datasets.LIBRISPEECH(base_dir, url="test-other")
        split_to_dataset["test-clean"] = dataset_test_clean
        split_to_dataset["test-other"] = dataset_test_other
    else:
        raise ValueError("Invalid split")
    json_data = {}
    json_file_path = "./json/librispeech_" + split + ".json"
    count = 0
    for each_split, dataset in split_to_dataset.items():
        for i in tqdm(range(len(dataset))):
            audio, sampling_rate, transcript, speaker_id, chapter_id, utterance_id = dataset[i]
            audio = audio.flatten()
            audio_sec = len(audio) / sampling_rate

            file_name = dataset._walker[i]
            wav_file_path = f"{base_dir}/LibriSpeech/{each_split}/{speaker_id}/{chapter_id}/{file_name}.flac"
            if os.path.exists(wav_file_path) is False:
                raise ValueError("Invalid file path.", wav_file_path)
            json_data[count] = {
                "wav_file_path": wav_file_path,
                "sampling_rate": sampling_rate,
                "audio_sec": audio_sec,
                "raw_transcript": transcript,
            }
            count += 1
    with open(json_file_path, "w") as f:
        json.dump(json_data, f, indent=2)
