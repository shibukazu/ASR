import json
import multiprocessing
import os

import torch
import torchaudio
from tqdm import tqdm


def create_concatenated_data_parallel(data_json, keys, NAME, queue):
    speech_key = keys[0].split("_")[0]
    concatenate_keys = []
    concatenate_audio_sec = 0
    previous_end_msec = 0
    result_json = {}
    MAX_AUDIO_SEC = 30

    def concatenate_audio():
        concatenate_start_msec = concatenate_keys[0].split("_")[1]
        concatenate_end_msec = concatenate_keys[-1].split("_")[2]
        concatenate_audios = []
        concatenate_wav_file_paths = []
        concatenate_raw_transcript = ""
        for i, key in enumerate(concatenate_keys):
            wav_file_path = data_json[key]["wav_file_path"]
            wav, sr = torchaudio.load(wav_file_path)
            wav = wav[0]
            concatenate_audios.append(wav)

            concatenate_raw_transcript += data_json[key]["raw_transcript"]
            concatenate_wav_file_paths.append(wav_file_path)

            if i != len(concatenate_keys) - 1:
                # 次のstart_msecまでの間にゼロパディング
                end_msec = int(key.split("_")[2])
                next_start_msec = int(concatenate_keys[i + 1].split("_")[1])
                zero_padding_sec = (next_start_msec - end_msec) / 1000
                zero_padding_len = int(zero_padding_sec * sr)
                zero_padding = torch.zeros(zero_padding_len, dtype=wav.dtype)
                concatenate_audios.append(zero_padding)
        concatenate_audio = torch.cat(concatenate_audios)
        concatenate_audio_sec = concatenate_audio.shape[0] / sr
        concatenate_wav_file_path = (
            f"datasets/csj/concatenated/{NAME}/{speech_key}_{concatenate_start_msec}_{concatenate_end_msec}.wav"
        )
        torchaudio.save(concatenate_wav_file_path, concatenate_audio.reshape(1, -1), sr)
        result_json[f"{speech_key}_{concatenate_start_msec}_{concatenate_end_msec}"] = {
            "wav_file_path": concatenate_wav_file_path,
            "sampling_rate": sr,
            "audio_sec": concatenate_audio_sec,
            "raw_transcript": concatenate_raw_transcript,
            "metainfo": {"concatenated_wav_file_paths": concatenate_wav_file_paths},
        }

    for i, key in tqdm(enumerate(keys)):
        audio_sec = data_json[key]["audio_sec"]
        if concatenate_audio_sec + audio_sec > MAX_AUDIO_SEC or key.split("_")[0] != speech_key:
            # これまでの結合音声を書き出す
            concatenate_audio()
            speech_key = key.split("_")[0]
            concatenate_keys = []
            concatenate_audio_sec = 0

        concatenate_keys.append(key)
        if concatenate_audio_sec == 0:
            do_zero_padding = False
        else:
            do_zero_padding = True
        concatenate_audio_sec += audio_sec
        if do_zero_padding:
            zero_padding_sec = (int(key.split("_")[1]) - previous_end_msec) / 1000
            concatenate_audio_sec += zero_padding_sec
        previous_end_msec = int(key.split("_")[2])

    if len(concatenate_keys) > 0:
        concatenate_audio()

    queue.put(result_json)


# create concat json

NAME = "csj_train_nodup_sp"
NUM_PROCS = 20
os.makedirs(f"datasets/csj/concatenated/{NAME}", exist_ok=True)
with open(f"json/{NAME}.json", "r") as f:
    data_json = json.load(f)
all_keys = list(data_json.keys())

jobs = []
queue = multiprocessing.Queue()
for i in range(NUM_PROCS):
    start = int(len(all_keys) / NUM_PROCS * i)
    end = int(len(all_keys) / NUM_PROCS * (i + 1))
    if i == NUM_PROCS - 1:
        end = len(all_keys)
    keys = all_keys[start:end]
    p = multiprocessing.Process(target=create_concatenated_data_parallel, args=(data_json, keys, NAME, queue))
    p.start()
    jobs.append(p)

result_json = {}
for i in range(NUM_PROCS):
    result_json.update(queue.get())

for p in jobs:
    p.join()

# count num in metadata col
num = 0
for key in result_json.keys():
    num += len(result_json[key]["metainfo"]["concatenated_wav_file_paths"])

assert num == len(data_json)

with open(f"json/concat_{NAME}.json", "w") as f:
    json.dump(result_json, f, indent=4, ensure_ascii=False)
