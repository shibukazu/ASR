import os
import random

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


def create_aligned_noisy_pretrain_data_parallel(
    data_json,
    speakers,
    wav_folder_path,
    noise_data_jsons,
    queue,
):
    NOISE_ADDED_WAV_FILE_PATH_PREFIX = wav_folder_path

    MIN_SNR = 0
    MAX_SNR = 10

    result_json = {}

    for speaker in tqdm(speakers):
        keys = list(data_json[speaker].keys())
        for key in keys:
            clean_key = key
            noise_added_key = key

            clean_wav, sampling_rate = torchaudio.load(data_json[speaker][clean_key]["wav_file_path"])
            clean_wav = clean_wav.flatten().numpy()
            clean_length = clean_wav.shape[0]

            while True:
                noise_data_json = random.choice(noise_data_jsons)
                noise_key = random.choice(list(noise_data_json.keys()))
                if os.path.getsize(noise_data_json[noise_key]["wav_file_path"]) > os.path.getsize(
                    data_json[speaker][clean_key]["wav_file_path"]
                ):
                    break

            noise_wav, sampling_rate = torchaudio.load(noise_data_json[noise_key]["wav_file_path"])
            noise_wav = noise_wav.flatten().numpy()
            noise_length = noise_wav.shape[0]

            # trim noise
            start = np.random.randint(0, noise_length - clean_length)
            noise_wav = noise_wav[start : start + clean_length]

            # calculate RMS
            clean_rms = np.sqrt(np.mean(clean_wav**2))
            noise_rms = np.sqrt(np.mean(noise_wav**2))

            # adjust noise amplitude
            snr_d = np.random.uniform(MIN_SNR, MAX_SNR)
            noise_rms_d = clean_rms / (10 ** (snr_d / 20))
            noise_wav = noise_wav * noise_rms_d / noise_rms

            # check SNR
            assert (
                np.abs(20 * np.log10(np.sqrt(np.mean(clean_wav**2)) / np.sqrt(np.mean(noise_wav**2))) - snr_d)
                < 1e-3
            )

            # add noise
            noise_added_wav = clean_wav + noise_wav
            noise_added_wav = noise_added_wav.astype(np.float32).reshape(1, -1)
            noise_added_wav = torch.from_numpy(noise_added_wav)

            # save
            os.makedirs(NOISE_ADDED_WAV_FILE_PATH_PREFIX, exist_ok=True)
            noise_added_wav_file_path = os.path.join(NOISE_ADDED_WAV_FILE_PATH_PREFIX, noise_added_key + ".wav")
            torchaudio.save(filepath=noise_added_wav_file_path, src=noise_added_wav, sample_rate=sampling_rate)

            if speaker not in result_json:
                result_json[speaker] = {}

            result_json[speaker][noise_added_key] = {}
            result_json[speaker][noise_added_key]["wav_file_path"] = noise_added_wav_file_path
            result_json[speaker][noise_added_key]["sampling_rate"] = torchaudio.info(
                noise_added_wav_file_path
            ).sample_rate
            result_json[speaker][noise_added_key]["audio_sec"] = (
                torchaudio.info(noise_added_wav_file_path).num_frames
                / torchaudio.info(noise_added_wav_file_path).sample_rate
            )
            result_json[speaker][noise_added_key]["raw_transcript"] = data_json[speaker][clean_key]["raw_transcript"]
            result_json[speaker][noise_added_key]["vad"] = data_json[speaker][clean_key]["vad"]
            result_json[speaker][noise_added_key]["metainfo"] = {}
            result_json[speaker][noise_added_key]["metainfo"]["original_wav_file_path"] = data_json[speaker][
                clean_key
            ]["wav_file_path"]
            result_json[speaker][noise_added_key]["metainfo"]["noise_wav_file_path"] = noise_data_json[noise_key][
                "wav_file_path"
            ]
            result_json[speaker][noise_added_key]["metainfo"]["snr_d"] = snr_d

    queue.put(result_json)
    return


def create_aligned_noisy_pretrain_eval_data_parallel(
    data_json,
    speakers,
    wav_folder_path,
    noise_data_jsons,
    queue,
):
    NOISE_ADDED_WAV_FILE_PATH_PREFIX = wav_folder_path

    MIN_SNR = 0
    MAX_SNR = 10

    result_json = {}
    counter = 0

    is_choose_noise = True

    for speaker in tqdm(speakers):
        keys = list(data_json[speaker].keys())

        for key_idx, key in enumerate(keys):
            if is_choose_noise:
                # ノイズデータおよび開始インデックスを選択する
                while True:
                    noise_data_json = random.choice(noise_data_jsons)
                    noise_key = random.choice(list(noise_data_json.keys()))
                    noise_wav, noise_sampling_rate = torchaudio.load(noise_data_json[noise_key]["wav_file_path"])
                    noise_wav = noise_wav.flatten().numpy()
                    if noise_wav.shape[0] / noise_sampling_rate > 60:
                        break
                # DEMANDは300sなため、少なくとも数発話は含まれるようにする
                noise_start_sec = np.random.uniform(0, noise_wav.shape[0] / noise_sampling_rate - 60)
                noise_start_idx = int(noise_start_sec * noise_sampling_rate)

                is_choose_noise = False
                noise_added_key_prefix = f"{speaker}-{counter}"
                counter += 1
                # SN比を決定する(話者+ノイズの組内では同一のSN比を維持する)
                snr_d = np.random.uniform(MIN_SNR, MAX_SNR)

            clean_key = key
            noise_added_key = noise_added_key_prefix + "-" + key

            clean_wav, clean_sampling_rate = torchaudio.load(data_json[speaker][clean_key]["wav_file_path"])
            assert clean_sampling_rate == noise_sampling_rate
            clean_wav = clean_wav.flatten().numpy()
            clean_length = clean_wav.shape[0]

            # trim noise
            trimmed_noise_wav = noise_wav[noise_start_idx : noise_start_idx + clean_length]
            noise_start_idx = noise_start_idx + clean_length

            # 次のタイミングでノイズを再選択する必要があるか確認する
            next_clean_key = keys[key_idx + 1] if key_idx + 1 < len(keys) else None
            if next_clean_key is not None:
                next_clean_wav, _ = torchaudio.load(data_json[speaker][next_clean_key]["wav_file_path"])
                next_clean_wav = next_clean_wav.flatten().numpy()
                next_clean_length = next_clean_wav.shape[0]
                if noise_start_idx + next_clean_length > noise_wav.shape[0]:
                    is_choose_noise = True
            else:
                # speaker間で引き継がない
                is_choose_noise = True

            # calculate RMS
            clean_rms = np.sqrt(np.mean(clean_wav**2))
            trimmed_noise_rms = np.sqrt(np.mean(trimmed_noise_wav**2))

            # adjust noise amplitude
            trimmed_noise_rms_d = clean_rms / (10 ** (snr_d / 20))
            trimmed_noise_wav = trimmed_noise_wav * trimmed_noise_rms_d / trimmed_noise_rms

            # check SNR
            assert (
                np.abs(
                    20 * np.log10(np.sqrt(np.mean(clean_wav**2)) / np.sqrt(np.mean(trimmed_noise_wav**2))) - snr_d
                )
                < 1e-3
            )

            # add noise
            noise_added_wav = clean_wav + trimmed_noise_wav
            noise_added_wav = noise_added_wav.astype(np.float32).reshape(1, -1)
            noise_added_wav = torch.from_numpy(noise_added_wav)

            # save
            os.makedirs(NOISE_ADDED_WAV_FILE_PATH_PREFIX, exist_ok=True)
            # 同一のファイル名が存在しないことを確認する
            assert not os.path.exists(os.path.join(NOISE_ADDED_WAV_FILE_PATH_PREFIX, noise_added_key + ".wav"))
            noise_added_wav_file_path = os.path.join(NOISE_ADDED_WAV_FILE_PATH_PREFIX, noise_added_key + ".wav")
            torchaudio.save(filepath=noise_added_wav_file_path, src=noise_added_wav, sample_rate=clean_sampling_rate)

            if speaker not in result_json:
                result_json[speaker] = {}
            # 同一のnoise_added_keyが存在しないことを確認する
            assert noise_added_key not in result_json[speaker]

            result_json[speaker][noise_added_key] = {}
            result_json[speaker][noise_added_key]["wav_file_path"] = noise_added_wav_file_path
            result_json[speaker][noise_added_key]["sampling_rate"] = torchaudio.info(
                noise_added_wav_file_path
            ).sample_rate
            result_json[speaker][noise_added_key]["audio_sec"] = (
                torchaudio.info(noise_added_wav_file_path).num_frames
                / torchaudio.info(noise_added_wav_file_path).sample_rate
            )
            result_json[speaker][noise_added_key]["raw_transcript"] = data_json[speaker][clean_key]["raw_transcript"]
            result_json[speaker][noise_added_key]["vad"] = data_json[speaker][clean_key]["vad"]
            result_json[speaker][noise_added_key]["metainfo"] = {}
            result_json[speaker][noise_added_key]["metainfo"]["original_wav_file_path"] = data_json[speaker][
                clean_key
            ]["wav_file_path"]
            result_json[speaker][noise_added_key]["metainfo"]["noise_wav_file_path"] = noise_data_json[noise_key][
                "wav_file_path"
            ]
            result_json[speaker][noise_added_key]["metainfo"]["snr_d"] = snr_d

    queue.put(result_json)
    return


def concat_pretrain_eval_with_subsampled_vad_parallel(
    data_json,
    speakers,
    wav_folder_path,
    queue,
):
    WAV_FILE_PATH_PREFIX = wav_folder_path

    result_json = {}

    for speaker in tqdm(speakers):
        keys = list(data_json[speaker].keys())
        for key in keys:
            identifier = key.split("-")[0] + "-" + key.split("-")[1]  # speaker + noise idx
            if identifier not in result_json:
                result_json[identifier] = {}
                result_json[identifier]["sampling_rate"] = data_json[speaker][key]["sampling_rate"]
                result_json[identifier]["audio_sec"] = 0
                result_json[identifier]["raw_transcript"] = ""
                result_json[identifier]["vad"] = []
                result_json[identifier]["subsampled_vad"] = []
                result_json[identifier]["metainfo"] = {}
                result_json[identifier]["metainfo"]["original_wav_file_paths"] = []
                result_json[identifier]["metainfo"]["noise_wav_file_path"] = data_json[speaker][key]["metainfo"][
                    "noise_wav_file_path"
                ]
                result_json[identifier]["metainfo"]["snr_d"] = data_json[speaker][key]["metainfo"]["snr_d"]

            result_json[identifier]["audio_sec"] += data_json[speaker][key]["audio_sec"]
            result_json[identifier]["raw_transcript"] += data_json[speaker][key]["raw_transcript"]
            result_json[identifier]["vad"] += data_json[speaker][key]["vad"]
            result_json[identifier]["subsampled_vad"] += data_json[speaker][key]["subsampled_vad"]
            result_json[identifier]["metainfo"]["original_wav_file_paths"].append(
                data_json[speaker][key]["wav_file_path"]
            )
            assert result_json[identifier]["sampling_rate"] == data_json[speaker][key]["sampling_rate"]
            assert (
                abs(result_json[identifier]["metainfo"]["snr_d"] - data_json[speaker][key]["metainfo"]["snr_d"]) < 1e-3
            )
            assert (
                result_json[identifier]["metainfo"]["noise_wav_file_path"]
                == data_json[speaker][key]["metainfo"]["noise_wav_file_path"]
            )

    identifiers = list(result_json.keys())

    for identifier in tqdm(identifiers):
        # concat all wav files in original_wav_file_paths
        wav_file_paths = result_json[identifier]["metainfo"]["original_wav_file_paths"]
        wav, sr = torchaudio.load(wav_file_paths[0])
        for wav_file_path in wav_file_paths[1:]:
            wav_, sr_ = torchaudio.load(wav_file_path)
            wav = torch.cat([wav, wav_], dim=1)
        assert sr == sr_
        assert sr == result_json[identifier]["sampling_rate"]

        # save
        os.makedirs(WAV_FILE_PATH_PREFIX, exist_ok=True)
        # 同一のファイル名が存在しないことを確認する
        assert not os.path.exists(os.path.join(WAV_FILE_PATH_PREFIX, identifier + ".wav"))
        torchaudio.save(filepath=os.path.join(WAV_FILE_PATH_PREFIX, identifier + ".wav"), src=wav, sample_rate=sr)
        result_json[identifier]["wav_file_path"] = os.path.join(WAV_FILE_PATH_PREFIX, identifier + ".wav")

    queue.put(result_json)
    return


def array_concat_pretrain_eval_with_subsampled_vad_parallel(
    data_json,
    speakers,
    queue,
):
    result_json = {}

    for speaker in tqdm(speakers):
        keys = list(data_json[speaker].keys())
        for key in keys:
            identifier = key.split("-")[0] + "-" + key.split("-")[1]  # speaker + noise idx
            if identifier not in result_json:
                result_json[identifier] = {}
                result_json[identifier]["sampling_rate"] = data_json[speaker][key]["sampling_rate"]
                result_json[identifier]["audio_sec"] = 0
                result_json[identifier]["wav_file_paths"] = []
                result_json[identifier]["raw_transcripts"] = []
                result_json[identifier]["vads"] = []
                result_json[identifier]["subsampled_vads"] = []
                result_json[identifier]["metainfos"] = []

            result_json[identifier]["audio_sec"] += data_json[speaker][key]["audio_sec"]
            result_json[identifier]["wav_file_paths"].append(data_json[speaker][key]["wav_file_path"])
            result_json[identifier]["raw_transcripts"].append(data_json[speaker][key]["raw_transcript"])
            result_json[identifier]["vads"].append(data_json[speaker][key]["vad"])
            result_json[identifier]["subsampled_vads"].append(data_json[speaker][key]["subsampled_vad"])
            metainfo = {
                "noise_wav_file_path": data_json[speaker][key]["metainfo"]["noise_wav_file_path"],
                "snr_d": data_json[speaker][key]["metainfo"]["snr_d"],
            }
            result_json[identifier]["metainfos"].append(metainfo)

            assert result_json[identifier]["sampling_rate"] == data_json[speaker][key]["sampling_rate"]
            assert abs(metainfo["snr_d"] - data_json[speaker][key]["metainfo"]["snr_d"]) < 1e-3
            assert metainfo["noise_wav_file_path"] == data_json[speaker][key]["metainfo"]["noise_wav_file_path"]

    queue.put(result_json)
    return
