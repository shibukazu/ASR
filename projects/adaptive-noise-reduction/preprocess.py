import os
import random

import numpy as np
import pyroomacoustics as pra
import torch
import torchaudio
from tqdm import tqdm


def create_reverberated_data(data_json, folder_name):

    rt60_min = 0.2
    rt60_max = 0.6
    room_len_min = 4
    room_len_max = 8
    room_width_min = 3
    room_width_max = 7
    room_height_min = 2.5
    room_height_max = 4.0
    array_height_min = 1.0
    array_height_max = 1.5
    array_margin = 0.8
    source_height_min = 1.5
    source_height_max = 1.8
    distance_min = 0.8
    distance_max = 2.1

    result_json = {}
    REVERBERATED_WAV_FILE_PATH_PREFIX = f"./datasets/non_concat_csj/reverberated/{folder_name}"

    for key in tqdm(data_json.keys()):
        clean_key = key
        reverberated_key = "reverberated_" + key
        clean_wav, sampling_rate = torchaudio.load(data_json[clean_key]["wav_file_path"])
        clean_wav = clean_wav.flatten().numpy()
        clean_length = clean_wav.shape[0]

        # create room
        rt60_tgt = np.random.uniform(rt60_min, rt60_max)
        room_len = np.random.uniform(room_len_min, room_len_max)
        room_width = np.random.uniform(room_width_min, room_width_max)
        room_height = np.random.uniform(room_height_min, room_height_max)
        room_dim = [room_len, room_width, room_height]
        e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
        room = pra.ShoeBox(
            room_dim,
            fs=sampling_rate,
            materials=pra.Material(e_absorption),
            max_order=max_order,
        )

        # create mic array
        array_height = np.random.uniform(array_height_min, array_height_max)
        array_x = np.random.uniform(array_margin, room_len - array_margin)
        array_y = np.random.uniform(array_margin, room_width - array_margin)
        array_2d_loc = np.array([array_x, array_y])
        mic_locs = pra.circular_2D_array(center=array_2d_loc, M=6, phi0=0, radius=0.1)
        mic_locs = np.vstack(
            (mic_locs, np.array([array_height, array_height, array_height, array_height, array_height, array_height]))
        )
        room.add_microphone_array(mic_locs)

        # put source
        source_height = np.random.uniform(source_height_min, source_height_max)
        distance = np.random.uniform(distance_min, distance_max)
        while True:
            angle = np.random.uniform(0, np.pi)
            # Mimuraさんのコードではnp.sin
            source_x = array_x + distance * np.cos(angle)
            source_y = array_y + distance * np.sin(angle)
            source_loc = np.array([source_x, source_y, source_height])
            if room.is_inside(source_loc):
                break
        room.add_source(source_loc, signal=clean_wav, delay=0.0)

        # simulate
        room.simulate()
        reverberated_wavs = room.mic_array.signals
        channel = np.random.randint(0, 6)

        # trim
        reverberated_wav = reverberated_wavs[channel, :]
        reverberated_wav = reverberated_wav[0:clean_length].astype(np.float32).reshape(1, -1)
        reverberated_wav = torch.from_numpy(reverberated_wav)
        # save
        os.makedirs(REVERBERATED_WAV_FILE_PATH_PREFIX, exist_ok=True)
        reverberated_wav_file_path = os.path.join(REVERBERATED_WAV_FILE_PATH_PREFIX, reverberated_key + ".wav")
        torchaudio.save(filepath=reverberated_wav_file_path, src=reverberated_wav, sample_rate=sampling_rate)

        result_json[reverberated_key] = {}
        result_json[reverberated_key]["wav_file_path"] = reverberated_wav_file_path
        result_json[reverberated_key]["sampling_rate"] = torchaudio.info(reverberated_wav_file_path).sample_rate
        result_json[reverberated_key]["audio_sec"] = (
            torchaudio.info(reverberated_wav_file_path).num_frames
            / torchaudio.info(reverberated_wav_file_path).sample_rate
        )
        result_json[reverberated_key]["raw_transcript"] = data_json[clean_key]["raw_transcript"]

    return result_json


def create_reverberated_data_parallel(data_json, keys, folder_name, queue):

    rt60_min = 0.2
    rt60_max = 0.6
    room_len_min = 4
    room_len_max = 8
    room_width_min = 3
    room_width_max = 7
    room_height_min = 2.5
    room_height_max = 4.0
    array_height_min = 1.0
    array_height_max = 1.5
    array_margin = 0.8
    source_height_min = 1.5
    source_height_max = 1.8
    distance_min = 0.8
    distance_max = 2.1

    result_json = {}
    REVERBERATED_WAV_FILE_PATH_PREFIX = f"./datasets/non_concat_csj/reverberated/{folder_name}"

    for key in tqdm(keys):
        clean_key = key
        reverberated_key = "reverberated_" + key
        clean_wav, sampling_rate = torchaudio.load(data_json[clean_key]["wav_file_path"])
        clean_wav = clean_wav.flatten().numpy()
        clean_length = clean_wav.shape[0]

        # create room
        rt60_tgt = np.random.uniform(rt60_min, rt60_max)
        room_len = np.random.uniform(room_len_min, room_len_max)
        room_width = np.random.uniform(room_width_min, room_width_max)
        room_height = np.random.uniform(room_height_min, room_height_max)
        room_dim = [room_len, room_width, room_height]
        e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
        room = pra.ShoeBox(
            room_dim,
            fs=sampling_rate,
            materials=pra.Material(e_absorption),
            max_order=max_order,
        )

        # create mic array
        array_height = np.random.uniform(array_height_min, array_height_max)
        array_x = np.random.uniform(array_margin, room_len - array_margin)
        array_y = np.random.uniform(array_margin, room_width - array_margin)
        array_2d_loc = np.array([array_x, array_y])
        mic_locs = pra.circular_2D_array(center=array_2d_loc, M=6, phi0=0, radius=0.1)
        mic_locs = np.vstack(
            (mic_locs, np.array([array_height, array_height, array_height, array_height, array_height, array_height]))
        )
        room.add_microphone_array(mic_locs)

        # put source
        source_height = np.random.uniform(source_height_min, source_height_max)
        distance = np.random.uniform(distance_min, distance_max)
        while True:
            angle = np.random.uniform(0, np.pi)
            # Mimuraさんのコードではnp.sin
            source_x = array_x + distance * np.cos(angle)
            source_y = array_y + distance * np.sin(angle)
            source_loc = np.array([source_x, source_y, source_height])
            if room.is_inside(source_loc):
                break
        room.add_source(source_loc, signal=clean_wav, delay=0.0)

        # simulate
        room.simulate()
        reverberated_wavs = room.mic_array.signals
        channel = np.random.randint(0, 6)

        # trim
        reverberated_wav = reverberated_wavs[channel, :]
        reverberated_wav = reverberated_wav[0:clean_length].astype(np.float32).reshape(1, -1)
        reverberated_wav = torch.from_numpy(reverberated_wav)
        # save
        os.makedirs(REVERBERATED_WAV_FILE_PATH_PREFIX, exist_ok=True)
        reverberated_wav_file_path = os.path.join(REVERBERATED_WAV_FILE_PATH_PREFIX, reverberated_key + ".wav")
        torchaudio.save(filepath=reverberated_wav_file_path, src=reverberated_wav, sample_rate=sampling_rate)

        result_json[reverberated_key] = {}
        result_json[reverberated_key]["wav_file_path"] = reverberated_wav_file_path
        result_json[reverberated_key]["sampling_rate"] = torchaudio.info(reverberated_wav_file_path).sample_rate
        result_json[reverberated_key]["audio_sec"] = (
            torchaudio.info(reverberated_wav_file_path).num_frames
            / torchaudio.info(reverberated_wav_file_path).sample_rate
        )
        result_json[reverberated_key]["raw_transcript"] = data_json[clean_key]["raw_transcript"]

    queue.put(result_json)
    return


def create_noisy_data(
    data_json,
    folder_name,
    noise_data_jsons,
):
    """
    以下の式に基づいてノイズの振幅を調整することで所望のSN比を実現するような加算を行う
    まず、SN比は以下のように計算される
    SN比 = 20log10(RMS_s/RMS_n)
    そこで、所望のSN比をSN比_dとすると、変換後のRMS_n'は以下のようになる
    RMS_n' = RMS_s / 10^(SN比_d/20)
    ここで、RMS_n/RMS_n' = AMP_n/AMP_n'が成立するため、以下のように振幅を変換する
    AMP_n' = AMP_n * RMS_n' / RMS_n
    これを加算することで、所望のSN比のノイズが加算された信号を得ることができる
    """

    NOISE_ADDED_WAV_FILE_PATH_PREFIX = f"./datasets/non_concat_csj/noise_added/{folder_name}"

    MIN_SNR = 0
    MAX_SNR = 10

    result_json = {}

    for key in tqdm(data_json.keys()):
        clean_key = key
        noise_added_key = "noisy_" + key

        clean_wav, sampling_rate = torchaudio.load(data_json[clean_key]["wav_file_path"])
        clean_wav = clean_wav.flatten().numpy()
        clean_length = clean_wav.shape[0]

        while True:
            noise_data_json = random.choice(noise_data_jsons)
            noise_key = random.choice(list(noise_data_json.keys()))
            if os.path.getsize(noise_data_json[noise_key]["wav_file_path"]) > os.path.getsize(
                data_json[clean_key]["wav_file_path"]
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
            np.abs(20 * np.log10(np.sqrt(np.mean(clean_wav**2)) / np.sqrt(np.mean(noise_wav**2))) - snr_d) < 1e-3
        )

        # add noise
        noise_added_wav = clean_wav + noise_wav
        noise_added_wav = noise_added_wav.astype(np.float32).reshape(1, -1)
        noise_added_wav = torch.from_numpy(noise_added_wav)

        # save
        os.makedirs(NOISE_ADDED_WAV_FILE_PATH_PREFIX, exist_ok=True)
        noise_added_wav_file_path = os.path.join(NOISE_ADDED_WAV_FILE_PATH_PREFIX, noise_added_key + ".wav")
        torchaudio.save(filepath=noise_added_wav_file_path, src=noise_added_wav, sample_rate=sampling_rate)

        result_json[noise_added_key] = {}
        result_json[noise_added_key]["wav_file_path"] = noise_added_wav_file_path
        result_json[noise_added_key]["sampling_rate"] = torchaudio.info(noise_added_wav_file_path).sample_rate
        result_json[noise_added_key]["audio_sec"] = (
            torchaudio.info(noise_added_wav_file_path).num_frames
            / torchaudio.info(noise_added_wav_file_path).sample_rate
        )
        result_json[noise_added_key]["raw_transcript"] = data_json[clean_key]["raw_transcript"]
        result_json[noise_added_key]["metainfo"] = {}
        result_json[noise_added_key]["metainfo"]["reverberated_wav_file_path"] = data_json[clean_key]["wav_file_path"]
        result_json[noise_added_key]["metainfo"]["noise_wav_file_path"] = noise_data_json[noise_key]["wav_file_path"]
        result_json[noise_added_key]["metainfo"]["snr_d"] = snr_d

    return result_json


def create_noisy_data_parallel(
    data_json,
    keys,
    folder_name,
    noise_data_jsons,
    queue,
):
    """
    以下の式に基づいてノイズの振幅を調整することで所望のSN比を実現するような加算を行う
    まず、SN比は以下のように計算される
    SN比 = 20log10(RMS_s/RMS_n)
    そこで、所望のSN比をSN比_dとすると、変換後のRMS_n'は以下のようになる
    RMS_n' = RMS_s / 10^(SN比_d/20)
    ここで、RMS_n/RMS_n' = AMP_n/AMP_n'が成立するため、以下のように振幅を変換する
    AMP_n' = AMP_n * RMS_n' / RMS_n
    これを加算することで、所望のSN比のノイズが加算された信号を得ることができる
    """

    NOISE_ADDED_WAV_FILE_PATH_PREFIX = f"./datasets/non_concat_csj/noise_added/{folder_name}"

    MIN_SNR = 0
    MAX_SNR = 10

    result_json = {}

    for key in tqdm(keys):
        clean_key = key
        noise_added_key = "noisy_" + key

        clean_wav, sampling_rate = torchaudio.load(data_json[clean_key]["wav_file_path"])
        clean_wav = clean_wav.flatten().numpy()
        clean_length = clean_wav.shape[0]

        while True:
            noise_data_json = random.choice(noise_data_jsons)
            noise_key = random.choice(list(noise_data_json.keys()))
            if os.path.getsize(noise_data_json[noise_key]["wav_file_path"]) > os.path.getsize(
                data_json[clean_key]["wav_file_path"]
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
            np.abs(20 * np.log10(np.sqrt(np.mean(clean_wav**2)) / np.sqrt(np.mean(noise_wav**2))) - snr_d) < 1e-3
        )

        # add noise
        noise_added_wav = clean_wav + noise_wav
        noise_added_wav = noise_added_wav.astype(np.float32).reshape(1, -1)
        noise_added_wav = torch.from_numpy(noise_added_wav)

        # save
        os.makedirs(NOISE_ADDED_WAV_FILE_PATH_PREFIX, exist_ok=True)
        noise_added_wav_file_path = os.path.join(NOISE_ADDED_WAV_FILE_PATH_PREFIX, noise_added_key + ".wav")
        torchaudio.save(filepath=noise_added_wav_file_path, src=noise_added_wav, sample_rate=sampling_rate)

        result_json[noise_added_key] = {}
        result_json[noise_added_key]["wav_file_path"] = noise_added_wav_file_path
        result_json[noise_added_key]["sampling_rate"] = torchaudio.info(noise_added_wav_file_path).sample_rate
        result_json[noise_added_key]["audio_sec"] = (
            torchaudio.info(noise_added_wav_file_path).num_frames
            / torchaudio.info(noise_added_wav_file_path).sample_rate
        )
        result_json[noise_added_key]["raw_transcript"] = data_json[clean_key]["raw_transcript"]
        result_json[noise_added_key]["metainfo"] = {}
        result_json[noise_added_key]["metainfo"]["reverberated_wav_file_path"] = data_json[clean_key]["wav_file_path"]
        result_json[noise_added_key]["metainfo"]["noise_wav_file_path"] = noise_data_json[noise_key]["wav_file_path"]
        result_json[noise_added_key]["metainfo"]["snr_d"] = snr_d

    queue.put(result_json)
    return


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
