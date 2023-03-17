# ノイズの追加 (train)
import json
import multiprocessing

from noise_mixer import create_noisy_data_parallel

NUM_PROCS = 32

with open("json/chime3.json", "r") as f:
    chime3_data_json = json.load(f)
with open("json/musan.json", "r") as f:
    musan_data_json = json.load(f)
with open("json/demand.json", "r") as f:
    demand_data_json = json.load(f)

noise_data_jsons = [chime3_data_json, musan_data_json, demand_data_json]

NAME = "csj_train_nodup_sp"
with open(f"json/reverberated_{NAME}.json", "r") as f:
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
    p = multiprocessing.Process(
        target=create_noisy_data_parallel, args=(data_json, keys, NAME, noise_data_jsons, queue)
    )
    p.start()
    jobs.append(p)

# concat result_jsons in queue
result_json = {}
for i in range(NUM_PROCS):
    result_json.update(queue.get())

for p in jobs:
    p.join()

assert len(result_json) == len(data_json)

with open(f"json/noisy_{NAME}.json", "w") as f:
    json.dump(result_json, f, indent=4, ensure_ascii=False)
