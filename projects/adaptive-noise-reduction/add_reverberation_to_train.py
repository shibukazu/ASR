# 残響の付加(train)
import json
import multiprocessing

from preprocess import create_reverberated_data_parallel

NUM_PROCS = 32

NAME = "csj_train_nodup_sp"
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
    p = multiprocessing.Process(target=create_reverberated_data_parallel, args=(data_json, keys, NAME, queue))
    p.start()
    jobs.append(p)

# concat result_jsons in queue
result_json = {}
for i in range(NUM_PROCS):
    result_json.update(queue.get())

for p in jobs:
    p.join()

assert len(result_json) == len(data_json)

with open(f"json/reverberated_{NAME}.json", "w") as f:
    json.dump(result_json, f, indent=4, ensure_ascii=False)
