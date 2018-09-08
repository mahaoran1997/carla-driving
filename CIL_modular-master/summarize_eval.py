import json, sys, os
import numpy as np

if __name__ == "__main__":
    exp_id = sys.argv[1]

    for town in ["Town01", "Town02"]:
        print(town)
        path = "/scratch/yang/aws_data/CIL_modular_data/_benchmarks_results/" + exp_id + "_YangExp_" + town + "/metrics.json"

        if not os.path.exists(path):
            print("not ready yet")
            continue

        obj=json.load(open(path, "r"))
        t = obj['episodes_fully_completed']

        weathers = {"train": ["1.0", "10.0"],
                    "val": ["13.0", "14.0"]}
        for phase in ["train", "val"]:
            for weather in weathers[phase]:
                for task in range(len(t[weather])):
                    perf = np.mean(t[weather][task])
                    try:
                        nsample = len(t[weather][task])
                    except TypeError:
                        nsample = 0

                    print(phase,
                          " weather ", weather,
                          " task ", task,
                          " average performance ", perf,
                          "(num sample %d)" % nsample)
