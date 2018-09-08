import json, sys, os
import numpy as np
import glob
import csv



if __name__ == "__main__":
    exp_id = sys.argv[1]

    for town in ["Town01", "Town02"]:
        print(town)
        path = "/scratch/yang/aws_data/CIL_modular_data/_benchmarks_results/" + exp_id + "_*" + "_YangExp3cam_" + town + "/summary.csv"

        results = {}
        for item in glob.glob(path):
            with open(item) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    line_count += 1
                    if line_count == 1:
                        continue

                    weather = int(row[0])
                    success = int(row[2])
                    if weather in results:
                        results[weather].append(success)
                    else:
                        results[weather] = [success]

        for key in sorted(results.keys()):
            print("weather %d, success rate %f, num sample %d" % (key, np.mean(results[key]), len(results[key])) )


