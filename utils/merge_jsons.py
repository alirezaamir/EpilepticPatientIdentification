import json
from utils.results_to_json import text2json
import numpy as np

base_file = open("../results/final_results.json", 'r')
base_json = json.load(base_file)
first_exp = text2json("../results/results_normalized1.txt")
second_exp = text2json("../results/results_normalized2.txt")

print("Base: {}, First: {}, Second: {}".format(len(base_json), len(first_exp), len(second_exp)))

final_file = open("../results/final_normalized.json", "w")
final_list = []
for i in range(len(first_exp)):
    exp = base_json[i]
    exp['orig'] = np.round(100 * exp['orig'], 1)
    exp['synt'] = np.round(100 * exp['synt'], 1)
    if first_exp[i]['patients'] != exp['patients']:
        print("Not compatible")
        break
    exp['norm_orig'] = np.round(100 * first_exp[i]['orig'], 1)
    exp['norm_synt'] = np.round(100 * first_exp[i]['synt'], 1)
    final_list.append(exp)

for i in range(len(second_exp)):
    exp = base_json[i+len(first_exp)]
    exp['orig'] = np.round(100 * exp['orig'], 1)
    exp['synt'] = np.round(100 * exp['synt'], 1)
    exp['norm_orig'] = np.round(100 * second_exp[i]['orig'], 1)
    exp['norm_synt'] = np.round(100 * second_exp[i]['synt'], 1)
    final_list.append(exp)

json.dump(final_list, final_file)
final_file.close()

