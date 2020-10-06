import json
import ast

filename = '../results/final_results.txt'
with open(filename) as f:
    content = f.readlines()

patient_list_start = []
patient_list_end = []
for idx, l in enumerate(content):
    if l.startswith('['):
        patient_list_start.append(idx)
    elif l.startswith('orig'):
        patient_list_end.append(idx)

for start, end in zip(patient_list_start, patient_list_end):
    patient_list = []
    for patient_line in range(start,end):
        patient_list.append(content[patient_line].strip())
    as_list = ast.literal_eval(''.join(patient_list))
    print(as_list)


