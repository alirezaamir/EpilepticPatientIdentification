import json
import ast


def text2json(filename):
    with open(filename) as f:
        content = f.readlines()

    patient_list_start = []
    patient_list_end = []
    for idx, l in enumerate(content):
        if l.startswith('['):
            patient_list_start.append(idx)
        elif l.startswith('orig'):
            patient_list_end.append(idx)

    exp = []
    for start, end in zip(patient_list_start, patient_list_end):
        patient_list = []
        for patient_line in range(start,end):
            patient_list.append(content[patient_line].strip())
        as_list = ast.literal_eval(''.join(patient_list))

        result_line = content[end]
        orig_end_index = result_line.find(',')
        orig = float(result_line[10:orig_end_index])
        synt = float(result_line[orig_end_index + 13: -1])

        exp.append({'patients' : as_list,
                    'orig': orig,
                    'synt': synt})

    return exp


def main ():
    filename = '../results/results_normalized2.txt'
    exp = text2json(filename)
    with open('../results/final_results2.json', 'w') as outfile:
        json.dump(exp, outfile)


if __name__ == '__main__':
    main()
