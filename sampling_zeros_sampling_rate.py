import json
import csv
import numpy as np

train_data = './training_sample_cate5.json'
csv_file = r'D:\Download\Kim and Bansal (2023)_TR-C\data_NHTSint_indiv.csv'

with open(train_data) as f:
    data = json.load(f)
print(data[:10])
print(len(data))

tuple_data = []
for item in data:
    tuple_data.append(tuple(item))
print(len(set(tuple_data)))

saved_voca = './models/DDPM_category5/voca_map.json'

reverse_voc_map = {}
with open(saved_voca) as f:
    voca_map = json.load(f)

    for key in voca_map.keys():
        reverse_voc_map[voca_map[key]] = key

print(reverse_voc_map)

all_data = []
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    count = 0

    idx_map = {}
    for row in reader:
        if count == 0:
            # gene_pop_dict[sheet_row] = row
            col_count = 0
            for item in row:
                idx_map[col_count] = item
                # excel.cell(sheet_row, col_count+1).value = item

                # sheet.write(sheet_row, col_count, item)
                col_count += 1

            # sheet_row += 1

            count = count + 1
        else:
            one_sample = []
            for j in range(len(row)):
                tem_key = f'{idx_map[j]}:{row[j]}'

                one_sample.append(int(reverse_voc_map[tem_key]))
            all_data.append(tuple(one_sample))

            count = count + 1

print(all_data[:10])

print(len(set(all_data)))
print(len(all_data))

ratio_popu_count = 0
for item in all_data:
    if item in tuple_data:
        ratio_popu_count = ratio_popu_count + 1

print(ratio_popu_count)
print(ratio_popu_count / len(all_data))

np.random.seed(456)
total_sample = []
for sr in np.linspace(0.1, 1, 10):
    print(sr)

    one_sample = []
    for item in all_data:
        r = np.random.uniform(low=0.0, high=1.0)
        if r < sr:
            one_sample.append(item)

    print(f'{sr} sampled combinations: {len(set(one_sample))/264005.0}')

    ratio_popu_count = 0
    for item in all_data:
        if item in one_sample:
            ratio_popu_count = ratio_popu_count + 1

    print(f'{sr} sampled instances: {ratio_popu_count}')
    print(f'{sr} sampled instances ratio: {ratio_popu_count / len(all_data)}')
