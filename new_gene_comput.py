import json
import csv
import torch
from modules import Net_model
from models import Diffusion

train_data = './training_sample_cate5.json'

with open(train_data) as f:
    data = json.load(f)
print(data[:10])
print(len(data))

tuple_data = []
for item in data:
    tuple_data.append(tuple(item))
print(len(set(tuple_data)))

num_h_sample = len(data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# diffusion = Diffusion()
diffusion = Diffusion(noise_steps=1000, beta_start=1e-4, beta_end=0.02, channels=13, voc_size=70, device='cuda')

model = Net_model(c_in=13, c_out=13, time_dim=512, voca_size=70, device="cuda").to(device)

ckpt = torch.load("./models/DDPM_category5/ckpt_671.pt")
model.load_state_dict(ckpt)

saved_voca = './models/DDPM_category5/voca_map.json'

reverse_voc_map = {}
with open(saved_voca) as f:
    voca_map = json.load(f)

    for key in voca_map.keys():
        reverse_voc_map[voca_map[key]] = key

print(reverse_voc_map)
# print(voca_map)


csv_file = r'D:\Download\Kim and Bansal (2023)_TR-C\data_NHTSint_indiv.csv'
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

model.eval()
for i in range(20, 22, 2):
    num_gene = i*num_h_sample
    if i == 20:
        num_gene = 1066319

    k = num_gene // 300
    r = num_gene % 300

    file_json = open(f"./million_genes_ddpm_category5_{i+2}_times_samples.json", "w")

    gene_population = []
    gene_pop_dict = {}
    sheet_row = 1

    for ii in range(k):
        # sub_file_json = open(f"./gene_category5_1/sub_genes_{ii}.json", "w")
        # sub_count = 0
        # sub_dict = {}

        # out = diffusion.sample(model, 300)
        out = diffusion.sample_acc(model, 300)

        sample_out = out.detach().cpu().numpy()
        sample_out = sample_out.tolist()
        for item in sample_out:
            # sub_dict[sub_count] = item
            # sub_count = sub_count + 1

            gene_pop_dict[sheet_row] = item
            gene_population.append(tuple(item))
            # for j in range(len(item)):
            #     # sheet.write(sheet_row, j, item[j])
            #     excel.cell(sheet_row, j + 1).value = item[j]
            sheet_row = sheet_row + 1
        # json.dump(sub_dict, sub_file_json)
        # sub_file_json.close()

    # out = diffusion.sample(model, r)
    out = diffusion.sample_acc(model, r)

    sample_out = out.detach().cpu().numpy()
    sample_out = sample_out.tolist()

    for item in sample_out:
        # sub_dict[sub_count] = item
        # sub_count = sub_count + 1

        gene_pop_dict[sheet_row] = item
        gene_population.append(tuple(item))
        # for j in range(len(item)):
        #     # sheet.write(sheet_row, j, item[j])
        #     excel.cell(sheet_row, j + 1).value = item[j]
        sheet_row = sheet_row + 1

    json.dump(gene_pop_dict, file_json, ensure_ascii=False)
    file_json.close()

    print(gene_population[:10])

    print(len(gene_population))

    precision_count = 0
    for item in gene_population:
        if item in all_data:
            precision_count = precision_count + 1

    print(f'precision{i}: ', precision_count / len(gene_population))

    recall_count = 0
    for item in all_data:
        if item in gene_population:
            recall_count = recall_count + 1

    print(f'recall{i}: ', recall_count / len(all_data))
