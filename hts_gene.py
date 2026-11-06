import torch 
from modules import Net_model
from models import Diffusion
import json
import csv
from tqdm import tqdm


def append_dict_to_jsonl(file_path, data_dict):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data_dict, ensure_ascii=False) + '\n')
        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# diffusion = Diffusion()
diffusion = Diffusion(noise_steps=1000, beta_start=1e-4, beta_end=0.02, channels=13, voc_size=70, device='cuda')

model = Net_model(c_in=13, c_out=13, time_dim=512, voca_size=70, device="cuda").to(device)

ckpt = torch.load("./models/DDPM_category5/ckpt_671.pt")
model.load_state_dict(ckpt)

# saved_voca = './models/DDPM_category5/voca_map.json'

# reverse_voc_map = {}
# with open(saved_voca) as f:
#     voca_map = json.load(f)

#     for key in voca_map.keys():
#         reverse_voc_map[voca_map[key]] = key

# print(reverse_voc_map)
# # print(voca_map)


# csv_file = r'D:\Download\Kim and Bansal (2023)_TR-C\data_NHTSint_indiv.csv'
# all_data = []

# with open(csv_file, 'r', encoding='utf-8') as f:
#     reader = csv.reader(f)
#     count = 0

#     idx_map = {}
#     for row in reader:
#         if count == 0:
#             # gene_pop_dict[sheet_row] = row
#             col_count = 0
#             for item in row:
#                 idx_map[col_count] = item
#                 # excel.cell(sheet_row, col_count+1).value = item

#                 # sheet.write(sheet_row, col_count, item)
#                 col_count += 1

#             # sheet_row += 1

#             count = count + 1
#         else:
#             one_sample = []
#             for j in range(len(row)):
#                 tem_key = f'{idx_map[j]}:{row[j]}'

#                 one_sample.append(int(reverse_voc_map[tem_key]))
#             all_data.append(tuple(one_sample))

#             count = count + 1

# print(all_data[:10])

# print(len(set(all_data)))
# print(len(all_data))




model.eval()
num_gene = 1066319

k = num_gene // 300
r = num_gene % 300

# file_json = open(f"./million_genes_ddpm_category5_{num_gene}_sample_1.json", "w")
file_path = f"./million_genes_ddpm_category5_{num_gene}_sample_5.jsonl"

gene_population = []
gene_pop_dict = {}
sheet_row = 1


for ii in tqdm(range(k)):
    
    out = diffusion.sample(model, 300)
    # out = diffusion.sample_acc(model, 300)

    sample_out = out.detach().cpu().numpy().tolist()
    
    sub_dict = {}
    sub_count = 0
    for item in sample_out:
        sub_dict[sub_count] = item
        gene_pop_dict[sheet_row] = item
        gene_population.append(tuple(item))
        sheet_row = sheet_row + 1
        sub_count = sub_count + 1
        
    append_dict_to_jsonl(file_path, sub_dict)

out = diffusion.sample(model, r)
# out = diffusion.sample_acc(model, r)

sample_out = out.detach().cpu().numpy()
sample_out = sample_out.tolist()


sub_dict = {}
sub_count = 0
for item in sample_out:
    sub_dict[sub_count] = item
    gene_pop_dict[sheet_row] = item
    gene_population.append(tuple(item))

    sheet_row = sheet_row + 1
    sub_count = sub_count + 1
append_dict_to_jsonl(file_path, sub_dict)

# json.dump(gene_pop_dict, file_json, ensure_ascii=False)
# file_json.close()

print(gene_population[:10])

print(len(gene_population))



# precision_count = 0
# for item in gene_population:
#     if item in all_data:
#         precision_count = precision_count + 1

# print(f'precision: ', precision_count / len(gene_population))

# recall_count = 0
# for item in all_data:
#     if item in gene_population:
#         recall_count = recall_count + 1

# print(f'recall: ', recall_count / len(all_data))



# # 构建第二个集合（用于快速查找），统计gene_populationa中在all_data中的个数
# set2 = set(all_data)
# count1_in_2 = sum(1 for v in gene_population if v in set2)
# print(f'precision: ', count1_in_2 / len(gene_population))

# # 构建第一个集合（用于快速查找），统计all_data中在gene_population中的个数
# set1 = set(gene_population)
# count2_in_1 = sum(1 for v in all_data if v in set1)
# print(f'recall: ', count2_in_1 / len(all_data))

