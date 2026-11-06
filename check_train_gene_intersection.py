import os
import json

dir = './gene_category5_1'

all_genes = []
for filename in os.listdir(dir):
    f = open(os.path.join(dir, filename), 'r')
    data = json.load(f)
    f.close()

    for key in data.keys():
        all_genes.append(tuple(data[key]))

print(len(all_genes))

train_samples = './training_sample_cate5.json'
f = open(train_samples, 'r')
data = json.load(f)
f.close()

train_sam = []
for sample in data:
    train_sam.append(tuple(sample))

print(len(train_sam))

train_in_gene = 0
for one_item in train_sam:
    if one_item in all_genes:
        train_in_gene += 1
print('train_in_gene: ', train_in_gene)

print(train_in_gene/len(train_sam))

gene_in_train = 0
for one_item in all_genes:
    if one_item in train_sam:
        gene_in_train += 1

print('gene_in_train: ', gene_in_train)
print(gene_in_train/len(all_genes))
