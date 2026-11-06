from openpyxl import load_workbook
import csv
import random
import json


def investigate_functon(file_name):

    dict_dims = {}
    idx_map = {}

    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        count = 0
        for row in reader:
            if count == 0:
                col_count = 0
                for item in row:
                    dict_dims[item] = []
                    idx_map[col_count] = item
                    col_count += 1
                count = count + 1
            else:
                for j in range(len(row)):
                    dict_dims[idx_map[j]].append(row[j])
                count = count + 1

        print(count)

    unique_dims = {}
    total_ans = 0
    for key in dict_dims.keys():
        unique_dims[key] = list(set(dict_dims[key]))
        total_ans = total_ans + len(unique_dims[key])

    vocabulary = {}
    voca_count = 0
    for key in unique_dims.keys():
        for value in unique_dims[key]:
            vocabulary[f'{key}:{value}'] = voca_count
            voca_count = voca_count + 1

    print(vocabulary)
    print(voca_count)

    f = open('training_sample.json', 'w')
    table_input = []
    for i in range(count-1):
        # sampling only the 5% data
        thre = random.uniform(0.0, 1.0)
        if thre <= 0.05:
            one_sample = []
            for key in dict_dims.keys():
                one_sample.append(vocabulary[f'{key}:{dict_dims[key][i]}'])
            table_input.append(one_sample)

    json.dump(table_input, f, ensure_ascii=False)
    f.close()
    return table_input, vocabulary

    # sheet = file['Sheet1']
    # # break_cols = [12, 13, 16, 19, 26, 27, 28, 29, 30, 31]
    #
    # indicate_of_list_value = {}
    # big_dict = {}
    # total_answers = {}
    #
    # for j in range(1, sheet.max_column+1):
    #     tep_str = sheet.cell(1, j).value.split('、')[1]
    #     tep_str = tep_str.split('：')[0]
    #
    #     print(tep_str)
    #     big_dict[tep_str] = []
    #     total_answers[tep_str] = []
    #     indicate_of_list_value[tep_str] = 0
    #
    # # excel里行列索引都是从1开始
    # for i in range(2, sheet.max_row + 1):
    #     for j in range(1, sheet.max_column+1):
    #
    #         value = sheet.cell(i, j).value
    #
    #         tep_str = sheet.cell(1, j).value.split('、')[1]
    #         tep_str = tep_str.split('：')[0]
    #
    #         if '┋' in value:
    #             val_list = value.split('┋')
    #             total_answers[tep_str].extend(val_list)
    #
    #             big_dict[tep_str].append(val_list)
    #             indicate_of_list_value[tep_str] = 1
    #
    #         else:
    #             total_answers[tep_str].append(value)
    #             big_dict[tep_str].append(value)
    #
    # print(big_dict)
    # # print(total_answers)
    # print(indicate_of_list_value)
    #
    # num_answers = 0
    # vocabulary = {}
    # voca_count = 0
    # for key in total_answers.keys():
    #     total_answers[key] = list(set(total_answers[key]))
    #     num_answers = num_answers + len(total_answers[key])
    #     # if key == '子女年龄':
    #     #     total_answers[key] = ['6~12岁', '2~5岁', '无子女', '13~18岁', '0~2岁', '19岁以上']
    #     # if key == '酒店宾馆':
    #     #     total_answers[key] = ['旅馆招待所', '公寓式酒店', '经济型酒店', '度假村', '青年旅社', '星级酒店', '其他酒店宾馆']
    #
    #     for attrib in total_answers[key]:
    #         vocabulary[f'{key}:{attrib}'] = voca_count
    #         voca_count = voca_count + 1
    #
    # print(total_answers)
    # print(num_answers)
    # print(vocabulary)
    #
    # table_input = []
    # # excel里行列索引都是从1开始
    # for i in range(2, sheet.max_row + 1):
    #
    #     one_sample = []
    #     for j in range(1, sheet.max_column+1):
    #
    #         value = sheet.cell(i, j).value
    #
    #         tep_str = sheet.cell(1, j).value.split('、')[1]
    #         tep_str = tep_str.split('：')[0]
    #
    #         # voc_key = f'{tep_str}:{value}'
    #
    #         if '┋' in value:
    #             val_list = value.split('┋')
    #
    #             answer_tuple = ()
    #             for item in val_list:
    #                 voc_key = f'{tep_str}:{item}'
    #                 answer_tuple = answer_tuple + (vocabulary[voc_key],)
    #
    #             one_sample.append(answer_tuple)
    #             # total_answers[tep_str].extend(val_list)
    #             # big_dict[tep_str].append(val_list)
    #             # indicate_of_list_value[tep_str] = 1
    #
    #         else:
    #             # total_answers[tep_str].append(value)
    #             # big_dict[tep_str].append(value)
    #             voc_key = f'{tep_str}:{value}'
    #             one_sample.append(vocabulary[voc_key])
    #
    #     table_input.append(one_sample)
    # # print(table_input)
    #
    # return table_input, vocabulary



if __name__ == '__main__':
    file_path = r"D:\我的桌面\社区调查\副本254121960_0_社区人口基础调研问卷（湖口小区）_948_895.xlsx"
    table = investigate_functon(file_path)
