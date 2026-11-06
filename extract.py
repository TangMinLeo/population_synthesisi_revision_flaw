from openpyxl import load_workbook
import xlwt

file = load_workbook(filename=r'D:\我的桌面\社区调查\副本254121960_0_社区人口基础调研问卷（湖口小区合并杏园）.xlsx')
sheet = file['Sheet1']
break_cols = [12, 13, 16, 19, 26, 27, 28, 29, 30, 31]

workbook = xlwt.Workbook()
sheet_write = workbook.add_sheet('Sheet1')

for i in range(1, sheet.max_row + 1):
    if i == 1:
        col = 0
        for j in range(7, 32):
            if j in break_cols:
                continue
            else:
                tem = sheet.cell(row=i, column=j).value
                tem = tem.split('：')[0]
                sheet_write.write(i - 1, col, tem)
                col = col + 1
    else:
        col = 0
        for j in range(7, 32):
            if j in break_cols:
                continue
            else:
                tem = sheet.cell(row=i, column=j).value

                sheet_write.write(i - 1, col, tem)
                col = col + 1

workbook.save('results_xingyuanhukou.xlsx')

file.close()
