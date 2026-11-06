
import numpy as np
import pandas as pd

# 创建两个 DataFrame
# left_df = pd.DataFrame({'key': ['A', 'B', 'C'], 'value_left': [1, 2, 3]})
# right_df = pd.DataFrame({'key': ['B', 'C', 'D'], 'value_right': [4, 5, 6]})
#
# # 内连接合并
# result = pd.merge(left_df, right_df, how='inner', on='key')
# print(result)
# 输出:
#
# css
# 复制代码
#   key  value_left  value_right
# 0   B           2            4
# 1   C           3            5
# 解释: 只保留了键 B 和 C，因为它们在两个 DataFrame 中都存在。
#
# 示例 2: 外连接合并
# python
# 复制代码
# # 外连接合并
# result = pd.merge(left_df, right_df, how='outer', on='key')
# print(result)
# 输出:
#
# css
# 复制代码
#   key  value_left  value_right
# 0   A         1.0          NaN
# 1   B         2.0          4.0
# 2   C         3.0          5.0
# 3   D         NaN          6.0
# 解释: 保留了所有的键，缺失的值填充为 NaN。
#
# 示例 3: 使用索引进行合并
# python
# 复制代码
# 创建两个 DataFrame，索引上有相同的值
left_df = pd.DataFrame({'value_left': [1, 2, 3]}, index=['A', 'B', 'C'])
right_df = pd.DataFrame({'value_right': [4, 5, 6]}, index=['B', 'C', 'D'])

# 使用索引合并
result = pd.merge(left_df, right_df, how='outer', left_index=True, right_index=True)
result = result.fillna(0)
print(result)
print('left_df.shape[0]', left_df.shape[0])
print('result.iloc[:, 1].values', result.iloc[:, 1].values)
print(result.iloc[:, 1].values/left_df.shape[0])

a = np.array([0, 4, 5, 6])
b = np.array([1, 1, 1, 5])
c = []
c.append(a)
c.append(b)

print(c)
c = np.concatenate(c)
print(c)
