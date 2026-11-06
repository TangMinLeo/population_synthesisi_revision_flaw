import matplotlib.pyplot as plt
import numpy as np

# x = np.linspace(0, 1.0, 11)
# y = np.sin(x)

x = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
y = [0.1167, 0.2005, 0.3342, 0.4468, 0.5479, 0.6366, 0.7196, 0.7966, 0.8676, 0.9369, 1.0]
y1 = [0.5635, 0.6604, 0.7582, 0.8177, 0.8611, 0.8938, 0.9215, 0.9453, 0.9654, 0.9840, 1.0]

fig, ax = plt.subplots()
# fig1 = plt.figure()
curve1, = ax.plot(x, y1, marker='.', color='red', markersize=10, label='h-sample')
curve, = ax.plot(x, y, marker='.', color='orange', markersize=10, label='h-sample')

ax.plot([0.05, 0.05], [0.0, 1.0], color='black', linestyle='--')

ax.grid(True, linestyle='--', color='gray')
ax.set_ylim(0.0, 1.0)

ax.set_xticks(np.arange(0.0, 1.1, 0.1))
ax.set_yticks(np.arange(0.0, 1.1, 0.1))
# 在各个点上显示数值
for x_val, y_val in zip(x[:-1], y[:-1]):
    plt.text(x_val-0.03, y_val-0.05, f'{(y_val*100):.2f}%', fontsize=7)

for x_val, y_val in zip(x[:-2], y1[:-2]):
    plt.text(x_val-0.03, y_val-0.05, f'{(y_val*100):.2f}%', fontsize=7)

plt.annotate(f'{(y1[-2]*100):.2f}%', xy=(x[-2], y1[-2]), xytext=(+19, -12), textcoords='offset points', xycoords='data', fontsize=7,
                  arrowprops=dict(arrowstyle='-', connectionstyle='arc,angleA=180, angleB=0, armA=40, armB=0,rad=5', color='gray'))

ax.set_xlabel('Sampling rate')
ax.set_ylabel('Ratio')
# # 显示图例
plt.text(-0.03, 1+0.1, f'h-sample', fontsize=10)
plt.text(-0.09, 1+0.045, f'(30820 combinations)', fontsize=10)

plt.text(1-0.14, 1+0.1, f'h-population', fontsize=10)
plt.text(1-0.2, 1+0.045, f'(264005 combinations)', fontsize=10)
curve.set_label('Sampled combinations')
curve1.set_label('Sampled data instances')

ax.legend(loc='lower right')

plt.show()


# x = [0, 1, 2, 3, 4]
# y = [10, 20, 15, 25, 30]
#
# # 绘制数据点
# plt.plot(x, y, 'o')
#
# # 指定需要添加折线指示的点
# annotate_points = [(1, 20), (3, 25)]
#
# # 添加折线指示
# for point in annotate_points:
#     plt.annotate(f'({point[0]}, {point[1]})', xy=point, xytext=(+50, -50), textcoords='offset points', xycoords='data',
#                  arrowprops=dict(arrowstyle='-', connectionstyle='arc,angleA=180, angleB=0, armA=70, armB=0,rad=5'))
#
# # 显示图形
# plt.show()
