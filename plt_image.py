import matplotlib.pyplot as plt
import numpy as np

# 示例数据
x = np.arange(0, 10, 0.1)
y1 =[]
y2 = np.cos(x)
y3 = np.tan(x) / 10  # 适当缩小范围以便于显示

# 创建一个图形对象和子图对象
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制多条折线
ax.plot(x, y1, label='N w ', color='b', linestyle='-', linewidth=2, marker='o')
ax.plot(x, y2, label='cos(x)', color='g', linestyle='--', linewidth=2, marker='s')
ax.plot(x, y3, label='tan(x)/10', color='r', linestyle='-.', linewidth=2, marker='^')

# 添加标题和标签
ax.set_title('Multiple Line Plots Example', fontsize=16, fontweight='bold')
ax.set_xlabel('X-axis', fontsize=14)
ax.set_ylabel('Y-axis', fontsize=14)

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.5)

# 添加图例
ax.legend(loc='upper right', fontsize=12)

# 美化坐标轴
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.tick_params(axis='both', which='major', labelsize=12)

# 显示图形
plt.tight_layout()
plt.show()
