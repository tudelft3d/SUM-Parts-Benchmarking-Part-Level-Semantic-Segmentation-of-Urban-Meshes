import matplotlib.pyplot as plt
import numpy as np

# # 数据集 class 12

# categories = [
#     "terrain", "high_vegetation", "facade_surface", "water", "car",
#     "boat", "roof_surface", "chimney", "dormer", "balcony",
#     "building_part", "wall"
# ]
# areas = [
#     589720.62, 425717.34, 1061496.88, 310500.81, 38179.27,
#     25023.27, 485347.25, 50290.23, 19235.09, 25415.59,
#     26237.8, 15031.79
# ]
# colors = [
#     (170/255, 85/255, 0/255), (0/255, 255/255, 0/255), (255/255, 255/255, 0/255),
#     (0/255, 255/255, 255/255), (255/255, 0/255, 255/255), (0/255, 0/255, 153/255),
#     (85/255, 85/255, 127/255), (255/255, 50/255, 50/255), (85/255, 0/255, 127/255),
#     (50/255, 125/255, 150/255), (50/255, 0/255, 50/255), (215/255, 160/255, 140/255)
# ]
#
# # 创建图形和轴
# fig, ax = plt.subplots(figsize=(10, 6))
# plt.rcParams["font.family"] = "Times New Roman"
#
# # 绘制柱状图
# bar_width = 0.5
# bars = ax.bar(np.arange(len(categories)), areas, color=colors, width=bar_width)
#
# # 设置横轴
# ax.set_xticks(np.arange(len(categories)))
# ax.set_xticklabels(np.arange(1, len(categories)+1), fontsize=18)
# ax.set_xlabel('Semantic facet labels', fontsize=18)
#
# # 设置纵轴
# ax.set_yscale('log')
# ax.set_ylabel('Area ($m^2$)', fontsize=18)
# ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'$10^{int(np.log10(x))}$'))
# ax.set_yticks([10**i for i in range(int(np.log10(min(areas))), int(np.log10(max(areas)))+1)])
# ax.yaxis.set_minor_formatter(plt.FuncFormatter(lambda x, _: ''))
# ax.yaxis.set_ticks_position('left')
#
# ax.set_yticklabels([f'$10^{i}$' for i in range(int(np.log10(min(areas))), int(np.log10(max(areas)))+1)], fontsize=18)  # 设置纵轴标签字体大小
#
#
# # 添加背景格网
# ax.grid(True, which='both', linestyle='--', linewidth=0.5)
# ax.set_axisbelow(True)
#
# # 隐藏顶部和右侧边框
# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)
#
# # 添加图例
# # legend_labels = [f'{i+1}. {categories[i]}' for i in range(len(categories))]
# # legend_colors = [colors[i] for i in range(len(categories))]
# # legend_patches = [plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[i],
# #                              markersize=10, markerfacecolor=legend_colors[i]) for i in range(len(categories))]
# # ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#
# # 显示图形
# plt.tight_layout()
# plt.savefig('semantic_facet_distribution.png')
# plt.show()

#########################################################################################################################


#class 19
categories = [
    "high_vegetation", "facade_surface", "water", "car", "boat",
    "roof_surface", "chimney", "dormer", "balcony", "building_part",
    "wall", "window", "door", "low_vegetation", "impervious_surface",
    "road", "road_marking", "cycle_lane", "sidewalk"
]
pixels = [
    67961110, 130844229, 37949256, 6019488, 3981315,
    82676233, 10825555, 3314326, 3802030, 5237933,
    2423697, 37318841, 2509931, 9653311, 34967488,
    22183622, 1091556, 1164323, 13009539
]
colors = [
    (0/255, 255/255, 0/255), (255/255, 255/255, 0/255), (0/255, 255/255, 255/255),
    (255/255, 0/255, 255/255), (0/255, 0/255, 153/255), (85/255, 85/255, 127/255),
    (255/255, 50/255, 50/255), (85/255, 0/255, 127/255), (50/255, 125/255, 150/255),
    (50/255, 0/255, 50/255), (215/255, 160/255, 140/255), (100/255, 100/255, 255/255),
    (150/255, 30/255, 60/255), (200/255, 255/255, 0/255), (100/255, 150/255, 150/255),
    (200/255, 200/255, 200/255), (150/255, 100/255, 150/255), (255/255, 85/255, 127/255),
    (255/255, 255/255, 170/255)
]

# 创建图形和轴
fig, ax = plt.subplots(figsize=(10, 6))

# 设置字体为Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# 绘制柱状图
bars = ax.bar(np.arange(len(categories)), pixels, color=colors)

# 设置横轴
ax.set_xticks(np.arange(len(categories)))
ax.set_xticklabels(np.arange(1, len(categories)+1), fontsize=18)  # 设置横轴标签字体大小
ax.set_xlabel('Semantic pixel labels', fontsize=18)

# 设置纵轴
ax.set_yscale('log')
ax.set_ylabel('Pixel Count', fontsize=18)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'$10^{int(np.log10(x))}$'))
ax.yaxis.set_ticks_position('left')

# 增加纵轴刻度标签数量
ax.set_yticks([10**i for i in range(5, 9)])
ax.set_yticklabels([f'$10^{i}$' for i in range(5, 9)], fontsize=18)  # 设置纵轴标签字体大小

# 去除边框
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_position(('outward', 10))
# ax.spines['bottom'].set_position(('outward', 10))

# 添加网格线并置于底层
ax.grid(True, which='both', linestyle='--', lw=0.5)
ax.set_axisbelow(True)

# 取消标题 (SIGGRAPH论文的图例通常在caption中)
# ax.set_title('Semantic Facet Distribution', fontsize=10)

# 显示图形
plt.tight_layout()
plt.savefig('semantic_facet_distribution.png')
plt.show()