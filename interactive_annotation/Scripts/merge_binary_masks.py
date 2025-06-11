import cv2
import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="合成多个类别的二进制掩码为单个彩色图像")
    parser.add_argument("--input_dir", type=str, required=True, help="包含二进制掩码的输入目录")
    parser.add_argument("--output_image", type=str, default="output.png", help="输出图像文件名")
    return parser.parse_args()

def main():
    args = parse_args()

    # 定义每个类别的颜色
    # category_colors = {
    #     'facade': (0, 255, 255),
    #     'window': (255, 100, 100),
    #     'door': (60, 30, 150)
    # }

    # category_colors = {
    #     'veg': (0, 255, 200),
    #     'ground': (150, 150, 100)
    # }

    # category_colors = {
    #     'mark': (150, 100, 150),
    #     'sidewalk': (170, 255, 255),
    #     'road': (200, 200, 200),
    #     'imprevious': (150, 150, 100)
    # }

    category_colors = {
        'mark': (150, 100, 150),
        'sidewalk': (170, 255, 255),
        'road': (200, 200, 200),
        'imprevious': (150, 150, 100),
        'cyclelane': (127,85,255),
        'veg': (0, 255, 200)
    }

    # 排序类别以确保颜色覆盖顺序
    priority_order = ['mark', 'cyclelane', 'sidewalk', 'veg', 'imprevious', 'road']  # 可以根据需要调整这里的顺序

    # 获取所有png文件
    mask_files = [f for f in os.listdir(args.input_dir) if f.endswith('.png')]

    # 根据优先级排序文件
    mask_files.sort(key=lambda x: priority_order.index(x.split('_')[0]))

    composite_image = None
    applied_mask = None

    for mask_file in mask_files:
        # 解析文件名以获取类别
        category = mask_file.split('_')[0]
        if category in category_colors:
            # 读取图像
            mask_path = os.path.join(args.input_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if composite_image is None:
                # 初始化合成图像
                composite_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                applied_mask = np.zeros(mask.shape, dtype=bool)

            # 应用类别颜色
            color = category_colors[category]
            mask_indices = (mask == 255) & (~applied_mask)
            composite_image[mask_indices] = color
            applied_mask[mask_indices] = True

    # 保存最终的合成图像
    if composite_image is not None:
        cv2.imwrite(args.output_image, composite_image)
        print(f"合成图像已保存到 {args.output_image}")
    else:
        print("没有找到有效的掩码文件")

if __name__ == "__main__":
    main()
