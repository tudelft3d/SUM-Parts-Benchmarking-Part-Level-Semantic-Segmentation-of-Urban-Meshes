import cv2
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="生成预测图像和真实图像的错误掩码")
    parser.add_argument("--ground_truth", type=str, required=True, help="真实图像的文件路径")
    parser.add_argument("--prediction", type=str, required=True, help="预测图像的文件路径")
    parser.add_argument("--output_mask", type=str, default="error_mask.png", help="输出错误掩码的文件名")
    return parser.parse_args()

def load_image(path):
    # 加载图像，确保使用8位深度的RGB模式读取
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image.shape[2] == 4:  # 如果图像是32位的，去除alpha通道
        image = image[:, :, :3]
    return image

def main():
    args = parse_args()

    # 定义类别颜色
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

    # 将颜色转换为numpy数组以便比较
    color_values = np.array(list(category_colors.values()))

    # 加载图像
    ground_truth = load_image(args.ground_truth)
    prediction = load_image(args.prediction)

    # 确保图像尺寸相同
    if ground_truth.shape != prediction.shape:
        raise ValueError("基准真值图像和预测图像尺寸不匹配")

    # 初始化错误掩码，默认背景为白色
    error_mask = np.ones((ground_truth.shape[0], ground_truth.shape[1], 3), dtype=np.uint8) * 255

    # 检查基准真值中的颜色是否在预定义颜色中
    for color in color_values:
        ground_truth_mask = np.all(ground_truth == color, axis=2)
        prediction_mask = np.all(prediction == color, axis=2)
        correct_mask = ground_truth_mask & prediction_mask
        error_mask[correct_mask] = [0, 255, 0]  # 绿色表示匹配
        incorrect_mask = ground_truth_mask & ~prediction_mask
        error_mask[incorrect_mask] = [0, 0, 255]  # 红色表示错误

    # 保存错误掩码
    cv2.imwrite(args.output_mask, error_mask)
    print(f"错误掩码已保存到 {args.output_mask}")


if __name__ == "__main__":
    main()
