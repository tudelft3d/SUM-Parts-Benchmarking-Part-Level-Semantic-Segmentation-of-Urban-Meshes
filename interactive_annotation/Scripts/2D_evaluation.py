import cv2
import numpy as np
import argparse

# General util function to get the boundary of a binary mask.
def mask_to_boundary(mask, dilation=1):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]

    # cv2.imwrite("new_mask.png", np.uint8(new_mask))
    # cv2.imwrite("new_mask_erode.png", np.uint8(new_mask))
    # cv2.imwrite("mask_erode.png", np.uint8(mask_erode))
    # G_d intersects G in the paper.
    return mask - mask_erode


def boundary_iou(gt, dt, dilation=1):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    gt_boundary = mask_to_boundary(gt, dilation)
    dt_boundary = mask_to_boundary(dt, dilation)

    # cv2.imwrite("gt_boundary.png",np.uint8(gt_boundary))
    # cv2.imwrite("pred_boundary.png", np.uint8(dt_boundary))

    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return boundary_iou


def calculate_boundary_iou(mask1, mask2, class_colors, dilation=1):
    """
    计算两个mask图像中每个类别的边界IoU。

    :param mask1: 第一个mask图像（通常是ground truth）。
    :param mask2: 第二个mask图像（通常是prediction）。
    :param class_colors: 类别颜色列表，每个颜色是一个BGR三元组。
    :return: 返回一个字典，包含每个类别的边界IoU。
    """
    iou_scores = {}
    for color in class_colors:
        class_mask1 = cv2.inRange(mask1, color, color)
        class_mask2 = cv2.inRange(mask2, color, color)
        iou = boundary_iou(class_mask1, class_mask2, dilation)
        iou_scores[str(color)] = iou

    return iou_scores

def calculate_iou(mask1, mask2, class_colors):
    """
    计算两个mask图像中每个类别的IoU。

    :param mask1: 第一个mask图像（通常是ground truth）。
    :param mask2: 第二个mask图像（通常是prediction）。
    :param class_colors: 类别颜色列表，每个颜色是一个BGR三元组。
    :return: 返回一个字典，包含每个类别的IoU。
    """
    iou_scores = {}
    for color in class_colors:
        # 创建每个类别的二值图像
        class_mask1 = cv2.inRange(mask1, color, color)
        class_mask2 = cv2.inRange(mask2, color, color)

        # 计算交集和并集
        intersection = cv2.bitwise_and(class_mask1, class_mask2)
        union = cv2.bitwise_or(class_mask1, class_mask2)
        intersection_sum = np.sum(intersection > 0)
        union_sum = np.sum(union > 0)

        # 计算IoU
        if union_sum == 0:
            iou = float('nan')  # 避免除以零的错误
        else:
            iou = intersection_sum / union_sum
        iou_scores[str(color)] = iou

    return iou_scores

def apply_background_mask(gt_mask, pred_mask, class_colors, background_value=[0, 0, 0]):
    # Initialize a mask with all pixels set (assuming background is not part of class_colors)
    background_mask = np.ones(gt_mask.shape[:2], dtype=bool)

    # For each class color, mark the pixels
    for color in class_colors:
        class_mask = cv2.inRange(gt_mask, color, color)
        background_mask &= (class_mask == 0)  # Only keep background

    # Wherever background_mask is True, set pred_mask to background_value
    pred_mask[background_mask] = background_value

    return pred_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the image evaluation.")
    # 添加 SAVE_PATH 参数
    parser.add_argument('--gt_path', type=str, default="", help="Path to the ground truth mask.")
    parser.add_argument('--pred_path', type=str, default="", help="Path to the predict mask.")
    parser.add_argument('--background_correction', type=bool, default=True, help="Boundary correction for the predict mask.")
    parser.add_argument('--dilation', type=int, default=2, help="Boundary thickness.")
    # 解析命令行参数
    args = parser.parse_args()

    # 示例颜色，假设有两个类别  # Blue, Green, Red in BGR
    # facade
    class_colors = [(0, 255, 255), (255, 100, 100), (60, 30, 150)] # facade, window, door
    # # low_vegetation
    #class_colors = [(0, 255, 200), (150, 150, 100)] # low_vegetation, impervious_surface
    # # road 1
    # class_colors = [(200, 200, 200), (150, 100, 150), (170, 255, 255), (150, 150, 100)] # road, road_marking, sidewalk, impervious_surface
    # # road 2
    #class_colors = [(200, 200, 200), (150, 100, 150), (170, 255, 255), (150, 150, 100), (127,85,255), (0, 255, 200)] # road, road_marking, sidewalk, impervious_surface, cycle_lane, low_vegetation

    # 读取图像
    gt_mask = cv2.imread(args.gt_path)
    pred_mask = cv2.imread(args.pred_path)

    # Apply background correction
    if args.background_correction:
        pred_mask = apply_background_mask(gt_mask, pred_mask, class_colors)
    #cv2.imwrite("pred_mask.png", pred_mask)

    # 计算IoU
    iou_scores = calculate_iou(gt_mask, pred_mask, class_colors)
    boundary_iou_scores = calculate_boundary_iou(gt_mask, pred_mask, class_colors, args.dilation)

    print("IoU scores by class:", iou_scores)
    print("Boundary IoU scores by class:", boundary_iou_scores)