import pyautogui
import time
import os


def auto_screenshot(interval=300, display=0, save_path='.'):
    """
    自动截图并保存，文件名为程序运行时间的分钟数。

    :param interval: 截图的时间间隔，单位为秒。
    :param display: 用户选择的显示屏幕编号，0 表示主屏幕。
    :param save_path: 截图保存的目录路径。
    """
    start_time = time.time()  # 记录程序开始运行的时间
    while True:
        # 确保保存路径存在
        os.makedirs(save_path, exist_ok=True)
        # 计算当前时间与开始时间的差值（分钟数）
        time_elapsed = int(time.time() - start_time) #seconds
        # 生成文件名，包含完整的保存路径
        filename = os.path.join(save_path, f"{time_elapsed}.png")
        # 进行截图
        screenshot = pyautogui.screenshot()
        # 保存截图
        screenshot.save(filename)
        print(f"截图已保存：{filename}")
        # 等待下一次截图
        time.sleep(interval)


if __name__ == "__main__":
    # 设置间隔时间，比如每5分钟（300秒）
    INTERVAL = 600  # 可以根据需要修改
    # 设置要截图的显示屏幕编号
    DISPLAY = 3  # 0 通常代表主显示器
    # 设置截图保存的目录路径
    SAVE_PATH = 'D:\\pa3\\semi_auto_compare_methods\\sumv2_data_for_compare\\3D\\scene\\buidling_blocks_with_roof_superstructures\\manual\\log'  # 请根据你的需求修改这个路径
    auto_screenshot(INTERVAL, DISPLAY, SAVE_PATH)
