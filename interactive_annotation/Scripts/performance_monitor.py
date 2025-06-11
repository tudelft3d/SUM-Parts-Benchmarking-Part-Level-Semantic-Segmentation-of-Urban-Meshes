#pip install pynput pywin32

from pynput import mouse, keyboard
from pynput.keyboard import Key, Listener as KeyboardListener
import math
import time
import os
import argparse

class Tracker:
    def __init__(self):
        self.mouse_clicks = 0 #{Button.left: 0, Button.middle: 0, Button.right: 0, 'scroll': 0}
        self.key_presses = 0
        self.key_recording_press = 0
        self.scroll_rounds = 0  # 新增属性来记录滚轮滚动的整数圈数
        self.start_time = time.time()
        self.total_time = 0
        self.paused = False
        self.started = False
        self.mouse_distance = 0
        self.last_position = (-1, -1)
        self.dpi = 1000 #windll.user32.GetDpiForSystem() / 96.0  # Default DPI is 96, logitech M705
        self.shift_pressed = False
        self.special_keys_pressed = set()
        self.save_path = '.'

    def on_click(self, x, y, button, pressed):
        if not self.paused and self.started and pressed:
            self.mouse_clicks += 1
            # if button in self.mouse_clicks:
            #     self.mouse_clicks[button] += 1
            # else:
            #     self.mouse_clicks['scroll'] += 1  # Consider scroll as a click

    def on_scroll(self, x, y, dx, dy):
        if not self.paused and self.started:
            ticks_per_round = 24 #齿格
            self.scroll_rounds += abs(dy) / ticks_per_round  # 累加滚轮的tick数
            #print(f"Scroll Rounds: {self.scroll_ticks}")

    def on_move(self, x, y):
        if not self.paused and self.started:
            if self.last_position[0] > -1 and self.last_position[1] > -1:
                pixel_distance = math.sqrt((x - self.last_position[0]) ** 2 + (y - self.last_position[1]) ** 2)
                cm_distance = 2.54 * pixel_distance / self.dpi
                self.mouse_distance += cm_distance
                #print(pixel_distance)
            self.last_position = (x, y)

    def on_press(self, key):
        # 标记shift键是否被按下
        if key == keyboard.Key.shift_l or key == keyboard.Key.shift_r:
            self.shift_pressed = True
        elif key in [Key.f1, Key.f2]:
            self.special_keys_pressed.add(key)

        # 检查是否同时按下了shift和F1/F2
        if self.shift_pressed and (Key.f1 in self.special_keys_pressed or Key.f2 in self.special_keys_pressed):
            self.handle_shift_f_keys(key)
        self.save_data()

    def on_release(self, key):
        if not self.paused and self.started:
            self.key_presses += 1 #组合按键算一次
            #print(key)
        if key == keyboard.Key.shift_l or key == keyboard.Key.shift_r:
            self.shift_pressed = False
        if key in self.special_keys_pressed:
            self.special_keys_pressed.remove(key)

    def handle_shift_f_keys(self, key):
        if key == Key.f1:
            if not self.started:
                self.started = True
                self.start_time = time.time()
                self.key_recording_press += 2
                print("Start")
            elif self.paused:
                self.paused = False
                self.start_time = time.time()
                self.key_recording_press += 1
                print("Resume")
            else:
                self.paused = True
                self.total_time += time.time() - self.start_time
                print("Pause")
        elif key == Key.f2:
            self.total_time += time.time() - self.start_time
            self.key_presses -= self.key_recording_press
            if self.key_presses < 0:
                self.key_presses = 0
            self.stop_listening()
            print("Stop")
            self.save_data()
            print("Saved")
            exit()

    def stop_listening(self):
        self.mouse_listener.stop()
        self.keyboard_listener.stop()

    def save_data(self):
        os.makedirs(self.save_path, exist_ok=True)
        filename = os.path.join(self.save_path, f"tracker_log.txt")
        with open(filename, 'w') as f:
            f.write(f"Mouse clicks: {self.mouse_clicks}\n")
            f.write(f"Mouse scrolls: {self.scroll_rounds:.2f}\n")
            f.write(f"Mouse distance (cm): {self.mouse_distance:.2f}\n")
            f.write(f"Key presses: {self.key_presses}\n")
            f.write(f"Total time (seconds): {self.total_time:.2f}\n")

    def run(self):
        self.mouse_listener = mouse.Listener(on_click=self.on_click, on_scroll=self.on_scroll, on_move=self.on_move)
        self.keyboard_listener = KeyboardListener(on_press=self.on_press, on_release=self.on_release)
        self.mouse_listener.start()
        self.keyboard_listener.start()
        self.mouse_listener.join()
        self.keyboard_listener.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the tracker and save output to a specified path.")
    # 添加 SAVE_PATH 参数
    parser.add_argument('--SAVE_PATH', type=str, help="Path to save the tracker's output.")
    # 解析命令行参数
    args = parser.parse_args()

    print("Press shift+f1 to start.")
    tracker = Tracker()
    tracker.save_path = args.SAVE_PATH
    tracker.run()




