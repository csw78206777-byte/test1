import numpy as np
import cv2
import os
import random

# --- 清理旧文件 ---
if os.path.exists('test_images'):
    for f in os.listdir('test_images'):
        os.remove(os.path.join('test_images', f))
else:
    os.makedirs('test_images')

# --- 参数设置 ---
width, height = 640, 480
num_frames = 200  # 增加帧数

# --- 物体状态 ---
# 初始位置
x, y = float(width / 4), float(height / 4)
# 初始速度
vx, vy = 2.0, 2.0
# 物体大小
w, h = 50, 50

# --- 生成图像 ---
for i in range(num_frames):
    # 创建一个黑色背景
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # --- 更新速度和位置 ---
    # 引入随机加速度，使运动不规则
    ax = random.uniform(-0.5, 0.5)
    ay = random.uniform(-0.5, 0.5)
    vx += ax
    vy += ay

    # 限制最高速度
    vx = np.clip(vx, -7, 7)
    vy = np.clip(vy, -7, 7)

    # 更新位置
    x += vx
    y += vy

    # --- 边界碰撞检测 ---
    if x + w/2 >= width or x - w/2 <= 0:
        vx *= -0.9 # 反弹并衰减
        x += vx # 避免卡在边界
    if y + h/2 >= height or y - h/2 <= 0:
        vy *= -0.9 # 反弹并衰减
        y += vy # 避免卡在边界

    # --- 模拟检测噪声 ---
    # 在真实位置上增加一点随机噪声
    noise_x = x + random.uniform(-2, 2)
    noise_y = y + random.uniform(-2, 2)
    
    # 绘制带有噪声的物体
    draw_x, draw_y = int(noise_x - w/2), int(noise_y - h/2)
    cv2.rectangle(frame, (draw_x, draw_y), (draw_x + w, draw_y + h), (255, 255, 255), -1)

    # 保存图像
    cv2.imwrite(f'test_images/frame_{i:03d}.png', frame)

print(f"{num_frames} new test images with irregular movement generated in 'test_images' folder.")