import numpy as np
import cv2
import os

# 创建保存图像的目录
if not os.path.exists('test_images'):
    os.makedirs('test_images')

# 图像尺寸
width, height = 640, 480
num_frames = 64

# 移动物体的初始位置和大小
x, y, w, h = 100, 100, 50, 50
# 移动速度
vx, vy = 5, 2

# 生成64张图像
for i in range(num_frames):
    # 创建一个黑色背景
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # 更新物体位置
    x += vx
    y += vy

    # 边界碰撞检测
    if x + w > width or x < 0:
        vx = -vx
    if y + h > height or y < 0:
        vy = -vy

    # 绘制白色方块作为物体
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), -1)

    # 保存图像
    cv2.imwrite(f'test_images/frame_{i:02d}.png', frame)

print(f"{num_frames} test images generated in 'test_images' folder.")