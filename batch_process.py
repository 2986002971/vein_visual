"""
2024/07/14 -- 批处理，生成静脉增强图像
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

image_folder = "./contrast_origin"
image_files = [f for f in os.listdir(image_folder) if ".png" in f]

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 高光抑制
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    total_pixels = image.size
    high_threshold = np.searchsorted(cdf, total_pixels * 0.8)  # 参数可调
    high_light_suppressed = image.copy()
    high_light_suppressed[high_light_suppressed > high_threshold] = high_threshold

    # 创建高斯金字塔
    G = high_light_suppressed.copy()
    for i in range(4):
        G = cv2.pyrDown(G)

    lp = cv2.Laplacian(G, cv2.CV_64F)

    # 将拉普拉斯图像上采样回原始大小
    for i in range(4):
        lp = cv2.pyrUp(lp)

    # 曲率增强简化版，将负曲率置为零，然后归一化到0到255
    lp[lp < 0] = 0
    min_val, max_val = lp.min(), lp.max()
    lp_normalized = 255 * (lp - min_val) / (max_val - min_val)
    lp_normalized = np.uint8(lp_normalized)

    cv2.imwrite(f"after_{image_file}", lp_normalized)
