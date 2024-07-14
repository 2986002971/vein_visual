"""
2024/07/14 -- 闭运算展示
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

image_folder = "./contrast"
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

    laplacian_kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    lp = cv2.filter2D(G, cv2.CV_64F, laplacian_kernel, borderType=cv2.BORDER_REPLICATE)

    # 将拉普拉斯图像上采样回原始大小
    for i in range(4):
        lp = cv2.pyrUp(lp)

    # 曲率增强简化版，将负曲率置为零，然后归一化到0到255
    lp[lp < 0] = 0
    max_val = lp.max()
    lp_normalized = 255 * lp / max_val
    lp_normalized = np.uint8(lp_normalized)

    # 闭运算
    kernel = np.ones((33, 33), np.uint8)
    for _ in range(3):
        closed_image = cv2.morphologyEx(lp_normalized, cv2.MORPH_CLOSE, kernel)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title("Laplacian Enhanced")
    plt.imshow(lp_normalized, cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title("After Morphological Close")
    plt.imshow(closed_image, cmap="gray")

    plt.show()
