import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

image_folder = "../"
image_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    total_pixels = image.size
    high_threshold = np.searchsorted(cdf, total_pixels * 0.8)
    high_light_suppressed = image.copy()
    high_light_suppressed[high_light_suppressed > high_threshold] = high_threshold

    # 创建高斯金字塔
    G = image.copy()
    for i in range(6):
        G = cv2.pyrDown(G)

    lp = cv2.Laplacian(G, cv2.CV_64F)
    for i in range(6):
        lp = cv2.pyrUp(lp)

    # 将拉普拉斯变换后的图像缩放到0到255范围
    min_val, max_val = lp.min(), lp.max()
    lp_normalized = 255 * (lp - min_val) / (max_val - min_val)
    lp_normalized = np.uint8(lp_normalized)

    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 应用CLAHE
    cl1 = clahe.apply(lp_normalized)

    for _ in range(5):
        cl1 = cv2.erode(cl1, np.ones((9, 9), np.uint8), iterations=1)

    # 获取图像中心部分的坐标
    cl1 = cv2.resize(cl1, (image.shape[1], image.shape[0]))
    h, w = image.shape
    cx, cy = w // 2, h // 2
    half_w, half_h = w // 4, h // 4
    cl1_resized = cl1[cy - half_h : cy + half_h, cx - half_w : cx + half_w]

    # 将处理后的图像叠加回原图
    processed_image = image.copy()
    processed_image[cy - half_h : cy + half_h, cx - half_w : cx + half_w] = cl1_resized

    # 创建子图并显示图像
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(processed_image, cmap="gray")
    plt.title("Processed Image")
    plt.axis("off")

    plt.show()
