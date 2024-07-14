import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

image_folder = "../"
image_files = [f for f in os.listdir(image_folder) if f.endswith("cut3.png")]

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

    # # 获取图像中心部分的坐标
    # cl1 = cv2.resize(high_light_suppressed, (image.shape[1], image.shape[0]))
    # h, w = image.shape
    # cx, cy = w // 2, h // 2
    # half_w, half_h = w // 3, h // 3
    # cl1_resized = cl1[cy - half_h : cy + half_h, cx - half_w : cx + half_w]

    # 创建高斯金字塔
    G = high_light_suppressed.copy()
    for i in range(4):
        G = cv2.pyrDown(G)

    cv2.imshow("G", G)

    # 在金字塔顶部应用拉普拉斯变换
    lp = cv2.Laplacian(G, cv2.CV_64F)
    plt.imshow(lp, cmap="gray")
    plt.title("Laplacian")
    plt.show()

    # 将图像上采样回原始大小
    for i in range(4):
        lp = cv2.pyrUp(lp)
    plt.imshow(lp, cmap="gray")
    plt.title("Laplacian Pyramid")
    plt.show()

    # 曲率增强简化版，将负曲率置为零，然后归一化到0到255
    lp[lp < 0] = 0
    min_val, max_val = lp.min(), lp.max()
    lp_normalized = 255 * (lp - min_val) / (max_val - min_val)
    lp_normalized = np.uint8(lp_normalized)
    plt.imshow(lp_normalized, cmap="gray")
    plt.title("Normalized Laplacian Pyramid")
    plt.show()

    # # 创建CLAHE对象
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # # 应用CLAHE
    # cl1 = clahe.apply(lp_normalized)

    # 暴力阈值化
    threshold_image = np.where(lp_normalized > 0, 255, 0).astype(np.uint8)
    # 开运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opening = cv2.morphologyEx(threshold_image, cv2.MORPH_OPEN, kernel)
    plt.imshow(opening, cmap="gray")
    plt.title("Opening")
    plt.show()
    cv2.imwrite(f"after_{image_file}", opening)

    # # 获取图像中心部分的坐标
    # cl1 = cv2.resize(cl1, (image.shape[1], image.shape[0]))
    # h, w = image.shape
    # cx, cy = w // 2, h // 2
    # half_w, half_h = w // 4, h // 4
    # cl1_resized = cl1[cy - half_h : cy + half_h, cx - half_w : cx + half_w]

    # # 将处理后的图像叠加回原图
    # processed_image = image.copy()
    # processed_image[cy - half_h : cy + half_h, cx - half_w : cx + half_w] = cl1_resized

    # # 创建子图并显示图像
    # plt.figure(figsize=(10, 5))

    # plt.subplot(1, 2, 1)
    # plt.imshow(image, cmap="gray")
    # plt.title("Original Image")
    # plt.axis("off")

    # plt.subplot(1, 2, 2)
    # plt.imshow(processed_image, cmap="gray")
    # plt.title("Processed Image")
    # plt.axis("off")

    # plt.show()
