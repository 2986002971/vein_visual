"""
2024/07/14 -- 尝试获取真实宽度, 进行多尺度可视化
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random


def random_color():
    levels = range(32, 256, 32)
    return tuple(random.choice(levels) for _ in range(3))


def vein_pyramid_components(image, steps=6):
    G = image.copy()
    for i in range(steps):
        G = cv2.pyrDown(G)

    laplacian_kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    lp = cv2.filter2D(G, cv2.CV_64F, laplacian_kernel, borderType=cv2.BORDER_REPLICATE)

    for i in range(steps):
        lp = cv2.pyrUp(lp)

    lp[lp < 0] = 0
    max_val = lp.max()
    lp_normalized = 255 * lp / max_val
    lp_normalized = np.uint8(lp_normalized)

    lp_binary = lp.copy()
    lp_binary[lp_binary > 0] = 255

    num_labels, labels = cv2.connectedComponents(lp_binary.astype(np.uint8))

    # 可视化labels
    color_label = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    for label in range(1, num_labels):
        color = random_color()
        color_label[labels == label] = color

    return lp_normalized, lp_binary, num_labels, color_label


image_folder = "./contrast"
image_files = [f for f in os.listdir(image_folder) if ".png" in f]

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    vein_step1, bin1, num1, label_img1 = vein_pyramid_components(image, steps=1)
    vein_step2, bin2, num2, label_img2 = vein_pyramid_components(image, steps=2)
    vein_step3, bin3, num3, label_img3 = vein_pyramid_components(image, steps=3)
    vein_step4, bin4, num4, label_img4 = vein_pyramid_components(image, steps=4)
    vein_step5, bin5, num5, label_img5 = vein_pyramid_components(image, steps=5)
    vein_step6, bin6, num6, label_img6 = vein_pyramid_components(image, steps=6)

    plt.figure(figsize=(16, 8))

    plt.subplot(3, 6, 1)
    plt.title("original")
    plt.imshow(image, cmap="gray")
    plt.subplot(3, 6, 2)
    plt.title("step=2")
    plt.imshow(vein_step2, cmap="gray")
    plt.subplot(3, 6, 3)
    plt.title("step=3")
    plt.imshow(vein_step3, cmap="gray")
    plt.subplot(3, 6, 4)
    plt.title("step=4")
    plt.imshow(vein_step4, cmap="gray")
    plt.subplot(3, 6, 5)
    plt.title("step=5")
    plt.imshow(vein_step5, cmap="gray")
    plt.subplot(3, 6, 6)
    plt.title("step=6")
    plt.imshow(vein_step6, cmap="gray")

    plt.subplot(3, 6, 7)
    plt.title("Binary 1")
    plt.imshow(bin1, cmap="gray")
    plt.subplot(3, 6, 8)
    plt.title("Binary 2")
    plt.imshow(bin2, cmap="gray")
    plt.subplot(3, 6, 9)
    plt.title("Binary 3")
    plt.imshow(bin3, cmap="gray")
    plt.subplot(3, 6, 10)
    plt.title("Binary 4")
    plt.imshow(bin4, cmap="gray")
    plt.subplot(3, 6, 11)
    plt.title("Binary 5")
    plt.imshow(bin5, cmap="gray")
    plt.subplot(3, 6, 12)
    plt.title("Binary 6")
    plt.imshow(bin6, cmap="gray")

    plt.subplot(3, 6, 13)
    plt.title(f"Labels 1: {num1}")
    plt.imshow(label_img1)
    plt.subplot(3, 6, 14)
    plt.title(f"Labels 2: {num2}")
    plt.imshow(label_img2)
    plt.subplot(3, 6, 15)
    plt.title(f"Labels 3: {num3}")
    plt.imshow(label_img3)
    plt.subplot(3, 6, 16)
    plt.title(f"Labels 4: {num4}")
    plt.imshow(label_img4)
    plt.subplot(3, 6, 17)
    plt.title(f"Labels 5: {num5}")
    plt.imshow(label_img5)
    plt.subplot(3, 6, 18)
    plt.title(f"Labels 6: {num6}")
    plt.imshow(label_img6)

    plt.tight_layout()
    plt.savefig(f"vein_pyramid_{image_file}")
    plt.show()
