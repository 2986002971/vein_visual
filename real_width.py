"""
2024/07/14 -- 尝试获取真实宽度
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def vein_pyramid(image, steps=5):
    G = image.copy()
    for i in range(steps):
        G = cv2.pyrDown(G)

    laplacian_kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    lp = cv2.filter2D(G, cv2.CV_64F, laplacian_kernel, borderType=cv2.BORDER_REPLICATE)

    for i in range(steps):
        lp = cv2.pyrUp(lp)

    lp[lp < 0] = 0
    # max_val = lp.max()
    # lp_normalized = 255 * lp / max_val
    # lp_normalized = np.uint8(lp_normalized)

    lp_binary = lp.copy()
    lp_binary[lp_binary > 0] = 255

    num_labels, labels = cv2.connectedComponents(lp_binary.astype(np.uint8))

    return lp_binary, num_labels, labels


image_folder = "./contrast"
image_files = [f for f in os.listdir(image_folder) if ".png" in f]

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    vein_step1, num1, label1 = vein_pyramid(image, steps=1)
    vein_step2, num2, label2 = vein_pyramid(image, steps=2)
    vein_step3, num3, label3 = vein_pyramid(image, steps=3)
    vein_step4, num4, label4 = vein_pyramid(image, steps=4)
    vein_step5, num5, label5 = vein_pyramid(image, steps=5)

    plt.figure(figsize=(15, 5))
    plt.subplot(2, 3, 1)
    plt.title("step=1")
    plt.imshow(vein_step1, cmap="gray")
    plt.subplot(2, 3, 2)
    plt.title("step=2")
    plt.imshow(vein_step2, cmap="gray")
    plt.subplot(2, 3, 3)
    plt.title("step=3")
    plt.imshow(vein_step3, cmap="gray")
    plt.subplot(2, 3, 4)
    plt.title("step=4")
    plt.imshow(vein_step4, cmap="gray")
    plt.subplot(2, 3, 5)
    plt.title("step=5")
    plt.imshow(vein_step5, cmap="gray")
    plt.subplot(2, 3, 6)
    plt.title("step=6")
    plt.imshow(image, cmap="gray")

    plt.show()
