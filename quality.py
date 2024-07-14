"""
2024/07/14 -- 生成信息熵对比折线图
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def my_measure(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    hist_norm = hist_norm[np.nonzero(hist_norm)]
    entropy = -np.sum(hist_norm * np.log2(hist_norm))
    return entropy


if __name__ == "__main__":
    image_folder = "./contrast_origin"
    image_files = [f for f in os.listdir(image_folder) if ".png" in f]

    original_entropy = []
    after_entropy = []
    labels = []

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        entropy = my_measure(image)

        if "after" in image_file:
            after_entropy.append(entropy)
        else:
            original_entropy.append(entropy)
            labels.append(image_file)

    # 绘制折线图
    x = np.arange(len(labels))  # 标签位置

    fig, ax = plt.subplots()
    ax.plot(x, original_entropy, marker="o", linestyle="-", label="Original")
    ax.plot(x, after_entropy, marker="o", linestyle="-", label="After")

    ax.set_xlabel("Image")
    ax.set_ylabel("Entropy")
    ax.set_title("Comparison of Entropy between Original and After Images")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()

    fig.tight_layout()

    plt.savefig("entropy_comparison.png")
    plt.show()


# def measure_quality_1(image, reference_gray_value=128):
#     sobel_kernels = {
#         0: np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
#         45: np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]]),
#         90: np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
#         135: np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]]),
#     }

#     edges = []
#     for angle, kernel in sobel_kernels.items():
#         plt.imshow(cv2.filter2D(image, cv2.CV_64F, kernel), cmap="gray")
#         plt.show()
#         edges.append(cv2.filter2D(image, cv2.CV_64F, kernel))

#     weighted_average = np.mean(edges, axis=0)
#     gw = np.mean(weighted_average)
#     Qg = 1 - abs(gw - reference_gray_value) / 255
#     Qg *= 255

#     return Qg


# def measure_quality_2(image):
#     hist = cv2.calcHist([image], [0], None, [256], [0, 256])
#     hist_norm = hist.ravel() / hist.sum()
#     hist_norm = hist_norm[np.nonzero(hist_norm)]
#     entropy = -np.sum(hist_norm * np.log2(hist_norm))
#     return entropy


# def measure_quality_3(image):
#     mu = np.mean(image)
#     sigma = np.std(image)

#     Q_ENL = mu / sigma

#     return Q_ENL


# def measure_quality_4(image, block_size=5):
#     overall_mean = np.mean(image)

#     height, width = image.shape

#     block_means = []
#     for i in range(0, height, block_size):
#         for j in range(0, width, block_size):
#             block = image[i : i + block_size, j : j + block_size]
#             block_mean = np.mean(block)
#             block_means.append(block_mean)

#     N_b = len(block_means)

#     block_means = np.array(block_means)
#     Q_a = np.sqrt(np.sum((block_means - overall_mean) ** 2) / N_b)

#     return Q_a
