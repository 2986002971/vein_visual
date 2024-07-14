import cv2
import numpy as np
from matplotlib import pyplot as plt


def apply_fourier_filter(image, filter_size, filter_weight):
    # 读取图像并转换为灰度图像
    if image is None:
        raise ValueError("图像读取失败，请检查路径是否正确。")

    # 获取图像尺寸
    rows, cols = image.shape

    # 进行傅里叶变换
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 创建滤波器
    mask = np.zeros((rows, cols, 2), np.float32)
    center_row, center_col = rows // 2, cols // 2
    for i in range(rows):
        for j in range(cols):
            if (i - center_row) ** 2 + (j - center_col) ** 2 <= filter_size**2:
                mask[i, j] = filter_weight

    # 应用滤波器
    fshift = dft_shift * mask

    # 进行逆傅里叶变换
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # 归一化图像
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)

    """
    # 显示原始图像和滤波后的图像
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.axis('off')
    plt.subplot(1, 2, 2), plt.imshow(img_back, cmap='gray')
    plt.title('Filtered Image'), plt.axis('off')
    plt.show()
    """
    return img_back


def invert_image(image):
    # 读取图像

    # 检查图像是否成功读取
    if image is None:
        raise ValueError("图像读取失败，请检查。")

    # 黑白反转
    inverted_image = 255 - image

    return inverted_image


def remove_low_high_intensity(
    image_path, low_threshold, high_threshold, filter_size, filter_weight
):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    height, width = image.shape
    fill_value = high_threshold
    """
    # 截取图像的右半部分
    right_half = image[:, width // 2:]
    mask = (right_half >= low_threshold) & (right_half <= high_threshold)
    processedright_image = np.full_like(right_half, fill_value)
    processedright_image[mask] = right_half[mask]
    

    # 将均衡化后的右半部分放回原图像
    processed_image = image.copy()
    processed_image[:, width // 2:] = processedright_image
    # 创建一个掩码，去除灰度值低于 low_threshold 和高于 high_threshold 的部分

    mask = (sharpened_image >= low_threshold) & (sharpened_image <= high_threshold)
    #processedf_image = np.zeros_like(sharpened_image)
    
    """

    fourier_image = apply_fourier_filter(image, filter_size, filter_weight)

    mask1 = fourier_image <= high_threshold
    processed_image1 = np.full_like(fourier_image, fill_value)
    processed_image1[mask1] = fourier_image[mask1]

    # mask2s
    mask2 = processed_image1 >= low_threshold
    processed_image2 = np.full_like(processed_image1, 0)
    processed_image2[mask2] = processed_image1[mask2]

    clahe = cv2.createCLAHE(
        clipLimit=4.0, tileGridSize=(10, 10)
    )  # default:clipLimit=2.0, tileGridSize=(8, 8)

    # 应用 CLAHE 增强对比度
    clahe_image = clahe.apply(processed_image2)

    # 使用拉普拉斯算子进行锐化
    laplacian = cv2.Laplacian(clahe_image, cv2.CV_64F)
    sharpened_image = cv2.convertScaleAbs(clahe_image - laplacian)
    # 再截取一部分灰度阈值

    # 整体mask

    # 显示原始图像和处理后的图像
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("CLAHE Image")
    plt.imshow(clahe_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Processed Image")
    plt.imshow(sharpened_image, cmap="gray")
    plt.axis("off")

    plt.show()

    """
    mask = (image >= low_threshold) & (image <= high_threshold)
    processed_image = np.zeros_like(image)
    processed_image[mask] = image[mask]
    equalized_image = cv2.equalizeHist(processed_image)

    plt.subplot(1, 4, 2)
    plt.title('CLAHE Image')
    plt.imshow(clahe_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title('Sharpened Image')
    plt.imshow(sharpened_image, cmap='gray')
    plt.axis('off')

    
    """
    # 显示原始图像和处理后的图像

    plt.show()


# 示例用法
image_path = "79/tuibuxueguan/16.png"  # 替换为你的图像路径
low_threshold = 30  # 设置低阈值
high_threshold = 210  # 设置高阈值
filter_size = 80  # 滤波器大小
filter_weight = 2  # 滤波器权重
remove_low_high_intensity(
    image_path, low_threshold, high_threshold, filter_size, filter_weight
)
