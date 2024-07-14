import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

image = cv2.imread("e8648394651f2f9e6e579314ba05a05.png", cv2.IMREAD_GRAYSCALE)
max_val = image.max()
hist, bins = np.histogram(image.flatten(), 256, [0, 256])
cdf = hist.cumsum()
total_pixels = image.size
high_threshold = np.searchsorted(cdf, total_pixels * 0.4)
high_light_suppressed = image.copy()
high_light_suppressed[high_light_suppressed > high_threshold] = high_threshold

edge = cv2.Canny(high_light_suppressed, 50, 150)

image_equalized = cv2.equalizeHist(high_light_suppressed)

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(image, cmap="gray", norm=mcolors.Normalize(vmin=0, vmax=max_val))
plt.title("Original Image")
plt.subplot(132)
plt.imshow(
    high_light_suppressed, cmap="gray", norm=mcolors.Normalize(vmin=0, vmax=max_val)
)
plt.title("High Light Suppressed Image")
plt.subplot(133)
plt.imshow(edge, cmap="gray", norm=mcolors.Normalize(vmin=0, vmax=max_val))
plt.title("Edge")
plt.show()

cv2.imwrite("suppression.png", high_light_suppressed)
