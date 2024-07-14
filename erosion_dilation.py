import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image_path = "750-1.png"
image = cv2.imread(image_path, 0)
cv2.imshow("image", image)

kernel = np.ones((9, 9), np.uint8)

for i in range(1):
    image = cv2.erode(image, kernel)
cv2.imshow("erode1", image)

for i in range(3):
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
cv2.imshow("close1", image)

image = cv2.GaussianBlur(image, (9, 9), 0)
lp = cv2.Laplacian(image, cv2.CV_64F)
cv2.imshow("lp", lp)


# opening1 = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel1)
# opening2 = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel2)
# cv2.imshow("opening1", opening1)
# cv2.imshow("opening2", opening2)

cv2.waitKey(0)
cv2.destroyAllWindows()
