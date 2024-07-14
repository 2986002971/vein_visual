"""
2024/07/12 -- 代码原型，高斯金字塔与拉普拉斯卷积与上采样
"""

import cv2

# 读取图像
image = cv2.imread("750-1.png", cv2.IMREAD_GRAYSCALE)

# 创建高斯金字塔
G = image.copy()
for i in range(6):
    G = cv2.pyrDown(G)
cv2.imshow("Down", G)

lp = cv2.Laplacian(G, cv2.CV_64F)
cv2.imshow("Down_Laplacian", lp)

for i in range(6):
    lp = cv2.pyrUp(lp)
cv2.imshow("Up_Laplacian", lp)

cv2.waitKey(0)
cv2.destroyAllWindows()
