import cv2
import timeit


image = cv2.imread("contrast_pyramid/12.png", cv2.IMREAD_GRAYSCALE)

steps = 5

G = image.copy()
# 使用timeit测量卷积操作的时间
start_time = timeit.default_timer()


for i in range(steps):
    G = cv2.pyrDown(G)

lp = cv2.Laplacian(G, cv2.CV_8U, ksize=3)

for i in range(steps):
    lp = cv2.pyrUp(lp)

_, lp_non_negative = cv2.threshold(lp, 0, 0, cv2.THRESH_TOZERO)
lp_normalized = cv2.normalize(
    lp_non_negative, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
)


end_time = timeit.default_timer()

processing_time = end_time - start_time
print(f"卷积操作耗时: {processing_time:.6f} 秒")
