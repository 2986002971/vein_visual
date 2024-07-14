"""
2024/07/14 -- 有bug 没写完
"""

import cv2
import numpy as np


def meshgrid(xgv, ygv):
    x = np.arange(xgv[0], xgv[1] + 1)
    y = np.arange(ygv[0], ygv[1] + 1)
    X, Y = np.meshgrid(x, y)
    return X, Y


def conv(src, kernel):
    flipped = cv2.flip(kernel, -1)
    result = cv2.filter2D(src, -1, flipped, borderType=cv2.BORDER_REPLICATE)
    return result


def manual_argmax(d):
    max_val = -np.inf
    pos_max = 0
    for i in range(len(d)):
        if d[i] > max_val:
            max_val = d[i]
            pos_max = i
    return pos_max


def max_curvature(src, mask, sigma):
    src = src.astype(np.float32) / 255.0
    mask = mask.astype(np.uint8)

    sigma2 = sigma**2
    sigma4 = sigma**4

    winsize = int(np.ceil(4 * sigma))
    X, Y = meshgrid((-winsize, winsize), (-winsize, winsize))

    X2 = X**2
    Y2 = Y**2
    X2Y2 = X2 + Y2

    expXY = np.exp(-X2Y2 / (2 * sigma2))
    h = (1 / (2 * np.pi * sigma2)) * expXY

    hx = h * (-X / sigma2)
    temp = (X2 - sigma2) / sigma4
    hxx = h * temp
    hy = hx.T
    hyy = hxx.T
    hxy = h * (X * Y / sigma4)

    fx = -conv(src, hx)
    fxx = conv(src, hxx)
    fy = conv(src, hy)
    fyy = conv(src, hyy)
    fxy = -conv(src, hxy)

    f1 = 0.5 * np.sqrt(2.0) * (fx + fy)
    f2 = 0.5 * np.sqrt(2.0) * (fx - fy)
    f11 = 0.5 * fxx + fxy + 0.5 * fyy
    f22 = 0.5 * fxx - fxy + 0.5 * fyy

    img_h, img_w = src.shape

    k1 = np.zeros_like(src, dtype=np.float32)
    k2 = np.zeros_like(src, dtype=np.float32)
    k3 = np.zeros_like(src, dtype=np.float32)
    k4 = np.zeros_like(src, dtype=np.float32)

    for x in range(img_w):
        for y in range(img_h):
            if mask[y, x]:
                k1[y, x] = fxx[y, x] / ((1 + fx[y, x] ** 2) ** 1.5)
                k2[y, x] = fyy[y, x] / ((1 + fy[y, x] ** 2) ** 1.5)
                k3[y, x] = f11[y, x] / ((1 + f1[y, x] ** 2) ** 1.5)
                k4[y, x] = f22[y, x] / ((1 + f2[y, x] ** 2) ** 1.5)

    Wr = 0
    Vt = np.zeros_like(src, dtype=np.float32)

    for y in range(img_h):
        for x in range(img_w):
            if k1[y, x] > 0:
                Wr += 1
            if Wr > 0 and (x == img_w - 1 or k1[y, x] <= 0):
                pos_end = x if x == img_w - 1 else x - 1
                pos_start = pos_end - Wr + 1
                pos_max = pos_start + manual_argmax(k1[y, pos_start : pos_end + 1])
                Scr = k1[y, pos_max] * Wr
                Vt[y, pos_max] += Scr
                Wr = 0

    for x in range(img_w):
        for y in range(img_h):
            if k2[y, x] > 0:
                Wr += 1
            if Wr > 0 and (y == img_h - 1 or k2[y, x] <= 0):
                pos_end = y if y == img_h - 1 else y - 1
                pos_start = pos_end - Wr + 1
                pos_max = pos_start + manual_argmax(k2[pos_start : pos_end + 1, x])
                Scr = k2[pos_max, x] * Wr
                Vt[pos_max, x] += Scr
                Wr = 0

    for start in range(img_h + img_w - 1):
        if start < img_w:
            x, y = start, 0
        else:
            x, y = 0, start - img_w + 1

        done = False
        while not done:
            if k3[y, x] > 0:
                Wr += 1
            if Wr > 0 and (y == img_h - 1 or x == img_w - 1 or k3[y, x] <= 0):
                pos_x_end = x if x == img_w - 1 else x - 1
                pos_y_end = y if y == img_h - 1 else y - 1
                pos_x_start = pos_x_end - Wr + 1
                pos_y_start = pos_y_end - Wr + 1

                print(
                    f"pos_x_start: {pos_x_start}, pos_x_end: {pos_x_end}, pos_y_start: {pos_y_start}, pos_y_end: {pos_y_end}"
                )  # Debug output

                if pos_y_end >= pos_y_start and pos_x_end >= pos_x_start:
                    d = np.diagonal(
                        k3[pos_y_start : pos_y_end + 1, pos_x_start : pos_x_end + 1]
                    )
                    print(f"d shape: {d.shape}")  # Debug output
                    if d.size > 0:
                        pos_max = manual_argmax(d)
                        pos_x_max = pos_x_start + pos_max
                        pos_y_max = pos_y_start + pos_max
                        Scr = k3[pos_y_max, pos_x_max] * Wr
                        Vt[pos_y_max, pos_x_max] += Scr
                Wr = 0

            if x == img_w - 1 or y == img_h - 1:
                done = True
            else:
                x += 1
                y += 1

    for start in range(img_h + img_w - 1):
        if start < img_w:
            x, y = start, img_h - 1
        else:
            x, y = 0, img_w + img_h - start - 2

        done = False
        while not done:
            if k4[y, x] > 0:
                Wr += 1
            if Wr > 0 and (y == 0 or x == img_w - 1 or k4[y, x] <= 0):
                pos_x_end = x if x == img_w - 1 else x - 1
                pos_y_end = y if y == 0 else y + 1
                pos_x_start = pos_x_end - Wr + 1
                pos_y_start = pos_y_end + Wr - 1

                print(
                    f"pos_x_start: {pos_x_start}, pos_x_end: {pos_x_end}, pos_y_start: {pos_y_start}, pos_y_end: {pos_y_end}"
                )  # Debug output

                if pos_y_start >= pos_y_end and pos_x_end >= pos_x_start:
                    d = np.diagonal(
                        np.flipud(
                            k4[pos_y_end : pos_y_start + 1, pos_x_start : pos_x_end + 1]
                        )
                    )
                    print(f"d shape: {d.shape}")  # Debug output
                    if d.size > 0:
                        pos_max = manual_argmax(d)
                        pos_x_max = pos_x_start + pos_max
                        pos_y_max = pos_y_start - pos_max
                        Scr = k4[pos_y_max, pos_x_max] * Wr
                        if pos_y_max < 0:
                            pos_y_max = 0
                        Vt[pos_y_max, pos_x_max] += Scr
                Wr = 0

            if x == img_w - 1 or y == 0:
                done = True
            else:
                x += 1
                y -= 1

    Cd1 = np.zeros_like(src, dtype=np.float32)
    Cd2 = np.zeros_like(src, dtype=np.float32)
    Cd3 = np.zeros_like(src, dtype=np.float32)
    Cd4 = np.zeros_like(src, dtype=np.float32)

    for x in range(2, src.shape[1] - 3):
        for y in range(2, src.shape[0] - 3):
            Cd1[y, x] = min(
                max(Vt[y, x + 1], Vt[y, x + 2]), max(Vt[y, x - 1], Vt[y, x - 2])
            )
            Cd2[y, x] = min(
                max(Vt[y + 1, x], Vt[y + 2, x]), max(Vt[y - 1, x], Vt[y - 2, x])
            )
            Cd3[y, x] = min(
                max(Vt[y - 1, x - 1], Vt[y - 2, x - 2]),
                max(Vt[y + 1, x + 1], Vt[y + 2, x + 2]),
            )
            Cd4[y, x] = min(
                max(Vt[y - 1, x + 1], Vt[y - 2, x + 2]),
                max(Vt[y + 1, x - 1], Vt[y + 2, x - 2]),
            )

    veins = np.maximum.reduce([Cd1, Cd2, Cd3, Cd4])
    return veins


def main():
    finger = cv2.imread("750-1.png", cv2.IMREAD_GRAYSCALE)
    mask = np.ones_like(finger, dtype=np.uint8)

    result = max_curvature(finger, mask, 8)

    min_val, max_val = np.min(result), np.max(result)
    print(max_val)
    result = (result * 255.0 / max_val * 20).astype(np.uint8)
    min_val, max_val = np.min(result), np.max(result)
    print(max_val)
    cv2.imwrite("result.png", result)


if __name__ == "__main__":
    main()
