import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ========= 选择处理模式（可选：original / dark / low_contrast） =========
mode = "original"
# mode = "dark"
# mode = "low_contrast"


# ========= 加载图像并进行预处理 =========
path = r"test.jpg"  # 图像路径
gray_img = cv.imread(path, 0)
color_img = cv.imread(path)

if gray_img is None or color_img is None:
    print("Error: Image not loaded. Check the file path.")
    exit()

# 图像预处理（模拟质量差图像）
if mode == "dark":
    color_img = (color_img * 0.9).astype(np.uint8)
    gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)

elif mode == "low_contrast":
    color_img = cv.normalize(color_img, None, alpha=70, beta=150, norm_type=cv.NORM_MINMAX)
    gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)

# ========= 图像增强处理 =========
# HE：传统均衡化（灰度图）
equalized_img = cv.equalizeHist(gray_img)

# CLAHE：彩色图 Y 通道增强
ycrcb = cv.cvtColor(color_img, cv.COLOR_BGR2YCrCb)
y, cr, cb = cv.split(ycrcb)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
y_clahe = clahe.apply(y)
merged = cv.merge((y_clahe, cr, cb))
clahe_img = cv.cvtColor(merged, cv.COLOR_YCrCb2BGR)

# ========= 工具函数：直方图 + CDF =========
def get_hist_cdf(img):
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    return hist, cdf_normalized

# ========= 绘图：2×4 输出对比图 =========
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 原始灰度图
axs[0, 0].imshow(gray_img, cmap='gray')
axs[0, 0].set_title('Original Grayscale Image')
axs[0, 0].axis('off')
hist, cdf = get_hist_cdf(gray_img)
axs[0, 1].plot(cdf, color='b', label='CDF')
axs[0, 1].hist(gray_img.ravel(), 256, [0, 256], color='r', alpha=0.5)
axs[0, 1].set_title('Histogram (Grayscale)')

# HE 灰度图
axs[0, 2].imshow(equalized_img, cmap='gray')
axs[0, 2].set_title('Equalized Grayscale Image (HE)')
axs[0, 2].axis('off')
hist, cdf = get_hist_cdf(equalized_img)
axs[0, 3].plot(cdf, color='b', label='CDF')
axs[0, 3].hist(equalized_img.ravel(), 256, [0, 256], color='r', alpha=0.5)
axs[0, 3].set_title('Histogram (HE)')

# 原始彩色图
axs[1, 0].imshow(cv.cvtColor(color_img, cv.COLOR_BGR2RGB))
axs[1, 0].set_title('Original Color Image')
axs[1, 0].axis('off')
hist, cdf = get_hist_cdf(y)
axs[1, 1].plot(cdf, color='b', label='CDF')
axs[1, 1].hist(y.ravel(), 256, [0, 256], color='r', alpha=0.5)
axs[1, 1].set_title('Y Channel Histogram (Original)')

# CLAHE 彩色图
axs[1, 2].imshow(cv.cvtColor(clahe_img, cv.COLOR_BGR2RGB))
axs[1, 2].set_title('CLAHE Enhanced Color Image')
axs[1, 2].axis('off')
hist, cdf = get_hist_cdf(y_clahe)
axs[1, 3].plot(cdf, color='b', label='CDF')
axs[1, 3].hist(y_clahe.ravel(), 256, [0, 256], color='r', alpha=0.5)
axs[1, 3].set_title('Y Channel Histogram (CLAHE)')

plt.tight_layout()
plt.show()
