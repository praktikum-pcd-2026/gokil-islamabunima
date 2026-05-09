[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/W0VO56Ws)
# Modul 2 : Ekualisasi dan Spesifikasi

## Peraturan
### Tidak Boleh menggunakan fungsi dari cv2 selain untuk membaca dan mengubah image ke grayscale

### IMPORT LIBRARAY
---

```python
# استيراد المكتبات الأساسية
import numpy as np
import matplotlib.pyplot as plt
import cv2
```
Here is why we need to import these libraries:
`numpy` is used for efficiently manipulating arrays and matrices without using built-in functions, which is crucial since images are represented as arrays.
`matplotlib.pyplot` is used to visualize the image outputs and plot the histograms before and after processing.
`cv2` (OpenCV) is imported only to fulfill the basic requirements allowed by the assignment rules: reading images from the local path using `cv2.imread()` and converting the base color format to grayscale using `cv2.cvtColor()`.

### LOAD GAMBAR

```python
# قراءة الصور المطلوبة من المجلد
windut_img = cv2.imread('Assets/Windut.png')
# تحويل مسار الألوان من BGR إلى RGB باستخدام تقطيع المصفوفات (slicing)
windut_rgb = windut_img[:, :, ::-1]

bunga_img = cv2.imread('Assets/Bunga.png')
bunga_rgb = bunga_img[:, :, ::-1]

langit_img = cv2.imread('Assets/Langit.png')
langit_rgb = langit_img[:, :, ::-1]
```
The "Load Image" stage aims to retrieve images from local storage. By default, images read by OpenCV have a BGR (Blue-Green-Red) color channel format. Therefore, we reverse the image array channel order using the slicing technique `[:, :, ::-1]` so it becomes RGB (Red-Green-Blue), ensuring the image colors look natural and correct when plotted using matplotlib.

### Proses 1

Process 1 (Image Slicing and Reconstruction) is needed to demonstrate basic matrix data structure manipulation. In this process, we divide the main image (Windut) into four equal quadrants using the array slicing technique `[:mid_h, :mid_w]`. Then, we reconstruct them back together into an empty matrix created with `np.zeros()` to form the complete image again. The output of this step visualizes the four separate slices and the successfully reconstructed combined image.

```python
# 1. حساب أبعاد الصورة لإيجاد نقطة المنتصف
h, w, c = windut_rgb.shape
mid_h, mid_w = h // 2, w // 2

# 2. تقسيم الصورة إلى 4 أرباع باستخدام Slicing
q1 = windut_rgb[:mid_h, :mid_w]       # الربع العلوي الأيسر
q2 = windut_rgb[:mid_h, mid_w:]       # الربع العلوي الأيمن
q3 = windut_rgb[mid_h:, :mid_w]       # الربع السفلي الأيسر
q4 = windut_rgb[mid_h:, mid_w:]       # الربع السفلي الأيمن

# 3. عرض الأجزاء الأربعة
fig, axs = plt.subplots(2, 2, figsize=(6, 6))
axs[0, 0].imshow(q1); axs[0, 0].set_title("Top Left")
axs[0, 1].imshow(q2); axs[0, 1].set_title("Top Right")
axs[1, 0].imshow(q3); axs[1, 0].set_title("Bottom Left")
axs[1, 1].imshow(q4); axs[1, 1].set_title("Bottom Right")
for ax in axs.flat:
    ax.axis('off')
plt.tight_layout()
plt.show()

# 4. إعادة بناء الصورة باستخدام np.zeros
reconstructed = np.zeros((h, w, c), dtype=np.uint8)
reconstructed[:mid_h, :mid_w] = q1
reconstructed[:mid_h, mid_w:] = q2
reconstructed[mid_h:, :mid_w] = q3
reconstructed[mid_h:, mid_w:] = q4

# 5. عرض الصورة المُعاد بناؤها
plt.figure(figsize=(4, 4))
plt.imshow(reconstructed)
plt.title("Reconstructed Image")
plt.axis('off')
plt.show()
```

### Proses 2

Process 2 (Manual Histogram Equalization) is required because the image might have poor contrast or an uneven light intensity distribution. In this process, we convert the image to grayscale and manually compute the frequency of each pixel value (Histogram) and the Cumulative Distribution Function (CDF) using loops, without using OpenCV's built-in functions. Then, we map the original intensities to new, more evenly distributed intensities to enhance the image contrast. The final output displays a comparison between the original image and the equalized image, along with their respective histograms.

```python
# تحويل الصورة إلى الأبيض والأسود (Grayscale) باستخدام cv2 كاستثناء مسموح
windut_gray = cv2.cvtColor(windut_img, cv2.COLOR_BGR2GRAY)

# دوال يدوية لحساب الـ Histogram
def get_manual_histogram(image):
    hist = [0] * 256
    rows, cols = image.shape
    for i in range(rows):
        for j in range(cols):
            hist[image[i, j]] += 1
    return hist

# دوال يدوية لحساب الـ CDF
def get_manual_cdf(hist):
    cdf = [0] * 256
    total = 0
    for i in range(256):
        total += hist[i]
        cdf[i] = total
    return cdf

hist_original = get_manual_histogram(windut_gray)
cdf_original = get_manual_cdf(hist_original)

# إنشاء مصفوفة التحويل (Mapping) لتسوية المدرج التكراري
total_pixels = windut_gray.shape[0] * windut_gray.shape[1]
mapping_eq = [0] * 256
for i in range(256):
    mapping_eq[i] = int(np.round((cdf_original[i] / total_pixels) * 255))

# تطبيق خريطة التحويل على الصورة
windut_equalized = np.zeros_like(windut_gray)
for i in range(windut_gray.shape[0]):
    for j in range(windut_gray.shape[1]):
        windut_equalized[i, j] = mapping_eq[windut_gray[i, j]]

hist_equalized = get_manual_histogram(windut_equalized)

# عرض الصور والرسوم البيانية
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].imshow(windut_gray, cmap='gray'); axs[0, 0].set_title("Original Grayscale"); axs[0, 0].axis('off')
axs[0, 1].imshow(windut_equalized, cmap='gray'); axs[0, 1].set_title("Equalized Image"); axs[0, 1].axis('off')
axs[1, 0].plot(hist_original, color='blue'); axs[1, 0].set_title("Histogram Before Equalization")
axs[1, 1].plot(hist_equalized, color='red'); axs[1, 1].set_title("Histogram After Equalization")
plt.tight_layout()
plt.show()
```

### Proses 3

Process 3 (Manual Histogram Specification & Merging) is needed because we want to match the color and intensity distribution properties of the main image with a reference image (Bunga Ireng). In this stage, we manually compute the cumulative probabilities to find the closest minimum difference as the specification mapping. In addition, this process involves cropping the image and separating the Windut object from its background using manual color thresholding. Finally, we merge the extracted object onto the Langit (Sky) background to form a complete and visually appealing final image composition.

```python
# 1. التحضير للمطابقة: حساب المدرج التكراري و CDF للصورة المرجعية
bunga_gray = cv2.cvtColor(bunga_img, cv2.COLOR_BGR2GRAY)
hist_ref = get_manual_histogram(bunga_gray)
cdf_ref = get_manual_cdf(hist_ref)
total_pixels_ref = bunga_gray.shape[0] * bunga_gray.shape[1]

# تحويل لقيم احتمالية للمقارنة
cdf_norm_original = [c / total_pixels for c in cdf_original]
cdf_norm_ref = [c / total_pixels_ref for c in cdf_ref]

# 2. بناء مصفوفة المطابقة (Specification Mapping)
mapping_spec = [0] * 256
for i in range(256):
    min_diff = abs(cdf_norm_original[i] - cdf_norm_ref[0])
    best_match = 0
    for j in range(1, 256):
        diff = abs(cdf_norm_original[i] - cdf_norm_ref[j])
        if diff < min_diff:
            min_diff = diff
            best_match = j
    mapping_spec[i] = best_match

windut_specified = np.zeros_like(windut_gray)
for i in range(windut_gray.shape[0]):
    for j in range(windut_gray.shape[1]):
        windut_specified[i, j] = mapping_spec[windut_gray[i, j]]

hist_specified = get_manual_histogram(windut_specified)

# عرض النتائج
fig, axs = plt.subplots(3, 2, figsize=(8, 10))
axs[0, 0].imshow(windut_gray, cmap='gray'); axs[0, 0].set_title("Source Image")
axs[0, 1].plot(hist_original, color='blue'); axs[0, 1].set_title("Source Histogram")
axs[1, 0].imshow(bunga_gray, cmap='gray'); axs[1, 0].set_title("Reference Image")
axs[1, 1].plot(hist_ref, color='green'); axs[1, 1].set_title("Reference Histogram")
axs[2, 0].imshow(windut_specified, cmap='gray'); axs[2, 0].set_title("Specified Image")
axs[2, 1].plot(hist_specified, color='purple'); axs[2, 1].set_title("Specified Histogram")
for ax in axs[:, 0]: ax.axis('off')
plt.tight_layout()
plt.show()

# 3. القص والدمج الملون النهائي (Cropping & Merging RGB)
min_h = min(windut_rgb.shape[0], langit_rgb.shape[0])
min_w = min(windut_rgb.shape[1], langit_rgb.shape[1])
windut_crop_color = windut_rgb[:min_h, :min_w]
langit_crop_color = langit_rgb[:min_h, :min_w]

final_color = np.zeros((min_h, min_w, 3), dtype=np.uint8)
for i in range(min_h):
    for j in range(min_w):
        r, g, b = windut_crop_color[i, j]
        # عزل الخلفية: التأكد أن البكسل أبيض كخلفية (قيم عالية بالقنوات الثلاثة)
        if r > 235 and g > 235 and b > 235:
            final_color[i, j] = langit_crop_color[i, j]
        else:
            final_color[i, j] = windut_crop_color[i, j]

# عرض النتيجة النهائية بعد الدمج
plt.figure(figsize=(6, 6))
plt.imshow(final_color)
plt.title("Final Color Merged Image (Windut Cipularang)")
plt.axis('off')
plt.show()
```

# Note : Cell dengan output gambar penjelasan di atas cell, Cell tanpa output gambar penjelasan di bawah cell
