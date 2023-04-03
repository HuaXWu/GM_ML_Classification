import numpy as np
import cv2
import os

"""
    date 2022.03.01
    by wuhx  
    calculate the means and std
"""

# img_h, img_w = 32, 32
img_h, img_w = 50, 50
means, stdevs = [], []
img_list = []

# input your image path
imgs_path_list = ['/root/autodl-tmp/origin_image/Hepatocirrhosis/negative/sample10.png', '/root/autodl-tmp/origin_image/Hepatocirrhosis/negative/sample103.png', '/root/autodl-tmp/origin_image/Hepatocirrhosis/negative/sample121.png', '/root/autodl-tmp/origin_image/Hepatocirrhosis/negative/sample129.png', '/root/autodl-tmp/origin_image/Hepatocirrhosis/negative/sample32.png', '/root/autodl-tmp/origin_image/Hepatocirrhosis/negative/sample35.png', '/root/autodl-tmp/origin_image/Hepatocirrhosis/negative/sample36.png', '/root/autodl-tmp/origin_image/Hepatocirrhosis/negative/sample39.png', '/root/autodl-tmp/origin_image/Hepatocirrhosis/negative/sample50.png', '/root/autodl-tmp/origin_image/Hepatocirrhosis/negative/sample57.png', '/root/autodl-tmp/origin_image/Hepatocirrhosis/negative/sample79.png', '/root/autodl-tmp/origin_image/Hepatocirrhosis/negative/sample98.png', '/root/autodl-tmp/origin_image/Hepatocirrhosis/positive/sample101.png', '/root/autodl-tmp/origin_image/Hepatocirrhosis/positive/sample106.png', '/root/autodl-tmp/origin_image/Hepatocirrhosis/positive/sample117.png', '/root/autodl-tmp/origin_image/Hepatocirrhosis/positive/sample127.png', '/root/autodl-tmp/origin_image/Hepatocirrhosis/positive/sample17.png', '/root/autodl-tmp/origin_image/Hepatocirrhosis/positive/sample20.png', '/root/autodl-tmp/origin_image/Hepatocirrhosis/positive/sample26.png', '/root/autodl-tmp/origin_image/Hepatocirrhosis/positive/sample56.png', '/root/autodl-tmp/origin_image/Hepatocirrhosis/positive/sample66.png', '/root/autodl-tmp/origin_image/Hepatocirrhosis/positive/sample8.png', '/root/autodl-tmp/origin_image/Hepatocirrhosis/positive/sample87.png', '/root/autodl-tmp/origin_image/Hepatocirrhosis/positive/sample88.png', '/root/autodl-tmp/origin_image/Hepatocirrhosis/positive/sample9.png']


len_ = len(imgs_path_list)
i = 0
for item in imgs_path_list:
    img = cv2.imread( item)
    img = cv2.resize(img, (img_w, img_h))
    img = img[:, :, :, np.newaxis]
    img_list.append(img)
    i += 1
    # print(i, '/', len_)

imgs = np.concatenate(img_list, axis=3)
imgs = imgs.astype(np.float32) / 255.

for i in range(3):
    pixels = imgs[:, :, i, :].ravel()
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

# BGR --> RGB
means.reverse()
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
