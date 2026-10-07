import cv2
from skimage.feature import local_binary_pattern
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("texture.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

radius = 3
n_points = 8 * radius

lbp = local_binary_pattern(
    gray,
    n_points,
    radius,
    method='uniform'
)

hist, _ = np.histogram(
    lbp.ravel(),
    bins=np.arange(0, n_points + 3),
    range=(0, n_points + 2),
    density=True
)

# Display LBP image
plt.imshow(lbp, cmap='gray')
plt.title("LBP Image")
plt.axis("off")
plt.show()

# Display histogram
plt.bar(range(len(hist)), hist)
plt.title("LBP Histogram")
plt.xlabel("LBP Pattern")
plt.ylabel("Frequency")
plt.show()