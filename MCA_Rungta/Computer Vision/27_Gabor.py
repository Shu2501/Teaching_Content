import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("texture.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Image not found!")
    exit()


ksize = 31          # Size of Gabor kernel
sigma = 4           # Gaussian spread
theta = 0           # Orientation
lambd = 10          # Wavelength
gamma = 0.5         # Aspect ratio
psi = 0              # Phase offset

kernel = cv2.getGaborKernel(
    (ksize, ksize),
    sigma,
    theta,
    lambd,
    gamma,
    psi,
    ktype=cv2.CV_32F
)

response = cv2.filter2D(
    img,
    cv2.CV_32F,
    kernel
)


response = cv2.normalize(
    response,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

response = np.uint8(response)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(kernel, cmap="gray")
plt.title("Gabor Kernel")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(response, cmap="gray")
plt.title("Gabor Response")
plt.axis("off")

plt.tight_layout()
plt.show()