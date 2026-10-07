import cv2
from skimage.feature import graycomatrix, graycoprops

img = cv2.imread("texture.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

glcm = graycomatrix(
    img,
    distances=[1],
    angles=[0],
    levels=256,
    symmetric=True,
    normed=True
)

contrast = graycoprops(glcm, 'contrast')[0, 0]
homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
energy = graycoprops(glcm, 'energy')[0, 0]
correlation = graycoprops(glcm, 'correlation')[0, 0]

print("Contrast:", contrast)
print("Homogeneity:", homogeneity)
print("Energy:", energy)
print("Correlation:", correlation)