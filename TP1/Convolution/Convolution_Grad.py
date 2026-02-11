import numpy as np
import cv2

from matplotlib import pyplot as plt

#Lecture image en niveau de gris et conversion en float64
img=np.float64(cv2.imread('Image_Pairs/FlowerGarden2.png',0))
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")

#Methode simple#

kernel_x = np.array([[-1, 0, 1]], dtype=np.float64)
kernel_y = np.array([[-1], [0], [1]], dtype=np.float64)

t1 = cv2.getTickCount()
Ix = cv2.filter2D(img, cv2.CV_64F, kernel_x)
Iy = cv2.filter2D(img, cv2.CV_64F, kernel_y)

grad_norm = np.sqrt(Ix**2 + Iy**2)


# Il faut recaler les valeurs dans l'intervalle [0,255], car elles peuvent être négatives ou supérieures à 255
Ix_disp = cv2.normalize(Ix, None, 0, 255, cv2.NORM_MINMAX)
Iy_disp = cv2.normalize(Iy, None, 0, 255, cv2.NORM_MINMAX)

Ix_disp = Ix_disp.astype(np.uint8)
Iy_disp = Iy_disp.astype(np.uint8)

grad_disp = cv2.normalize(grad_norm, None, 0, 255, cv2.NORM_MINMAX)
grad_disp = grad_disp.astype(np.uint8)

t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode filter2D Gradient Simple:",time,"s")

plt.figure(figsize=(12,4))

plt.subplot(131)
plt.imshow(Ix_disp, cmap='gray')
plt.title("I_x")

plt.subplot(132)
plt.imshow(Iy_disp, cmap='gray')
plt.title("I_y")

plt.subplot(133)
plt.imshow(grad_disp, cmap='gray')
plt.title("||Grad(I)||")

plt.show()


#Method plus robuste#
t1 = cv2.getTickCount()
kx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])/8

ky = np.array([[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]])/8

Ix = cv2.filter2D(img, cv2.CV_64F, kx)
Iy = cv2.filter2D(img, cv2.CV_64F, ky)

plt.subplot(131)
plt.imshow(Ix, cmap='gray')
plt.title("I_x Sobel")

plt.subplot(132)
plt.imshow(Iy, cmap='gray')
plt.title("I_y Sobel")


grad = np.sqrt(Ix**2 + Iy**2)

grad_norm = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)
grad_uint8 = grad_norm.astype(np.uint8)

t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode filter2D Gradient Sobel:",time,"s")

cv2.imshow('Avec filter2D',Ix/255.0)
#Convention OpenCV : une image de type flottant est interprétée dans [0,1]
#cv2.waitKey(0)

plt.subplot(133)
plt.imshow(grad_uint8, cmap='gray')
plt.title("||Grad(I)|| Sobel")

plt.show()
