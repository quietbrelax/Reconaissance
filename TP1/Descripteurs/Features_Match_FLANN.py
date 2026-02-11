import numpy as np
import cv2
from matplotlib import pyplot as plt

def warp_image(img, M):
    rows, cols = img.shape[:2]
    return cv2.warpAffine(img, M, (cols, rows))

def evaluate_matches(kp1, kp2, good_matches, M):
    erreurs = []
    for m in good_matches:
        # ponto original
        p1 = np.array([kp1[m[0].queryIdx].pt[0], kp1[m[0].queryIdx].pt[1], 1])
        # posição verdadeira após transformação
        p1_proj = M @ p1
        # ponto detectado
        p2 = np.array(kp2[m[0].trainIdx].pt)
        # erro euclidiano
        erreurs.append(np.linalg.norm(p2 - p1_proj[:2]))
    erreurs = np.array(erreurs)
    return erreurs

#######################################
#             Paramètres             #
#######################################
angle = 15          # rotation en degrés
tx, ty = 10, 5      # translation en pixels
scale = 1.0         # échelle

img1 = cv2.imread('Image_Pairs/torb_small1.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

theta = np.deg2rad(30)
s = 1.2
tx, ty = 40, 20

M = np.array([
    [s*np.cos(theta), -s*np.sin(theta), tx],
    [s*np.sin(theta),  s*np.cos(theta), ty]
], dtype=np.float32)

img2 = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

###############
#     ORB     #
###############

t1 = cv2.getTickCount()
kp1 = cv2.ORB_create(nfeatures=500).detectAndCompute(gray1, None)[0]
kp1, desc1 = cv2.ORB_create(nfeatures=500).detectAndCompute(gray1,None)
kp2, desc2 = cv2.ORB_create(nfeatures=500).detectAndCompute(gray2,None)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Détection points et calcul descripteurs :",time,"s")

# Appariment avec ratio test
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(desc1, desc2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append([m])

# Evaluation quantitative
erreurs = evaluate_matches(kp1, kp2, good, M)
print("ORB: Erreur moyenne =", np.mean(erreurs), "px")
print("ORB: % de matches < 3 px =", np.mean(erreurs<3)*100, "%")

# Affichage
img_matches = cv2.drawMatchesKnn(gray1,kp1,gray2,kp2,good,None,
                                 matchColor=(0,255,0), singlePointColor=(255,0,0), flags=0)
plt.imshow(img_matches)
plt.title('ORB - Matches après transformation')
plt.show()

################
#     KAZE     #
################
print("###################################################\n")

t1 = cv2.getTickCount()
kp1, desc1 = cv2.KAZE_create().detectAndCompute(gray1,None)
kp2, desc2 = cv2.KAZE_create().detectAndCompute(gray2,None)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Détection points et calcul descripteurs :",time,"s")

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(desc1, desc2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append([m])

erreurs = evaluate_matches(kp1, kp2, good, M)
print("KAZE: Erreur moyenne =", np.mean(erreurs), "px")
print("KAZE: % de matches < 3 px =", np.mean(erreurs<3)*100, "%")

img_matches = cv2.drawMatchesKnn(gray1,kp1,gray2,kp2,good,None,
                                 matchColor=(0,255,0), singlePointColor=(255,0,0), flags=0)
plt.imshow(img_matches)
plt.title('KAZE - Matches après transformation')
plt.show()

