import cv2
import numpy as np
import matplotlib.pyplot as plt


path = 'D:\\att_faces\\s1\\1.pgm'
poza = cv2.imread(path, 0) # 0 = cv.2.IMREAD_GRAYSCALE
poza = np.array(poza)

poza.reshape(-1,)
print('poza = \n' + str(poza))
print('dim pozei = \n' + str(np.shape(poza)))

A = []
A = np.array(A)
A = np.append(A, poza)
A = A[:, np.newaxis]

plt.imshow(A.reshape(112,92), cmap='gray')
plt.show()