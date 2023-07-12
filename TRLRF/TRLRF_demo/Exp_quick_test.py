# Completion of a 256*256*3 image by TRLRF


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from TR_functions import gen_W, RSE_fun
import TRLRF

# Load data
image_path = 'lena.bmp'
im = Image.open(image_path)

image = np.array(im, dtype='double')
X = image / 255

mr = 0.5  # missing rate

W = gen_W(X.shape, mr)


# TRLRF
r = 5 * np.ones(3, dtype='int')  # TR-rank
maxiter = 300  # maxiter 300~500
tol = 1e-6  # 1e-6~1e-8
Lambda = 5  # usually 1~10
ro = 1.1  # 1~1.5
K = 1e0  # 1e-1~1e0

X_hat, _, __ = TRLRF.TRLRF(X, W, r, maxiter, K, ro, Lambda, tol)

# Evaluation
RSE = RSE_fun(X, X_hat, W)

print('Completion RSE is %s' % RSE[0])

image_o = np.array(image, dtype='uint8')
image_noise = np.array(255 * W * X, dtype='uint8')
image_hat = np.array(X_hat*255, dtype='uint8')
# print(image_hat.shape)


plt.figure()
plt.subplot(141)
plt.imshow(im)
plt.title('Original')

plt.subplot(142)
plt.imshow(image_noise)
plt.title("%s %% MISSING" % str(mr*100))

plt.subplot(143)
#plt.imshow(np.array(image_hat, dtype='double'))
plt.imshow(image_hat)
plt.title('Completion')

plt.subplot(144)
plt.imshow(W)
plt.title('debug')

plt.show()
