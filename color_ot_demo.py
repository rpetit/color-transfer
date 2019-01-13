import os
import warnings
import time
import matplotlib.pyplot as plt

from color_ot import Image, color_transfer


warnings.filterwarnings('ignore')

images_path = os.path.join(os.getcwd(), 'images')

u = Image(os.path.join(images_path, 'parrot_1.jpg'))
v = Image(os.path.join(images_path, 'parrot_2.jpg'))

num_iter = 100
rho = 1.0
tau = 0.1
mu = 0.0
alpha = 1e-3
beta = 0.0
num_segments = 500
sigma = 0.1

start_time = time.time()
w = color_transfer(u, v, num_iter, tau, rho, mu, alpha, beta, num_segments, sigma)
end_time = time.time()

print("Synthesis completed in {:.2f} seconds".format(end_time - start_time))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.imshow(u.array)
ax1.axis('off')
ax1.set_title('original image')

ax2.imshow(v.array)
ax2.axis('off')
ax2.set_title('target image')

ax3.imshow(w)
ax3.axis('off')
ax3.set_title('synthesized image')

plt.show()
