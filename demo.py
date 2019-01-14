import os
import warnings
import time
import PIL
import numpy as np
import matplotlib.pyplot as plt

from tools.image import Image
from tools.color_ot import color_transfer


warnings.filterwarnings('ignore')

images_path = os.path.join(os.getcwd(), 'images')
results_path = os.path.join(os.getcwd(), 'results')

source_img = 'flower_1.jpg'
ref_img = 'flower_2.jpg'

u = Image(os.path.join(images_path, source_img))
v = Image(os.path.join(images_path, ref_img))

num_iter = 1000
tau = 1e-4
rho = 2e-1
mu = 5e-1
alpha = 1e-3
beta = 0.0
num_neighbors = 50
num_segments = 300
sigma = 0.15

start_time = time.time()
w = color_transfer(u, v, num_iter, tau, rho, mu, alpha, beta, num_neighbors, num_segments, sigma)
end_time = time.time()

print("Synthesis completed in {:.2f} seconds".format(end_time - start_time))

raw_ot_w = color_transfer(u, v, 0, tau, rho, mu, alpha, beta, num_neighbors, num_segments, sigma)

relaxed_ot = PIL.Image.fromarray(np.array(255 * w, dtype='uint8'))
raw_ot = PIL.Image.fromarray(np.array(255 * raw_ot_w, dtype='uint8'))

num_results = len(os.listdir(results_path))
new_results_path = os.path.join(results_path, 'results{}'.format(num_results + 1))

if not os.path.exists(new_results_path):
    os.makedirs(new_results_path)

relaxed_ot.save(os.path.join(new_results_path, 'relaxed_ot.png'))
raw_ot.save(os.path.join(new_results_path, 'raw_ot.png'))

file = open(os.path.join(new_results_path, 'params.txt'), 'w')

file.write('source : {}\n'.format(source_img))
file.write('ref : {}\n'.format(ref_img))
file.write('num_iter : {}\n'.format(num_iter))
file.write('tau : {}\n'.format(tau))
file.write('rho : {}\n'.format(rho))
file.write('mu : {}\n'.format(mu))
file.write('alpha : {}\n'.format(alpha))
file.write('beta : {}\n'.format(beta))
file.write('num_neighbors : {}\n'.format(num_neighbors))
file.write('num_segments : {}\n'.format(num_segments))
file.write('sigma : {}'.format(sigma))

file.close()

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

ax1.imshow(u.array)
ax1.axis('off')
ax1.set_title('original image')

ax2.imshow(v.array)
ax2.axis('off')
ax2.set_title('target image')

ax3.imshow(w)
ax3.axis('off')
ax3.set_title('synthesized image')

ax4.imshow(raw_ot_w)
ax4.axis('off')
ax4.set_title('raw optimal transport')

plt.show()
