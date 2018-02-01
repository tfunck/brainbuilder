from sklearn.cluster import spectral_clustering
from sklearn.feature_extraction import image

from skimage.measure import block_reduce
from skimage import exposure

from scipy.ndimage import gaussian_filter


from PIL import Image

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

data_dir = 'E:/brains/L_slab_1/'
test_img = 'QF#HG#MR1s1#L#afdx#4266#01.tif'

downsampling_factor = 30

img = np.asarray(Image.open(data_dir + test_img), dtype='uint8')
# downsampled_img = block_reduce(img, block_size=(downsampling_factor, downsampling_factor), func=np.mean)

p2 = np.percentile(img, 2)
p98 = np.percentile(img, 98)
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

smoothed = gaussian_filter(img_rescale, 5)

small_img = smoothed[::downsampling_factor, ::downsampling_factor]
# mask = np.ones(small_img.shape, dtype='bool')

graph = image.img_to_graph(small_img)
graph.data = np.exp(-graph.data / graph.data.std())

print('Original image shape:', img.shape)
# print('Downsampled shape:', downsampled_img.shape)
print('Small image shape:', small_img.shape)
print('Graph shape:', graph.data.shape)

n_clusters = 2
cluster_search = range(2, 10)

plt.imshow(img_rescale, cmap='gray')
plt.axis('off')
plt.savefig(data_dir + 'rescaled.png')
plt.clf()

for n in cluster_search:
    print('Spectral clustering with ' + str(n) + ' clusters...')
    labels = spectral_clustering(graph, n_clusters=n, random_state=42)
    print('Done ' + str(n) + ' clusters.')
    label_img = np.reshape(labels, small_img.shape)

    plt.imshow(label_img, cmap='gray')
    plt.savefig(data_dir + str(n) + '_clusters.png')
    plt.clf()