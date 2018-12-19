import PIL
import numpy as np

from skimage.segmentation import slic  # super pixels


class Image:

    def __init__(self, path):
        pil_image = PIL.Image.open(path)
        self.array = np.array(pil_image, dtype='float') / 255

        self.segments = None
        self.clusters = None
        self.hist = None
        self.features = None
        self.quantized_array = None

        self.is_quantized = False

    def _compute_clusters(self, num_segments):
        segments = slic(self.array, n_segments=num_segments, sigma=5)
        clusters = [{} for _ in range(np.unique(segments).size)]

        for i in range(segments.shape[0]):
            for j in range(segments.shape[1]):
                cluster_idx = segments[i, j]

                if clusters[cluster_idx] == {}:
                    clusters[cluster_idx]['spatial_mean'] = np.array([i, j], dtype='float')
                    clusters[cluster_idx]['color_mean'] = self.array[i, j]
                    clusters[cluster_idx]['num_pixels'] = 1
                else:
                    clusters[cluster_idx]['spatial_mean'] += np.array([i, j])
                    clusters[cluster_idx]['color_mean'] += self.array[i, j]
                    clusters[cluster_idx]['num_pixels'] += 1

        for cluster_idx in range(len(clusters)):
            clusters[cluster_idx]['spatial_mean'] *= 1.0 / clusters[cluster_idx]['num_pixels']
            clusters[cluster_idx]['color_mean'] *= 1.0 / clusters[cluster_idx]['num_pixels']

        self.segments = segments
        self.clusters = clusters

    def _compute_histogram(self):
        num_pixels = self.array.size
        hist = []
        features = []

        for cluster in self.clusters:
            hist.append(cluster['num_pixels'] / num_pixels)
            features.append(np.hstack([cluster['spatial_mean'], cluster['color_mean']]))

        self.hist = np.array(hist)
        self.features = np.array(features)

    def _compute_quantized_image(self):
        quantized_array = np.zeros_like(self.array)

        for i in range(self.array.shape[0]):
            for j in range(self.array.shape[1]):
                quantized_array[i, j] = self.clusters[self.segments[i, j]]['color_mean']

        self.quantized_array = quantized_array

    def quantize(self, num_segments=500):
        self._compute_clusters(num_segments)
        self._compute_histogram()
        self._compute_quantized_image()

        self.is_quantized = True


def simplex_projection(x):
    n = x.size
    x_sorted = np.sort(x)

    i = n - 2
    t_i = (np.sum(x_sorted[i + 1:]) - 1) / (n - (i + 1))

    while t_i < x_sorted[i] and i >= 0:
        i = i - 1
        t_i = (np.sum(x_sorted[i + 1:n]) - 1) / (n - (i + 1))

    if i < 0:
        t_hat = (np.sum(x) - 1) / n
    else:
        t_hat = t_i

    projection = np.maximum(x - t_hat, 0)
    assert (np.allclose(projection.dot(np.ones_like(projection)), 1))

    return projection


def coupling_matrix_projection(coupling_matrix, hist):
    projection = np.copy(coupling_matrix)

    for i in range(coupling_matrix.shape[0]):
        projection[i] = hist[i] * simplex_projection(coupling_matrix[i])

    return projection


def compute_transport_map(original_image, target_image, num_iter=10, gamma=0.1):
    if not original_image.is_quantized:
        original_image.quantize()

    if not target_image.is_quantized:
        target_image.quantize()

    cost_matrix = np.array([[np.linalg.norm(original_image.features[i, 2:] - target_image.features[j, 2:]) ** 2
                             for j in range(target_image.features.shape[0])]
                            for i in range(original_image.features.shape[0])])

    coupling_matrix = coupling_matrix_projection(np.random.rand(original_image.features.shape[0],
                                                                target_image.features.shape[0]),
                                                 original_image.hist)

    for _ in range(num_iter):
        coupling_matrix = coupling_matrix_projection(coupling_matrix - gamma * cost_matrix,
                                                     original_image.hist)

    assert (np.allclose(coupling_matrix.dot(np.ones(coupling_matrix.shape[1])), original_image.hist))

    transport_map = np.diag(1 / original_image.hist) @ coupling_matrix @ target_image.features[:, 2:]

    return transport_map
