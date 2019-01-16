import PIL
import numpy as np

from skimage.segmentation import slic  # super pixels
from scipy.sparse import csr_matrix  # sparse matrices for graph weights


class Image:
    """
    Class to handle quantization of images
    """

    def __init__(self, path):
        pil_image = PIL.Image.open(path)
        self.array = np.array(pil_image, dtype='float') / 255

        self.segments = None
        self.clusters = None
        self.hist = None
        self.features = None
        self.quantized_array = None

        self.is_quantized = False

        self.weights = None
        self.G = None

    def _compute_clusters(self, num_segments):
        """
        performs a segmentation in (approximately) num_segments super-pixels (or clusters), and compute the spatial
        and color means, as well as the size of each of them
        """

        segments = slic(self.array, n_segments=num_segments, sigma=5)
        clusters = [{} for _ in range(np.unique(segments).size)]

        for i in range(segments.shape[0]):
            for j in range(segments.shape[1]):
                cluster_idx = segments[i, j]

                if clusters[cluster_idx] == {}:
                    clusters[cluster_idx]['spatial_mean'] = np.array([i / self.array.shape[0],
                                                                      j / self.array.shape[1]])
                    clusters[cluster_idx]['color_mean'] = self.array[i, j]
                    clusters[cluster_idx]['num_pixels'] = 1
                else:
                    clusters[cluster_idx]['spatial_mean'] += np.array([i / self.array.shape[0],
                                                                       j / self.array.shape[1]])
                    clusters[cluster_idx]['color_mean'] += self.array[i, j]
                    clusters[cluster_idx]['num_pixels'] += 1

        for cluster_idx in range(len(clusters)):
            clusters[cluster_idx]['spatial_mean'] *= 1.0 / clusters[cluster_idx]['num_pixels']
            clusters[cluster_idx]['color_mean'] *= 1.0 / clusters[cluster_idx]['num_pixels']

        self.segments = segments
        self.clusters = clusters

    def _compute_histogram(self):
        """
        compute the feature vector and the weight of each cluster
        """

        num_pixels = self.array.shape[0] * self.array.shape[1]
        hist = []
        features = []

        for cluster in self.clusters:
            hist.append(cluster['num_pixels'] / num_pixels)
            features.append(np.hstack([cluster['spatial_mean'], cluster['color_mean']]))

        self.hist = np.array(hist)
        self.features = np.array(features)

        # matrices used to compute the transport problem's objective
        self.Dh = np.diag(1 / self.hist)
        self.Dh_inv = np.diag(self.hist)

        assert(np.sum(self.hist) == 1)

    def _compute_quantized_image(self):
        """
        assign to all pixels of a given cluster its mean color (only useful for visualization)
        """

        quantized_array = np.zeros_like(self.array)

        for i in range(self.array.shape[0]):
            for j in range(self.array.shape[1]):
                quantized_array[i, j] = self.clusters[self.segments[i, j]]['color_mean']

        self.quantized_array = quantized_array

    def quantize(self, num_segments):
        self._compute_clusters(num_segments)
        self._compute_histogram()
        self._compute_quantized_image()

        self.is_quantized = True

    def build_graph(self, num_neighbors):
        """
        build a graph whose vertices are clusters and edges have a weight proportional to their features' similarity
        this graph is used through the gradient it defines to regularize the transport map
        """

        num_features = self.features.shape[0]

        # weights[i, j] measures the similarity between cluster i and j
        weights = np.array([[0 if i == j else 1 / np.linalg.norm(self.features[i] - self.features[j])
                             for j in range(num_features)]
                            for i in range(num_features)])

        graph_grad = np.zeros((num_features, num_features, num_features))

        for i in range(num_features):
            ranked_neighbors = np.argsort(weights[i])[::-1]  # only keeps

            for j in range(num_neighbors):  # only keep top neighbors for computational efficiency
                graph_grad[i, ranked_neighbors[j], i] = weights[i, ranked_neighbors[j]]
                graph_grad[i, ranked_neighbors[j], ranked_neighbors[j]] = - weights[i, ranked_neighbors[j]]

        graph_grad = np.reshape(graph_grad, (num_features ** 2, num_features))
        self.G = csr_matrix(graph_grad)
