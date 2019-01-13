import PIL
import numpy as np

from skimage.segmentation import slic  # super pixels


class Image:
    """
    Class to handle images
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

    def _compute_clusters(self, num_segments):
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
        num_pixels = self.array.size
        hist = []
        features = []

        for cluster in self.clusters:
            hist.append(cluster['num_pixels'] / num_pixels)
            features.append(np.hstack([cluster['spatial_mean'], cluster['color_mean']]))

        self.hist = np.array(hist)
        self.features = np.array(features)

    def _compute_graph(self, num_neighbors):
        num_features = self.features.shape[0]

        self.weights = np.array([[1 / np.linalg.norm(self.features[i] - self.features[j]) for j in range(num_features)]
                                 for i in range(num_features)])

        for i in range(num_features):
            sorted_weights = np.sort(self.weights[i])



    def _compute_quantized_image(self):
        quantized_array = np.zeros_like(self.array)

        for i in range(self.array.shape[0]):
            for j in range(self.array.shape[1]):
                quantized_array[i, j] = self.clusters[self.segments[i, j]]['color_mean']

        self.quantized_array = quantized_array

    def quantize(self, num_segments=500, num_neighbors=10):
        self._compute_clusters(num_segments)
        self._compute_histogram()
        self._comute_graph(num_neighbors)
        self._compute_quantized_image()

        self.is_quantized = True


def simplex_projection(x):
    """
    Projection of a vector x on the simplex
    """

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

    # assert the projection is in the simplex
    assert (np.allclose(projection.dot(np.ones_like(projection)), 1))

    return projection


def coupling_matrix_projection(P, hist):
    """
    Projection of a coupling matrix P on the set of right stochastic matrices with "marginal" hist
    """

    projection = np.copy(P)

    for i in range(P.shape[0]):
        projection[i] = hist[i] * simplex_projection(P[i])

    return projection


def compute_transport_map(u, v, num_iter, tau, rho, mu, alpha, beta):
    assert(u.is_quantized and v.is_quantized)

    n, m = u.features.shape[0], v.features.shape[0]

    # cost only taking color features into account
    C = np.array([[np.linalg.norm(u.features[i, 2:] - v.features[j, 2:]) ** 2 for j in range(m)] for i in range(n)])

    P = coupling_matrix_projection(np.random.rand(n, m), u.hist)
    former_P = P

    for _ in range(num_iter):
        previous_update = P - former_P
        former_P = P

        grad_F = np.ones(n) @ P @ np.diag(1 / v.hist)  # fidelity term
        grad_R = 0  # regularization term

        Y = v.features[:, 2:]
        grad_D = np.ones(m) @ np.diag(Y @ Y.T).T - 2 * np.diag(u.hist) @ P @ Y @ Y.T  # dispertion term

        P = P - tau * (C + rho * grad_F + mu * grad_R + alpha * grad_D) - beta * previous_update
        P = coupling_matrix_projection(P, u.hist)

    assert (np.allclose(P.dot(np.ones(m)), u.hist))

    transport_map = np.diag(1 / u.hist) @ P @ v.features[:, 2:]

    return transport_map


def post_processing(u, transport_map, sigma):
    assert(u.is_quantized)

    w = np.zeros_like(u.array)  # synthesis result

    for i in range(u.array.shape[0]):
        for j in range(u.array.shape[1]):

            point_feature = np.hstack([np.array([i / u.array.shape[0], j / u.array.shape[1]]), u.array[i, j]])

            clusters_weight = np.exp(- np.linalg.norm(u.features - point_feature[np.newaxis, :], axis=1)
                                     / (2 * sigma ** 2))

            w[i, j] = np.sum(clusters_weight[:, np.newaxis] * transport_map, axis=0) / np.sum(clusters_weight)

    return w


def color_transfer(u, v, num_iter=10, tau=0.1, rho=1e2, mu=0.0, alpha=0.0, beta=0.5, num_segments=300, sigma=0.1):
    if not u.is_quantized:
        u.quantize(num_segments)

    if not v.is_quantized:
        v.quantize(num_segments)

    transport_map = compute_transport_map(u, v, num_iter, tau, rho, mu, alpha, beta)

    return post_processing(u, transport_map, sigma)


