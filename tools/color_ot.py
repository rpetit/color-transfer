import PIL
import ot
import numpy as np


def pseudo_simplex_projection(x, sum):
    """
    Projection of a vector x on the set of postive vectors whose coefficient sum to sum. This is a slight modification
    of Chen and Ye's algorithm (see their "projection onto a simplex" article)
    """

    n = x.size
    x_sorted = np.sort(x)

    i = n - 2
    t_i = (np.sum(x_sorted[i + 1:]) - sum) / (n - (i + 1))

    while t_i < x_sorted[i] and i >= 0:
        i = i - 1
        t_i = (np.sum(x_sorted[i + 1:n]) - sum) / (n - (i + 1))

    if i < 0:
        t_hat = (np.sum(x) - sum) / n
    else:
        t_hat = t_i

    projection = np.maximum(x - t_hat, 0)

    # assert the projection is valid
    assert (np.allclose(projection.dot(np.ones_like(projection)), sum))

    return projection


def coupling_matrix_projection(P, hist):
    """
    Projection of a coupling matrix P on the set of positive matrices whose rows' sum are given by hist
    """

    projection = np.copy(P)

    for i in range(P.shape[0]):
        projection[i] = pseudo_simplex_projection(P[i], hist[i])

    return projection


def compute_objective(u, v, C, P, rho, mu, alpha):
    X, Y = u.features[:, 2:], v.features[:, 2:]  # color features
    n, m = X.shape[0], Y.shape[0]

    T = u.Dh @ P @ Y  # transport map
    V = T - X  # transport flow
    L = u.l_weights @ V - V * (u.l_weights @ np.ones(n))[:, np.newaxis]  # pseudo laplacian

    F = 0.5 * np.linalg.norm(np.dot(np.ones(n), P @ np.sqrt(v.Dh))) ** 2 - 0.5  # fidelity term
    R = 0.5 * np.sum([u.Dh_inv[i] ** 2 * u.weights[i, j] ** 2 * np.linalg.norm(T[i] - T[j]) ** 2  # regularity term
                      for (i, j) in np.stack(u.weights.nonzero(), axis=1)])
    D = np.sum(P * (np.ones(m) @ np.diag(Y @ Y.T).T - u.Dh_inv @ P @ Y @ Y.T))  # dispersion term

    return np.sum(C * P) + rho * F + mu * R + alpha * D


def compute_objective_grad(u, v, C, P, rho, mu, alpha):
    X, Y = u.features[:, 2:], v.features[:, 2:]  # color features
    n, m = X.shape[0], Y.shape[0]

    T = u.Dh @ P @ Y  # transport map
    V = T - X  # transport flow
    L = u.l_weights @ V - V * (u.l_weights @ np.ones(n))[:, np.newaxis]  # pseudo laplacian

    grad_F = np.ones((n, n)) @ P @ v.Dh  # fidelity term
    grad_R = u.Dh @ L @ Y.T  # regularization term
    grad_D = np.ones(m) @ np.diag(Y @ Y.T).T - 2 * u.Dh_inv @ P @ Y @ Y.T  # dispersion term

    return C + rho * grad_F + mu * grad_R + alpha * grad_D


def compute_transport_map(u, v, num_iter, tau, rho, mu, alpha, beta, num_neighbors):
    assert(u.is_quantized and v.is_quantized)

    X, Y = u.features[:, 2:], v.features[:, 2:]  # color features
    n, m = X.shape[0], Y.shape[0]

    C = np.array([[np.linalg.norm(X[i] - Y[j]) ** 2 for j in range(m)] for i in range(n)])  # cost matrix

    u.build_graph(num_neighbors)

    P = ot.emd(u.hist, v.hist, C)  # initialization as the optimal transport matrix
    assert(np.allclose(coupling_matrix_projection(P, u.hist), P))  # assert the projection behaves normally

    objective = compute_objective(u, v, C, P, rho, mu, alpha)

    former_P = P
    former_objective = objective

    for _ in range(num_iter):
        previous_update = P - former_P
        former_P = P
        former_objective = objective

        objective_grad = compute_objective_grad(u, v, C, P, rho, mu, alpha)

        # projected gradient step
        P = P - tau * objective_grad + beta * previous_update
        P = coupling_matrix_projection(P, u.hist)

        objective = compute_objective(u, v, C, P, rho, mu, alpha)

        if np.abs(objective - former_objective) / former_objective < 1e-5:
            break

    assert (np.allclose(P.dot(np.ones(m)), u.hist))  # assert

    transport_map = u.Dh @ P @ Y

    return transport_map


def post_processing(u, transport_map, sigma):
    """
    allows to recover geometrical information from the source image
    """

    assert(u.is_quantized)

    w = np.zeros_like(u.array)  # synthesis result

    for i in range(u.array.shape[0]):
        for j in range(u.array.shape[1]):

            point_feature = np.hstack([np.array([i / u.array.shape[0], j / u.array.shape[1]]), u.array[i, j]])

            # similarity of pixel (i, j) in the source image with each cluster
            clusters_weight = np.exp(- np.linalg.norm(u.features - point_feature[np.newaxis, :], axis=1)
                                     / (2 * sigma ** 2))

            w[i, j] = np.sum(clusters_weight[:, np.newaxis] * transport_map, axis=0) / np.sum(clusters_weight)

    return w


def color_transfer(u, v, num_iter, tau, rho, mu, alpha, beta, num_neighbors, num_segments, sigma):

    if not u.is_quantized:
        u.quantize(num_segments)

    if not v.is_quantized:
        v.quantize(num_segments)

    transport_map = compute_transport_map(u, v, num_iter, tau, rho, mu, alpha, beta, num_neighbors)

    return post_processing(u, transport_map, sigma)


