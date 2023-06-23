import numpy as np
from scipy.stats import norm
from scipy.optimize import leastsq

def scaler(x, axis=0):
    """Generic function whose default method centers and/or scales the columns of a numeric matrix. (Same function as R)"""
    mean = np.nanmean(x, axis=axis)
    std = np.nanstd(x, axis=axis)

    if axis==1:
        return (x - mean[:, np.newaxis])/std[:, np.newaxis]
    else:
        return (x - mean) / std


def svd_flip(u, v, u_based_decision =True):
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    Parameters
    ----------
    u : ndarray
        u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.

    v : ndarray
        u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
        The input v should really be called vt to be consistent with scipy's
        ouput.

    u_based_decision : bool, default=True
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.


    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.

    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]

    return u, v


def fitfunc_constant_radius(p, coords, r=1):
    x0, y0, z0 = p
    x, y, z = coords.T

    return np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2) - r


def fit_sphere_constant_radius(Z: np.ndarray, r=1):
    coords_pca = Z[:, :3]
    p0 = [0, 0, 0]
    errfunc = lambda p, x: fitfunc_constant_radius(p, x, r=r)
    p1, flag, = leastsq(errfunc, p0, args=(coords_pca,))

    assert flag, "The optimization did not converge."

    loss = errfunc(p1, coords_pca).sum()

    return *p1, r, loss

def create_dataset(source="hastie"):

    if source == "hastie":
        np.random.seed(1234)
        n_points = 200
        lambda_ = np.random.uniform(0, 2 * np.pi, n_points)
        error1 = np.random.normal(loc=0, scale=0.5, size=(n_points))
        error2 = np.random.normal(loc=0, scale=0.5, size=(n_points))

        x1 = 5 * np.sin(lambda_) + error1
        x2 = 5 * np.cos(lambda_) + error2

        t = np.linspace(0, 2 * np.pi, n_points)
        x1_clean = 5 * np.sin(t)
        x2_clean = 5 * np.cos(t)

        x_data = np.vstack([x1, x2]).T.copy()
        x_data = scaler(x_data)

    elif source == "parabola":
        # Data from a parabola
        np.random.seed(1234)
        n_points = 200

        error1 = np.random.normal(loc=0, scale=0.1, size=(n_points))
        error2 = np.random.normal(loc=0, scale=0.2, size=(n_points))

        x1 = np.linspace(-2.0, 2.0, n_points) + error1
        x2 = x1 ** 2 + error2

        # x1_clean = np.linspace(-2.0, 2.0, n_points)
        # x2_clean = x1_clean ** 2

        x_data = np.vstack([x1,x2]).T.copy()
        x_data = x_data - np.mean(x_data, 0)

    elif source == "snake":
        np.random.seed(1234)
        n_points = 20
        n_samples = 100

        rv = norm()
        x = np.linspace(norm.ppf(0.001, loc=1), norm.ppf(0.999, loc=1), n_points)

        gradients = np.linspace(1, -1, n_samples)
        norm_shifts = np.linspace(-4, 4, n_samples)

        f1 = norm(loc=1).pdf(x)[np.newaxis, :]
        # f2 = lambda u: -norm(loc=u, scale=0.5).pdf(x)[np.newaxis, :] * 0.5
        f2 = lambda u: -norm(loc=u, scale=1).pdf(x)[np.newaxis,:] * 1

        X = gradients[0] * f1 + (1 - gradients[0]) * f2(norm_shifts[0])
        for gradient, norm_shift in zip(gradients[1:], norm_shifts[1:]):
            X = np.vstack([X, gradient * f1 + (1 - gradient) * f2(norm_shift)])

        # Random noise
        X = X + np.random.normal(0, 1, size=X.shape) * 0.03

        # Shuffle the indexes of matrix X
        np.random.seed(1234)
        indexes = np.arange(X.shape[0])
        np.random.shuffle(indexes)

        # Standardize
        X_stded = scaler(X, axis=1)
        X_stded = X_stded / np.sqrt(X_stded.shape[1])
        X_centered = X_stded - X_stded.mean(axis=0)

        U, S_, Vt = np.linalg.svd(X_centered, full_matrices=False)  # Single values are sorted
        U, Vt = svd_flip(U, Vt, u_based_decision=True)

        n_components_ = 3
        # Centered, because SVD was calculated over centered data
        z_svd = U[:, :n_components_] @ np.diag(S_[:n_components_])

        x0, y0, z0, r0, loss = fit_sphere_constant_radius(z_svd, r=1)
        z_svd = z_svd[:, :3] - np.array([x0, y0, z0])
        x_data = z_svd.copy()

    elif source == "polynomial":
        # #%% Data from R example
        # # https://www.r-bloggers.com/2016/04/principal-curves-example-elements-of-statistical-learning/
        np.random.seed(1234)
        # x1 = np.arange(1,10,0.3)
        x1 = np.linspace(1, 10, 100)
        w = 0.6067
        a0 = 1.6345
        a1 = -.6235
        b1 = -1.3501
        a2 = -1.1622
        b2 = -.9443;
        x2 = a0 + a1 * np.cos(x1*w) + b1 * np.sin(x1*w) + a2 * np.cos(2*x1*w) + \
             b2 * np.sin(2*x1*w) + np.random.normal(0, 3/4, len(x1))

        x = np.vstack([x1, x2]).T
        x_data = scaler(x)

    elif source == "helix":
        # Taken from the Exercise 14.13 (p.582) from [1]
        # [1] Hastie, Trevor, Robert Tibshirani, and Jerome Friedman. 2009. "The Elements of Statistical Learning."
        #     Springer Series in Statistics. New York, NY: Springer New York. https://doi.org/10.1007/978-0-387-84858-7.

        s = np.linspace(0, 2 * np.pi, 200)
        x_1 = np.cos(s) + 0.1 * np.random.randn(200)
        x_2 = np.sin(s) + 0.1 * np.random.randn(200)
        x_3 = s + 0.1 * np.random.randn(200)
        x_data = np.vstack([x_1, x_2, x_3]).T

    else:
        raise ValueError("wrong source selected")

    return x_data