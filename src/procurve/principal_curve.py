import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy import optimize
import warnings

class SplineEstimator:
    """
    Group splines in a single estimator.

    Creates the function f(s) = [f_1(s), ..., f_j(s)].T for j = 1, ..., dim.

    Each of the functions f_j() is a uni-variate spline.

    This class is mandatory to create a whole copy of the function (you can not do this with lambda functions).
    https://stackoverflow.com/questions/10802002/why-deepcopy-doesnt-create-new-references-to-lambda-function
    """
    def __init__(self, *, s, x, k=2, ext=0):
        """
        Parameters:
        ----------
            s: sorted values (a.k.a. lambdas by Hastie's paper).
            x: data to fit (dimension is n_samples (rows) x dimensions (columns))
            k: Degree of the spline. Default = 2.
            ext: Extra parameter for the spline function of scipy. Default = 0.
        """

        assert np.ndim(s) == 1, "Parameters for spline should be 1-D."

        if np.ndim(x) == 1:
            x_fit = np.atleast_2d(x).T
        else:
            x_fit = x

        dim = x.shape[1]

        self._f = [UnivariateSpline(s, x_fit[:, ii], k=k, ext=ext) for ii in range(dim)]

    def __call__(self, s):
        """Evaluates the multidimensional spline in the value "s" """

        return np.array([f_j(s) for f_j in self._f]).T


class PrincipalCurve:
    def __init__(self):
        # Placeholders
        self.max_iterations = None
        self.tolerance = None
        self.init_fn = None
        self.f0 = None  # First principal curve function
        self.f = None  # Principal curve function from last iteration
        self.log = None  # Logs of the iterative procedures
        self.last_iteration_log = None  # Logs of the last iteration

        # Default spline parameters
        self.param = {"degree": 5,
                      "low_angle_deg": -40.0,
                      "high_angle_deg": 180.0,
                      "radius": 1.0}

    def fit(self, X: np.ndarray, model="spline", init_fn: str = "pca", f0: SplineEstimator = None,
            iterations: int = 50, tol: float=1e-3, param_fun:dict=None):

        self.max_iterations = iterations
        self.tolerance = tol
        self.init_fn = init_fn

        assert isinstance(X, np.ndarray), "X must be a np.ndarray."
        assert X.ndim == 2, "Dataset must have 2D"

        if param_fun is not None:
            self.param.update(param_fun)

        degree = self.param["degree"]  # Polynomial degree of the spline
        low_angle_deg = self.param["low_angle_deg"]
        high_angle_deg = self.param["high_angle_deg"]
        radius = self.param["radius"]

        dim = X.shape[1]

        if model != "spline":
            print("Only spline function supported right now.")
            raise NotImplementedError

        if init_fn == "pca":
            # Compute PCA:
            U, S, Vt = np.linalg.svd(X)  #
            a1 = Vt[0, :][:, np.newaxis]  # First component of the lower dimensional representation of the data
            a1_space = np.kron([-1, 1], a1).T  # 2 dots in the line connected together, to plot the space
            lambda_ = X @ a1  # scores (lambda values) // Equivalent: lambda_ = U[:,0] * S[0]  // Not normalized
            z1_vectors = np.kron(lambda_.ravel(), a1).T  # rows (x,y,z) positions, equivalent of f(lambda_)

            # (0) Initialization with PCA (Iteration 0)
            s = (X @ a1).flatten()  # initial lambdas
            f = lambda s_i: np.kron(s_i, a1).T
            p = f(s)  # rows (x, y) positions from the smoother
            _, X, p = self._sort_wrt_s(s, X, p)  # Sorting to order the data and normalize to unit speed.
            s = self._unit_speed_transform(p)
            z1_vectors = p.copy()

            self.f0 = f

        elif init_fn == "curve":
            # (0) Initialization with Standard function (Iteration 0)
            low_angle = np.deg2rad(low_angle_deg)
            high_angle = np.deg2rad(high_angle_deg)

            theta = np.linspace(low_angle, high_angle, X.shape[0])
            r = radius

            s = np.linspace(low_angle, high_angle, X.shape[0])
            f = lambda s_i: np.array([r * np.cos(s_i), r * np.sin(s_i), np.zeros(len(s_i))]).T
            p = f(s)  # rows (x, y) positions from the smoother

            # ========================================================================================================
            # Note: The next is not needed in PCA because the "s" is implicitly ordered by the projection of the data
            s_ordered = []
            p_ordered = []
            for x_ii in X:
                # Closest lambda value to the sample "x_ii". i.e., d_ik = min_s {||x_ii - f(s)||}
                d_ik = np.linalg.norm(x_ii - p, axis=1)
                s_ordered.append(s[np.argmin(d_ik)])  # If more than one values is the min. It takes the first one.
                p_ordered.append(p[np.argmin(d_ik)])
            s_ordered = np.array(s_ordered)
            p_ordered = np.array(p_ordered)
            # ========================================================================================================

            _, X, p = self._sort_wrt_s(s_ordered, X, p_ordered)  # Sorting to order the data and normalize to unit speed.
            s = self._unit_speed_transform(p)
            z1_vectors = p.copy()

            self.f0 = f
        else:
            print("'pca' and 'curve' are the only initialization optinons available")
            raise NotImplementedError

        # %%
        iteration_logging = {"data_set": X,
                             "tolerance": self.tolerance,
                             "max_iter": self.max_iterations,
                             "dimension": dim,
                             "iteration": {0: {"data_non_sorted": X,
                                               "data_sorted": X,
                                               # "pc_space": a1_space,
                                               "pc_vectors": z1_vectors}}}

        iteration = 1
        dist_old = np.inf
        converge = False
        iteration_data = None # Place holder for the logging per iteration
        # %% Iterations start
        while iteration < self.max_iterations:
            # (1) Fit spline.
            f = SplineEstimator(s=s, x=X, k=degree)

            p_new = f(s)
            s_high_res = np.linspace(0, 1, 1000)
            p_high_res = f(s_high_res)  # For plotting and fine tune the parameter "s"

            # (2) Use projection operator and find the new lambdas (orthogonal to the spline)
            # (2.a.) Projection operator:
            s_new = []
            dist = 0
            for x_ii in X:
                # Closest lambda value to the sample "x_ii". i.e., d_ik = min_s {||x_ii - f(s)||}
                d_ik = np.linalg.norm(x_ii - p_high_res, axis=1)
                s_0 = s_high_res[np.argmin(d_ik)]  # If more than one values is the min. It takes the first one.

                # Fine tune the lambda to the closest point in the spline (orthogonal vector)
                # i.e., s = min_s {s: ||x_ii - f(s)|| = inf_t ||x_ii - f(t)||}
                result = optimize.minimize(self.objective_func,
                                           x0=s_0,
                                           method='L-BFGS-B',
                                           # bounds=[tuple(extend_range(s, f=0.5))],
                                           bounds=[(0, 1.0)],
                                           args=(x_ii,
                                                 f),
                                           options={'disp': False})
                # assert result.success

                if not result.success:
                    warnings.warn("Optimization did not converge. Using initial value as solution.")
                    dist += 0.0
                    s_new.append(s_0)

                else:
                    # print(f"Obj: {euc_dist(result.x, x_ii, f).round(4)}, Opt: {result.fun.round(4)}")
                    dist += result.fun
                    s_new.append(result.x[0])

            s_new = np.array(s_new)
            print(f"iteration: {iteration}, distance: {dist}, delta: {np.abs(dist_old - dist).round(3)}")

            if np.abs(dist_old - dist) < self.tolerance:
                converge = True
                break

            dist_old = dist

            # (2.b.) Transform the parameters so f() is at unit speed.
            s_new_sorted, x_sorted, _ = self._sort_wrt_s(s_new, X, p_new)
            p_orthogonal = f(s_new_sorted)
            s_new_sorted_normalized = self._unit_speed_transform(p_orthogonal)

            iteration_data = {iteration: {"function": f,
                                          "data_non_sorted": X,
                                          "data_sorted": x_sorted,
                                          "s_sorted": s_new_sorted_normalized,
                                          "p_new": p_new,  # Projections of the data in the cuve
                                          "p_orthogonal": p_orthogonal,
                                          # Projections of the data in the curve with the sorted values of s
                                          "avg_distance": dist,
                                          "error": np.abs(dist_old - dist)}}

            iteration_logging["iteration"].update(iteration_data)

            X = x_sorted.copy()
            s = s_new_sorted_normalized.copy()
            p = p_orthogonal.copy()
            iteration += 1

        if iteration == self.max_iterations and not converge:
            print("Failed.")
        else:
            print("Success!")

        self.f = f  # Save last principal curve function
        self.log = iteration_logging
        self.last_iteration_log = iteration_data[iteration - 1]

        return X, s, f


    def _sort_wrt_s(self, s, x, p):
        # Parameters:
        # -----------
        #   x: np.ndarray: Real data
        #   p: np.ndarray: Values of the spline in the same coordinates of the real data
        #   s: or lambda: np.ndarray: The lambda or "s" real value [0,1] that corresponds to each "p" in the dataset.
        #
        #
        #
        # concatenate the lambda vector (1-D), with a matrix x with shape (n_samples, dim)
        # The matrix is sorted according to the "lamda" values. Where "lam" is the input parameter of
        # the spline. i.e., f(t) == f(lam)

        assert np.ndim(s) == 1
        assert np.ndim(x) == np.ndim(p)
        assert len(s) == x.shape[0]
        assert x.shape[0] == p.shape[0]
        assert x.shape[1] == p.shape[1]

        dim = x.shape[1]

        spline_data = np.hstack([np.atleast_2d(s).T, x, p])
        spline_data = spline_data[spline_data[:, 0].argsort()]  # Sort with respect to lambda

        s_sorted = spline_data[:, 0]
        x_sorted = spline_data[:, 1:(dim + 1)]
        p_sorted = spline_data[:, (dim + 1):]

        return s_sorted, x_sorted, p_sorted

    @staticmethod
    def _euc_dist(lambda_, x_point, spline_function):
        # x1_pred = spline_1(lambda_[0])
        # x2_pred = spline_2(lambda_[0])
        prediction = spline_function(lambda_)

        return np.sum((x_point - prediction) ** 2)

    @staticmethod
    def objective_func(lam_x, *params):
        """ Wrapper of the objective function for the optimization """
        return PrincipalCurve._euc_dist(lam_x, x_point=params[0], spline_function=params[1])

    def _unit_speed_transform(self, p: np.ndarray):
        # Parametrize "s" to unit speed.
        #
        # Parameters:
        # ----------
        #       p: np.ndarray: Array with shape (n_samples, n_dim), which are the values of a
        #               smoother function. i.e., p = f(s),
        #               - f(): Smoother e.g. spline.
        #               - s: Parameter of the curve. e.g. called lambda in Hastie's paper

        arc_segment_lengths = np.linalg.norm(p[1:] - p[0:-1], axis=1)
        s = np.zeros(p.shape[0])
        s[1:] = np.cumsum(arc_segment_lengths)
        s = s / np.sum(arc_segment_lengths)

        return s

    def _svd_flip(self, u, v, u_based_decision=True):
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


