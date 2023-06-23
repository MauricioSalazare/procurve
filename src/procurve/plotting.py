import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .utils import fit_sphere_constant_radius

def segments(x_data, z_data):
    # Collect the line segments between the data and its projections.
    # This is to replicate the projection line from R:
    # https://www.r-bloggers.com/2016/04/principal-curves-example-elements-of-statistical-learning/
    # Code of line collection for matplotlib taken from:
    # https://stackoverflow.com/questions/21352580/matplotlib-plotting-numerous-disconnected-line-segments-with-different-colors
    #
    # Arguments:
    # ----------
    #   x_data: np.array: Real data (n_samples, dimension)
    #   z_data: np.array: Projected data (n_samples, dimension)
    #
    #     Notes: It is assumed that each row of x_data and z_data forms the line segment
    #
    # Returns:
    #   line_collections: np.array: Collection of lines that connects the real and projected data
    #       the dimension of the array is (n_samples, dimension, dimension)
    #       e.g.,
    #       Matrix for each line must be :
    #               [[x_1, x_2],  -> Real point coordinates (x_1, x_2)
    #                [z_1, z_2]]  -> Projected point coordinates (z_1, z_2)
    #       All small matrices of 2 x 2 are collected together in a 3D matrix called line_collections:
    #       So, e.g. for sample 10,        x[10,:] == line_collections[10, 0,:]
    #                             z1_vectors[10,:] == line_collections[10, 1,:]

    stacked = np.dstack([x_data, z_data])
    line_collections = np.array([stacked[i].T for i in range(len(stacked))])  # (n_samples, d, d)

    return line_collections

def create_sphere_data(r=1.0, p1=None):
    """Helper function to create the meshgrid for the wireframe plot of the sphere"""
    phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0 * np.pi:100j]
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    if p1 is not None:
        x += p1[0]
        y += p1[1]
        z += p1[2]

    return x, y, z

def plot_3d(X, plot_wireframe:bool=False, figsize:tuple=(5,5), ax=None):
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=8, color="C0", label="Data points")

    if plot_wireframe:
        x, y, z = create_sphere_data(r=1.0)
        ax.plot_wireframe(x, y, z, rstride=5, cstride=5, linewidth=0.2, color="grey")

    ax.set_xlim(-1.01, 1.01)
    ax.set_ylim(-1.01, 1.01)
    ax.set_zlim(-1.01, 1.01)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    return fig, ax


def plot_2d(X, figsize:tuple=(5,5), ax=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(X[:, 0], X[:, 1], s=2, color="C0", label="Data points")
    ax.set_box_aspect(1.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return fig, ax



