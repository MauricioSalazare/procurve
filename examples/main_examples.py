import matplotlib.pyplot as plt
from procurve.principal_curve import PrincipalCurve
from procurve.utils import create_dataset
from procurve.plotting import plot_3d, plot_2d, segments
from matplotlib import collections as mc
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("Qt5Agg")
import numpy as np
from figure_utils import set_figure_art
set_figure_art()


model_parameters = [("hastie", {"degree": 5}, "pca"),
                     ("polynomial", {"degree": 3}, "pca"),
                     ("helix", {"degree": 5}, "pca"),
                     ("snake", {"degree": 4,
                                "low_angle_deg": -40,
                                "high_angle_deg": 180,
                                "radius": 1.0}, "curve")]
plotting_params = {"hastie": {"xlim": (-1.7, 1.7),
                              "ylim": (-1.7, 1.7)},
                   "polynomial": {"xlim": (-2.0, 2.0),
                                  "ylim": (-3.0, 3.5)},
                   "helix": {"xlim": (-2.5, 2.5),
                             "ylim": (-2.5, 2.5),
                             "zlim": (-0.0, 6.8),
                             "aspect": (5/6.8, 5/6.8, 1),
                             "wireframe": False},
                   "snake": {"xlim": (-1.1, 1.1),
                             "ylim": (-1.1, 1.1),
                             "zlim": (-1.1, 1.1),
                             "aspect": (1, 1, 1),
                             "wireframe": True},
                   }


# fig, ax = plt.subplots(1, 4, figsize=(15,4))
fig = plt.figure(figsize=(15,4))
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
for ii, (data_name, spline_params, init_fn) in enumerate(model_parameters):
    x_data = create_dataset(source=data_name)
    pc = PrincipalCurve()
    X, s, f_spline = pc.fit(x_data.copy(), init_fn=init_fn, param_fun=spline_params)
    s_high_res = np.linspace(0, 1, 1000)
    f_s = f_spline(s_high_res)

    if data_name in ["helix", "snake"]:
        ax = fig.add_subplot(1, 4, ii + 1, projection='3d')
        plot_3d(x_data, plot_wireframe=plotting_params[data_name]["wireframe"], ax=ax)
        ax.plot(f_s[:, 0], f_s[:, 1], f_s[:, 2],  color="C3", linewidth=0.5, label="Principal curve")
        line_collections_fit = segments(pc.last_iteration_log["data_sorted"],
                                        pc.last_iteration_log["p_orthogonal"])
        lc_fit = Line3DCollection(line_collections_fit, colors="C1", linewidths=0.4, label="Orthogonal projection")
        ax.add_collection(lc_fit)
        ax.set_zlim(plotting_params[data_name]["zlim"])
    else:
        ax = fig.add_subplot(1, 4, ii + 1)
        plot_2d(x_data, ax=ax)
        ax.plot(f_s[:, 0], f_s[:, 1], color="C3", linewidth=0.5, label="Principal curve")
        line_collections_fit = segments(pc.last_iteration_log["data_sorted"],
                                        pc.last_iteration_log["p_orthogonal"])
        lc_fit = mc.LineCollection(line_collections_fit, colors="C1", linewidths=0.4, label="Orthogonal projection")
        ax.add_collection(lc_fit)
    ax.set_xlim(plotting_params[data_name]["xlim"])
    ax.set_ylim(plotting_params[data_name]["ylim"])
    ax.legend(fontsize="small")
    ax.set_title(f"{data_name.capitalize()}")
fig.savefig("plots\samples.png", dpi=1200)
fig.savefig("plots\samples_low_res.png", dpi=600)



