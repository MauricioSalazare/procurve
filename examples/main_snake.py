from procurve.principal_curve import PrincipalCurve
from procurve.utils import create_dataset
from procurve.plotting import plot_3d, segments
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from figure_utils import set_figure_art
set_figure_art()

#%% Create the dataset
X = create_dataset(source="snake")

#%% Set parameters for the spline
spline_params = {"degree": 4,
                 "low_angle_deg": -40,
                 "high_angle_deg": 180,
                 "radius": 1.0}

#%% Create the principal curve object and fit.
pc = PrincipalCurve()
X, s, f_spline = pc.fit(X, init_fn="curve", param_fun=spline_params)

#%% Use the principal curve function
s_high_res = np.linspace(0, 1, 1000)
f_s = f_spline(s_high_res)

#%% Plot data and see results
fig, ax = plot_3d(X, plot_wireframe=True)
ax.plot(f_s[:,0], f_s[:,1],  f_s[:,2], color="C3", linewidth=0.5, label="Principal curve")
line_collections_fit = segments(pc.last_iteration_log["data_sorted"],
                                pc.last_iteration_log["p_orthogonal"])
lc_fit = Line3DCollection(line_collections_fit, colors="C1", linewidths=0.4, label="Orthogonal projection")
ax.add_collection(lc_fit)
ax.legend(fontsize="small")
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_zlim(-1.1, 1.1)
fig.savefig("plots\snake.png", dpi=1200)