from procurve.principal_curve import PrincipalCurve
from procurve.utils import create_dataset
from procurve.plotting import plot_3d, segments
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib as mpl
mpl.use("Qt5Agg")
import numpy as np
from figure_utils import set_figure_art
set_figure_art()


#%%
x_data = create_dataset(source="helix")
spline_params = {"degree": 5}
pc = PrincipalCurve()
X = x_data.copy()
X, s, f_spline = pc.fit(X, init_fn="pca", param_fun=spline_params)
s_high_res = np.linspace(0, 1, 1000)
f_s = f_spline(s_high_res)

#%% Plot data
fig, ax = plot_3d(X, plot_wireframe=False, figsize=(5, 5))
ax.plot(f_s[:,0], f_s[:,1],  f_s[:,2], color="C3", linewidth=0.5, label="Principal curve")
line_collections_fit = segments(pc.last_iteration_log["data_sorted"],
                                pc.last_iteration_log["p_orthogonal"])
lc_fit = Line3DCollection(line_collections_fit, colors="C1", linewidths=0.4, label="Orthogonal projection")
ax.add_collection(lc_fit)
ax.legend(fontsize="small")
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_zlim(0.0, 6.8)
ax.set_box_aspect((5/6.8, 5/6.8, 1))
