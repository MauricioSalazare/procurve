from src.procurve.principal_curve import PrincipalCurve
from src.procurve.utils import create_dataset
from src.procurve.plotting import plot_2d, segments
from matplotlib import collections as mc
import numpy as np

x_data = create_dataset(source="hastie")
spline_params = {"degree": 5}
pc = PrincipalCurve()
X = x_data.copy()
X, s, f_spline = pc.fit(X, init_fn="pca", param_fun=spline_params)
s_high_res = np.linspace(0, 1, 1000)
f_s = f_spline(s_high_res)

#%% Plot data
ax = plot_2d(x_data)
ax.plot(f_s[:,0], f_s[:,1], color="C3", linewidth=0.5, label="Principal curve")
line_collections_fit = segments(pc.last_iteration_log["data_sorted"],
                                pc.last_iteration_log["p_orthogonal"])
lc_fit = mc.LineCollection(line_collections_fit, colors="C1", linewidths=0.4, label="Orthogonal projection")
ax.add_collection(lc_fit)
ax.legend(fontsize="small")



