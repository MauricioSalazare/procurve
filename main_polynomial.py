from src.procurve.principal_curve import PrincipalCurve
from src.procurve.utils import create_dataset
from src.procurve.plotting import plot_2d, segments
import numpy as np
from matplotlib import collections as mc

X = create_dataset(source="polynomial")
spline_params = {"degree": 3}
pc = PrincipalCurve()
X, s, f_spline = pc.fit(X, init_fn="pca", tol=1e-8, param_fun=spline_params)
s_high_res = np.linspace(0, 1, 1000)
f_s = f_spline(s_high_res)

#%% Plot data
ax = plot_2d(X)
ax.plot(f_s[:,0], f_s[:,1], color="C3", linewidth=0.5, label="Principal curve")
line_collections_fit = segments(pc.last_iteration_log["data_sorted"],
                                pc.last_iteration_log["p_orthogonal"])
lc_fit = mc.LineCollection(line_collections_fit, colors="C1", linewidths=0.4, label="Orthogonal projection")
ax.add_collection(lc_fit)
ax.legend(fontsize="small")

