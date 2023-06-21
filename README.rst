.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/MauricioSalazare/multi-copula/master?urlpath=lab/tree/examples
   :alt: binder



ProCurve
===============


What is ProCurve?
------------------------

It is a robust principal curve package focused on fitting data that lies in a sphere.
Splines are the estimators used for the principal curves.

How to install
--------------
The package can be installed via pip using:

.. code:: shell

    pip install procurve

Example:
--------
Run the load base case as:

.. code-block:: python

    from src.procurve.principal_curve import PrincipalCurve
    from src.procurve.utils import create_dataset
    from src.procurve.plotting import plot_3d, segments
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    X = create_dataset(source="snake")
    spline_params = {"degree": 4,
                     "low_angle_deg": -40,
                     "high_angle_deg": 180,
                     "radius": 1.0}
    pc = PrincipalCurve()
    X, s, f_spline = pc.fit(X, init_fn="curve", param_fun=spline_params)
    s_high_res = np.linspace(0, 1, 1000)
    f_s = f_spline(s_high_res)

    #%% Plot data
    ax = plot_3d(X, plot_wireframe=True)
    ax.plot(f_s[:,0], f_s[:,1],  f_s[:,2], color="C3", linewidth=0.5, label="Principal curve")
    line_collections_fit = segments(pc.last_iteration_log["data_sorted"],
                                    pc.last_iteration_log["p_orthogonal"])
    lc_fit = Line3DCollection(line_collections_fit, colors="C1", linewidths=0.4, label="Orthogonal projection")
    ax.add_collection(lc_fit)
    ax.legend(fontsize="small")

.. image:: https://raw.githubusercontent.com/MauricioSalazare/procurve/master/examples/plots/snake.svg
    :align: center


Reading and citations:
----------------------
..
    _The mathematical formulation of the generative model with the copula can be found at:

The pseudo-algorithm and mathematical formulation can be found `here  <https://https://github.com/MauricioSalazare/multi-copula/tree/master/examples>`_.



How to contact us
-----------------
Any questions, suggestions or collaborations contact Mauricio Salazar at <e.m.salazar.duque@tue.nl>