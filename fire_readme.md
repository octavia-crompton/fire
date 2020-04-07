

###  Next steps and issues

-  dimensionless groups

-  things are converging with an error
  -  see `Stability : Things are slow to converge near the stability boundary`

-  `G_postfire(self)` does not need RI or severity.
      - Is this function defined in `fire_analytic.py`?

- how do results depend on the degree of correlation between layers?


### List of notebooks


`analytic-dS` :
- Assess sensitivity to varying soil moisture, growth-limiting factor, etc.


`CRUR_analytic-random`
- Similar test of the analytical solution over a  grid of parameters, but with random ignition and severity.
- Saved to `random_RI_phi.pkl`

`CRUR_truncated_gaussian` :
 - Check that calculations of the truncated Gaussian moments are correct.
 - Test whether _G\_u_ and _G\_l_ will eventually converge to the analytical approximate solutions.


`Estimate_parameters` :
- A notebook for estimating parameters for ICB and SLC


`G_l_ignition_feedback` :
- Explore potential feedbacks between G_l and ignition probability.


`RCSR_analytic` :
- Tests the analytical solution with constant RI and severity (no randomness)


`RCSR_illustrate`:
-  Illustrative notebook showing a single parameter case.


`Stability`
- Still a hodgepodge, including code to parallelize


### List of pkl files:

`result.pkl` : test analytic solution for constant RI and severity


### src_v1 python code

Version 1 of the RCSR model

- `biome_params.py`

- `call_fire.py`

- `filepaths.py`

- `fire_analytic.py`

- `fire_model_readme.md`

- `fire_model.py`

- `fire_plot.py`

- `fire_utility.py`

- `plot_config.py`

- `startup.py`
