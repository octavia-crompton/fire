
# Contents

- `fire_analytic`
  - math writeup



### Notebooks in `fire_notebooks`

`analytic-dS` :
- Assess sensitivity to varying soil moisture, growth-limiting factor, etc.

- `Dimensionless`

`Estimate_parameters` :
- A notebook for estimating parameters for ICB and SLC


`G_l_ignition_feedback` :
- Explore potential feedbacks between G_l and ignition probability.


`Parallel.ipynb`
- Illustrating how to run the RCSR model in parallel

`RCSR_analytic.ipynb` :
- Tests the analytical solution with constant RI and severity (no randomness)


`RCSR_analytic_random.ipynb`
-  Test the analytical solution over a  grid of parameters, with random ignition and severity.
 - Test whether _G\_u_ and _G\_l_ will eventually converge to the analytical approximate solutions.

`RCSR_illustrate`:
-  Illustrative notebook showing a single parameter case.


- `RCSR_truncated_gaussian` :
   - Check that calculations of the truncated Gaussian moments are correct.


- `Stability`
  - Visualizing analytic predictions about canopy stability



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



###  Next steps and issues

-  dimensionless groups

-  things are converging with an error
  -  see `Stability : Things are slow to converge near the stability boundary`

-  `G_postfire(self)` does not need RI or severity.
      - Is this function defined in `fire_analytic.py`?

- how do results depend on the degree of correlation between layers?
