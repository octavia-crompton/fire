
### List of notebooks


`CRUR_analytic` :
- Tests the analytical solution with constant RI and severity (no randomness)
- currently over a grid of severities, RIs, alphas and r_ls.
  - the results of this gridded test are saved to `result.pkl`
-  Also tests sensitivity to time step

`CRUR_analytic-random`
- Similar test of the analytical solution over a  grid of parameters, but with random ignition and severity.
- Saved to `random_RI_phi.pkl`

`CRUR_truncated_gaussian` :
 - Check that calculations of the truncated Gaussian moments are correct.
 - Test whether _G\_u_ and _G\_l_ will eventually converge to the analytical approximate solutions.


<!-- #region -->

`analytic-dS` :
- Assess sensitivity to varying soil moisture, growth-limiting factor, etc.


`Estimate_parameters` :
- A notebook for estimating parameters for ICB and SLC 


`G_l_ignition_feedback` : 
- Explore potential feedbacks between G_l and ignition probability.

`RCSR_illustrate`:
-  Illustrative notebook showing a single parameter case.


`Stability`
- Still a hodgepodge, including code to parallelize
<!-- #endregion -->

### List of pkl files:

`result.pkl` : test analytic solution for constant RI and severity
