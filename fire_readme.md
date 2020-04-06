`all_params()` returns the default parameters, which are overwritten when the RCSR class is initialized.
The only was to change the RCSR parameters is to reinitialize!  




###  Next steps

- Sort out : {Lower canopy equilibrium}
- Need to be able to call fire from the command line



- write a script to call multiple runs.  call it:
   - `run_multiple`
   -  implement in parallel
   -  explore sensitivities
   - "call_fire_model" : takes param file

-  dimensionless groups

### Fire ignition and severity

- Fire ignition is predicted on annual timesteps, and
biomass growth on smaller timesteps.  
  - The sensitivity to this choice is small, may be worth rechecking.


- Fire severity
   -  the fraction of biomass removed from the fire.
   - Upper and lower fire severities are drawn from bivariate normal distribution
   - how do things depend on the degree of correlation between layers?


## Issues:
Stability : Things are slow to converge near the stability boundary
 -  things are converging with an error

-  `G_postfire(self)` does not need RI or severity.
 Is this function defined in `fire_analytic.py`?
