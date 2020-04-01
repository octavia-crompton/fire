fire: dimensionless groups; restructure code; questions

### Fire ignition and severity

- Fire ignition is predicted on annual timesteps, and
biomass growth on smaller timesteps.  
  - The sensitivity to this choice is small, may be worth rechecking.


- Fire severity
   -  the fraction of biomass removed from the fire.
   - Upper and lower fire severities are drawn from bivariate normal distribution
   - how do things depend on the degree of correlation between layers?




###  Version control the fire code

- document variable names and units
  - consider all variables that should be saved; those that can be removed.
  - folders:
  - src_v1
  - code_v1


- write a script to call multiple runs.  call it:
   - `run_multiple`
   -  implement in parallel
   -  explore sensitivities
