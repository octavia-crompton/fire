

### Summary

`default_params()` returns the default parameters, which are overwritten when the RCSR class is initialized.
The only was to change the RCSR parameters is to reinitialize!  




### Parameters to initialize the fire model

 - `alpha` : competition
   - upper canopy inhibition of the lower canopy
 - `beta` : vegetation growth-limiting factor
 - `k_u` : upper canopy carrying capacity (kg/m2)
 - `k_l` : lower canopy carrying capacity (kg/m2)
 - `r_u` :  upper canopy growth rate (1/yr)
 - `r_l` : lower canopy growth rate (1/yr)
 - `S` : relative soil moisture content
    - center around S = 0.5
 - `dt` : timestep
 - `dt_p` : 'print' timestep    
 -  `ti` : 'spin-up' time (years)
 - `tmax` : maximum time (years)
 - `seed` : random seed (int)

#### Ignition

Fire ignition is predicted on annual time steps, and
biomass growth on smaller time steps.


 - `RI` : fire return interval (yr)
 - `ignition_type`: how fire ignition thresholds are specified
   - `series`, `random`, or `G_l`
 - `chi` : `G_l` ignition feedback parameter


#### Severity
- `severity` : mean fire severity
   - fraction of biomass removed by fire
 - `severity_type` :  type of fire severity
     - `fixed` or `random`
     - if `fixed`, lower and upper canopy severity is the same
     - if  `random`, upper and lower fire severities are drawn from a bivariate normal distribution
 - `std_severity` : standard deviation of the sampled severity
    - applies to  `random` severity type
 - `a` , `b` : minimum and maximum severity
   - parameter in truncated gaussian
 - `r` : correlation between upper and lower canopy severity

#### Initialized with defaults

 -   `G_uo` and `G_lo` :  
   - initial upper and lower canopy biomass (kg/m2)
   - Initialized as 1/10th the carrying capacities
