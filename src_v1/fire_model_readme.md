Parameters to initialize the fire model

 - `alpha` : competition
 - `beta` : 1,
 - `chi` : `G_l` ignition feedback parameter
 - `k_u` : upper canopy carrying capacity (kg/m2)
 - `k_l` : lower canopy carrying capacity (kg/m2)
 - `dt` : 0.1,
 - `dt_p` : 0.1,
 - `S` : 0.5,    
-  `ti` : 1000,
 - `tmax` : 3000,
 - `seed` : 0,
 - `RI` : fire return interval (yr)
 - `ignition_type` :  - `series - `,            
 - `severity_type` :  - `fixed - `,
 - `severity` : 0.5,
 - `std_severity` : standard deviation of the sampled severity
   - 
 - `r_u` :  upper canopy growth rate (1/yr)
 - `r_l` : lower canopy growth rate (1/yr)
 - `a` : 0.01,
 - `b` : 0.99


 Parameters used throughout:



     severity : float
         fire severity (fraction of biomass removed by fire)
     r_lp : float
         modified growth rate for the lower canopy  (1/yr)
     k_lp :  float
         modified carrying capacity for the lower canopy (kg/m2)

    alpha : float
         competition term, upper canopy inhibition of the lower canopy.

     beta : float
         soil moisture growth-limiting factor [-]

     G_uo, G_lo : float
         initial upper and lower canopy biomass (kg/m2)

     ti : float
         "Spin-up" time (years)
