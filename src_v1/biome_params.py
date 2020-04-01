conifer :  {
    "alpha" : 0.05,                 
    "k_u" : 20.6, 
    "k_l" : 1.2,
    "r_u" : 0.25,
    "r_l" : 1.5,               
    "beta" : 0.5,
    "S" = 0.2 ### ?
    }
meadow :  {
    "alpha" : 0.05,        
    "k_l" : 3.14,
    "r_l" : 0.1,                   
    "beta" : 0.5
    }

grass :  {
    "alpha" : 0.05,
    "k_l" : 3.14,
    "r_l" : 0.1,               
    "beta" : 0.5
    }

shrub :  {
    "alpha" : 0.05,
    "k_l" : 3.14,
    "r_l" : 0.1,               
} 

## Actually, just want three parameter dictionaries :
## conifer/meadow, conifer/grass, conifer/shrub.
## need: soil moisture, growth and carrying capacities, and 
## betas.