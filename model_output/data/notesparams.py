import numpy as np

all_params = {
      "batch_dict" :    {               
        "beta" : [0.5],
        ---> "S" : [0.4, 0.2], 
        "alpha" :  [0.02, 0.06],
        ---> "ignition_type" : ["G_l", "random"]
        },
      "sim_dict" : "ICB",
      "common_dict" : {                      
            "k_u" : 60.0, # use for the initial biomass; # can use a flat dictionary
            "k_l" : 5.0,    
            "r_u" : 0.25,
            "r_l" : 1.5,  
            "dt" : 0.01,
            "dt_p" : 0.1, 
            "seed" : 0,
          --->   "ti" : 5000, 
          --->   "tmax" : 4000,
            "chi" : 1,
            "severity_type" : "random",
            "std_severity" : 0.1,
            "r" : 0.5, 
            "a" : 0.01,
            "b" : 0.99,
            --->  "flags" :[],
            "RI" : 
             }
    }
            "RI" : np.arange(10, 100, 10),
        "severity" : np.arange(0.1, 1, 0.1)   
    ### NEED TO MODIFY CODE SO THAT INITIAL CONDITIONS GU AND GL ARE READ FROM FILE, RATHER THAN A FRACTION OF CARRYING CAPACITY.
    ### I NEED TO PRESCRIBE INITIAL GL IN CONIFER CASE.s


      
      
