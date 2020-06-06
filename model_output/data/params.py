import numpy as np

all_params = {
      "batch_dict" :    {               
        "beta" : [0.2],
        "alpha" :  [0.02, 0.04]
        },
      "sim_dict" :
        "ICB",
      "common_dict" : {
            "RI" : 13.0,
            "k_u" : 20.0,
            "k_l" : 5.0,    
            "r_u" : 0.25,
            "r_l" : 1.5,
            "dt" : 0.01,
            "dt_p" : 0.1,            
            "seed" : 0,
            "ti" : 1,
            "tmax" : 100,
            "ignition_type" : "random",                              
            "severity_type" : "sample",
            "severity" : None,
            "std_severity" : None,
            "chi" : 1,
            "r" : 0.5, 
            "a" : 0.01,
            "b" : 0.99
             }
    }


      
      
