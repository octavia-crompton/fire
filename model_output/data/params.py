import numpy as np

all_params = {
      "batch_dict" :    {               
        "beta" : [0.0, 0.5],
        "alpha" :  [0.02, 0.04]
        },
      "sim_dict" :
          {"init" : "ICB"},
      "common_dict" : {
            "RI" : 27,
            "k_u" : 60.0,
            "k_l" : 7.0,
            "r_u" : 0.15,
            "r_l" : 1.5,
            "dt" : 0.01,
            "dt_p" : 0.1,            
            "seed" : "count",
            "ti" : 1,
            "tmax" : 40,
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


      
      
