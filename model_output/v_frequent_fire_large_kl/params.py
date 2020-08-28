import numpy as np

all_params = {
      "batch_dict" :    {               
        "beta" : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7, 0.8, 0.9, 1.0],
        "alpha" : [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006,
                    0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014,
                    0.015,0.016]
                },
      "sim_dict" :
          {"init" : "ICB"},
      "common_dict" : {
            "RI" : 26,
            "k_u" : 60.0,
            "k_l" : 10.0,
            "r_u" : 0.15,
            "r_l" : 1.5,
            "dt" : 0.01,
            "dt_p" : 1,            
            "seed" : "count",
            "ti" : 0,
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


      
      
