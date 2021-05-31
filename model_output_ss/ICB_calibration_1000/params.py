import numpy as np

all_params = {
      "batch_dict" :    {               
        "RI" : [20, 22, 24, 26, 28, 30, 32, 34],
        "alpha" : [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035],
        "beta" : [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] 
                },                
      "sim_dict" :
          {"init" : "ICB"},
      "common_dict" : {            
            "k_u" : 30.0,
            "k_l" : 6.0,
            "r_u" : 0.15,
            "r_l" : 1.5,
            "dt" : 0.1,
            "dt_p" : 0.1,            
            "seed" : "count",            
            "tmax" : 1200,
            "t_switch" : 1000,
            "ignition_type" : "random",                              
            "severity_type" : "sample",
            "ri_scenario" : "suppression",            
            "s_scenario" : "no_change",
            "soil_scenario"  : "none"         
             }
    }


      
      
