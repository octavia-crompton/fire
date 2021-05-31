import numpy as np

all_params = {
      "batch_dict" :    {               
          "best" : [0]
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
            "tmax" : 600,
            "t_switch" : 50,
            "ignition_type" : "random",                              
            "severity_type" : "sample",
            "s_scenario" : "none",
            "ri_scenario" : "RI30",
            "soil_scenario" : "none"
             }
    }


      
      