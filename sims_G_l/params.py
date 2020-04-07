import numpy as np

all_params = {
      "batch_dict" :    {               
        "beta" : [0.5],
        "alpha" :  [0.02, 0.04],
        "ignition_type" : ["G_l", "random"]
        },
      "sim_dict" : {        
        "RI" : np.arange(10, 100, 10),
        "severity" : np.arange(0.1, 1, 0.1)        
        },
      "common_dict" : {                      
            "k_u" : 20.0,
            "k_l" : 5.0,    
            "r_u" : 0.25,
            "r_l" : 1.5,                    
            "S" : 0.5,
            "dt" : 0.01,
            "dt_p" : 0.1,            
            "seed" : 0,
            "ti" : 3000,        
            "tmax" : 1000,                  
            "chi" : 1,                           
            "severity_type" : "random",
            "std_severity" : 0.1,
            "r" : 0.5, 
            "a" : 0.01,
            "b" : 0.99
             }
    }


      
      
