
import sys


mymodules = ['plot_config', "fire_model", "fire_utility", "filepaths",
             "fire_plot", "fire_analytic" ]

for mod in mymodules:
    if mod in sys.modules:
        del sys.modules[mod]


from plot_config import *
from fire_model import *
from fire_analytic import *
from fire_utility import *
from fire_plot import *
from filepaths import *


p = RCSR()
update = {
          "r_l" : 0.45,
          "r_u" : 0.15,           
          "alpha" : 0.02,
          "beta" : 0.5,
          "ti" : 1000, 
          "tmax" : 1000,
          "RI" : 20,
          "severity" : 0.80,
          "dt" : 0.01,
          "dt_p" : 0.1,
          "severity_type" : "random",
          "ignition_type" : "G_l",          
          "sigma_phi" : 0.01,
          "seed" : 0,
          "chi" : 1,
          "S" : 0.5
         }


p = RCSR(update)
p.run()
