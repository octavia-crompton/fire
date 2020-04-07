
import sys

sys.path.append("/Users/octavia/Dropbox/fire/src_v1" )

mymodules = ["plot_config", "fire_model", "fire_utility", "filepaths",
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