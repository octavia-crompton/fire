"""
 Run the RCSR model for the parameter file located in `name`

"""

name = "sims_random_check"

import sys
import os
from filepaths import *
from fire_utility import *

sim_dir = project_dir + "/" + name
sys.path.append(sim_dir)

if "params" in sys.modules:
    del sys.modules["params"]

from params import all_params


def main(argv):
	"""
	"""
	file_dir = sim_dir + "/all_sims"
	if os.path.isdir(file_dir)  == False:
	    os.mkdir(file_dir)

	
	all_sims = run_all_sims(all_params, file_dir)
	

if __name__ == '__main__':
    main(sys.argv)
