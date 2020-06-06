"""
 Run the RCSR model for the parameter file located in `name`

"""
import sys
import os
from multiprocessing import Pool
import pickle
import itertools as it
import pandas as pd
import numpy as np


from filepaths import *
from fire_model import *

model_dir = os.path.dirname(__file__)

name = "data"
output_dir = os.path.join(project_dir, "model_output", name)
sys.path.append(output_dir)

if "params" in sys.modules:
    del sys.modules["params"]

from params import all_params


def run_all_sims():
    """
    Run the RCSR model for all the parameter sets specified in `all_params`,
    and save to `file_dir`	
    """

    file_dir = os.path.join(output_dir, "all_sims")
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)

    if all_params['sim_dict'] == "ICB":
        common_dict, batch_combos, sim_combos = read_ICB_params()
    else:
        common_dict, batch_combos, sim_combos = interp_params(all_params)

    all_sims = []
    sim_num = 0

    for bdict in batch_combos:

        batch_name = ','.join(['-'.join([key, str(myround(bdict[key], 2))])
                               for key in bdict.keys()])

        param_list = []

        for sdict in sim_combos:
            sim_num += 1
            params = common_dict.copy()
            params.update(bdict)
            sim_name = ','.join(['-'.join([key, str(myround(sdict[key], 2))])
                                 for key in sdict.keys()])

            params.update(sdict)
            params["batch_name"] = batch_name
            params["sim_name"] = sim_name
            params["key"] = ",".join([batch_name, sim_name])

            if common_dict["seed"] == "count":
                params["seed"] = int(sim_num)

            param_list.append(params)

        pool = Pool(processes=8)
        result = (pool.map(run_RCSR, param_list))
        pool.close()
        for p in result:
            save_object(p, file_dir + '/{0}.pkl'.format(p.key))

        [all_sims.append(p) for p in result]

    df = pd.DataFrame(all_sims, index=[p.key for p in all_sims],
                      columns=["p"])

    return df

def read_ICB_params():
    """
    Interpret the `all_params` parameter dictionary

    Returns
    -------
    common_dict
    batch_combos
    sim_combos

    """

    batch_dict = all_params['batch_dict']
    common_dict = all_params['common_dict']

    IC = pd.read_csv(os.path.join(model_dir, "IC.csv"))
    IC = np.array(IC)

    batch_vars = sorted(batch_dict)

    sim_vars = ['veg', 'S']
    # sim_dict = {'veg': IC[:, 0], 'S': IC[:, 1]}
    common_vars = sorted(common_dict)

    test_for_overlap(sim_vars, common_vars)
    test_for_overlap(batch_vars, common_vars)
    test_for_overlap(batch_vars, sim_vars)

    batch_combos = [dict(zip(batch_vars, prod)) for prod in \
                    it.product(*(batch_dict[var_name] for var_name in batch_vars))]

    # sim_combos = [dict(zip(sim_vars, prod)) for prod in \
    #               it.product(*(sim_dict[var_name] for var_name in sim_vars))]
    sim_combos = [{'veg' : IC[i, 0], 'S' : np.round(IC[i, 1],3)} for i in range(len(IC))]
    return common_dict, batch_combos, sim_combos


def interp_params(all_params):
    """
    Interpret the `all_params` parameter dictionary 

    Returns
    -------
    common_dict
    batch_combos
    sim_combos

    """

    batch_dict = all_params['batch_dict']
    sim_dict = all_params['sim_dict']
    common_dict = all_params['common_dict']

    batch_vars = sorted(batch_dict)
    sim_vars = sorted(sim_dict)
    common_vars = sorted(common_dict)

    test_for_overlap(sim_vars, common_vars)
    test_for_overlap(batch_vars, common_vars)
    test_for_overlap(batch_vars, sim_vars)

    batch_combos = [dict(zip(batch_vars, prod)) for prod in \
                    it.product(*(batch_dict[var_name] for var_name in batch_vars))]
    sim_combos = [dict(zip(sim_vars, prod)) for prod in \
                  it.product(*(sim_dict[var_name] for var_name in sim_vars))]
    return common_dict, batch_combos, sim_combos


def run_RCSR(update):
    """
    Run a single RCSR instance
    """
    # params = default_params()
    # params.update(update)
    params = update
    p = RCSR(params)

    p.run()

    return p


def myround(x, precision):
    """
    Round floats, ignore strings, etc.
    """
    try:
        return np.round(x, precision)
    except:
        return x


"""
 Saving and loading fire sims
"""


def save_object(obj, filename):
    """
    Save object to pickle file

    Parameters:
    -----------
    obj : oject
        i.e. RCSR instance

    filename: str
        location to save `obj`

    Sample usage:
        save_object(p, 'p.pkl')
    """
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    """
    Parameters:
    -----------
    filename: str
        location of saved `obj`

    Usage:
    -----
        d = load_object('p.pkl')
    """
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj


def test_for_overlap(list1, list2):
    if list(set(list1) & set(list2)):
        print('overlapping vars!\n', list(set(list1) & set(list2)))
    return


if __name__ == '__main__':
    run_all_sims()
