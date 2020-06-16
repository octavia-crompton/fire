from fire_model import *
import pickle
import os
"""
To run simulations from a param file, `batch_fire.py`
"""

from multiprocessing import Pool


def read_all_sims(file_dir):
    """
    Read all simulations in file_dir
    """
    all_sims = []

    keys = os.listdir(file_dir)
    keys = [key.replace(".pkl", "") for key in keys]

    for key in keys:
        p = load_object(file_dir + '/{0}.pkl'.format(key))

        all_sims.append(p)

    df = pd.DataFrame(all_sims, index=keys, columns=["p"])
    return df


def myround(x, precision):
    """
    Round floats, ignore strings, etc.
    """
    try:
        return np.round(x, precision)
    except TypeError:
        return x


def compute_all_errors(all_sims, sim_dir, recomp=True):
    """
    Compute the errors for a list of RCSR instances, with
    regular ignition and severities
    
    """
    res = pd.DataFrame()

    for key in all_sims.index:

        p = all_sims.loc[key].p

        var_list = list(default_params().keys())
        var_list = list(set(var_list).intersection( set(vars(p))))
        param = pd.Series(vars(p))[var_list]

        if recomp:
            df, dfl = compute_errors_mean(p)
        else:
            df, dfl = compute_errors(p)

        dfl = dfl.append(param)
        res = res.append(dfl, ignore_index=True)

    res.index = all_sims.index
    res.to_pickle(sim_dir + "/analytic_errs.pkl")
    return res


"""
 Saving and loading fire sims
"""


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


def print_all_params(all_params):
    print('batch vars:')
    for key in all_params['batch_dict'].keys():
        print(' ', key, all_params['batch_dict'][key])
    print('sim vars:')
    for key in all_params['sim_dict'].keys():
        print(' ', key, all_params['sim_dict'][key])


"""
For viewing parameter files
"""


def print_dict(d):
    """
    Prints an input dictionary `d`
    """
    d = ',  '.join("$%s$ = %s" % item for
                   item in d.items())
    return d


def print_param(param):
    """
    Prints class attributes to look like a dictionary
    """
    return ','.join("%s-%s" % item for item in param.items())


def dictionary_diff(a, b):
    """
    Compares two dictionaries
    Must have the same entries
    """
    value = {k: difference(a[k], b[k]) for k in set(a)
             if a[k] != b[k]}
    return value


def test_for_overlap(list1, list2):
    if list(set(list1) & set(list2)):
        print('overlapping vars!\n', list(set(list1) & set(list2)))
    return
