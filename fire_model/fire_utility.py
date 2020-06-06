# TODO: figure out this silly a,b situation
# TODO: 

from fire_model import *
import pickle
import os

"""
To run simulations from a param file, `batch_fire.py`
"""

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

    import itertools as it
    batch_combos = [dict(zip(batch_vars, prod))  for prod in \
              it.product(*(batch_dict[var_name] for var_name in batch_vars))]
    sim_combos = [dict(zip(sim_vars, prod)) for prod in \
               it.product(*(sim_dict[var_name] for var_name in sim_vars))] 
    return common_dict, batch_combos, sim_combos


def run_RCSR(update):
    """
    Run a single RCSR instance
    """     
    params = default_params()
    params.update(update)
    
    p = RCSR(params)

    p.run()

    return p


from multiprocessing import Pool

def run_all_sims(all_params, file_dir):
    """
    Run the RCSR model for all the parameter sets specified in `all_params`,
    and save to `file_dir`
    """
    
    common_dict, batch_combos, sim_combos = interp_params(all_params)
    
    all_sims = []
    for bdict in batch_combos: 
        
        batch_name = ','.join(['-'.join([key, str(myround(bdict[key], 2)) ] ) 
            for key in bdict.keys()])  

        param_list = []
        for sdict in sim_combos:  

            params = common_dict.copy()
            params.update(bdict)      
            sim_name = ','.join(['-'.join([key, str(myround(sdict[key],2))]) 
                                 for key in sdict.keys()])

            params.update(sdict)
            params["batch_name"] = batch_name
            params["sim_name"] = sim_name
            params["key"] = ",".join([batch_name, sim_name]) 
        
            param_list.append(params)


        pool = Pool(processes=8)              
        result =  (pool.map(run_RCSR, param_list )  )       
        pool.close()
        for p in result:
            save_object(p, file_dir + '/{0}.pkl'.format(p.key))

        [all_sims.append(p) for p in result]
    
    
    df = pd.DataFrame(all_sims, index = [p.key for p in all_sims ],
            columns=["p"])

    return df

def read_all_sims(file_dir):
    """
    Read all simulations in file_dir
    """
    all_sims = []

    keys = os.listdir(file_dir)
    keys = [key.replace(".pkl", "") for key in keys]

    for key in keys:    

        p = load_object( file_dir + '/{0}.pkl'.format(key))

        all_sims.append(p) 
        
    df =  pd.DataFrame(all_sims, index = keys, columns=["p"]) 
    return df


def myround(x, precision):
    """
    Round floats, ignore strings, etc.
    """
    try:
        return np.round(x, precision)
    except:
        return x
        
def compute_all_errors(all_sims, sim_dir, recomp = True):
    """
    Compute the errors for a list of RCSR instances, with
    regular ignition and severities

    TODO: DOCUMENT THIS SILLINESS
    TODO: ADD TO PYCHARM project
    
    """
    res = pd.DataFrame()

    for key in all_sims.index:

        p = all_sims.loc[key][0]

        var_list = list(default_params().keys())
        param = pd.Series(vars(p))[var_list]

        if recomp == True:
            df, dfl = compute_errors_mean(p)
        else:
            df, dfl = compute_errors(p)
            
        dfl = dfl.append(param)
        res = res.append(dfl, ignore_index = True)
    res.index = all_sims.index
    res.to_pickle(sim_dir + "/analytic_errs.pkl")
    return res    

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
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object( filename):
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
    print ( 'batch vars:')
    for key in all_params['batch_dict'].keys():
        print( ' ', key, all_params['batch_dict'][key])
    print ( 'sim vars:')
    for key in all_params['sim_dict'].keys():
        print (' ', key, all_params['sim_dict'][key] )
"""
For viewing parameter files
""" 

def print_dict(d):
    """
    Prints an input dictionary `d`
    """
    d =  ',  '.join("$%s$ = %s" % item for 
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
    value = { k : difference(a[k], b[k])  for k in set(a) 
        if a[k] != b[k]}
    return value


def test_for_overlap(list1, list2):
    if list(set(list1) & set(list2)):
        print( 'overlapping vars!\n',  list(set(list1) & set(list2)))
    return  

