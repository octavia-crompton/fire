from fire_model import *
import pickle

class param_class:
    """    
    """
    def __init__(self, params = {}):
        """
        """
        for k, v in params.items():
             setattr(self, k, v)

def print_limits(p):
    """
    Predict the max severity and min frequency to sustain biomass 
    in the upper canopy
    """

    print ("The minimum return interval with severity = {0:.3f} is {1:.2f} years".format(p.severity,p.min_RI_u()))
    print ("The maximum severity with RI = {0} years is {1:.4f}".format(
        p.RI, p.max_severity_u()))    

def print_dict(d):
    # attrs = vars(self)
    d =  ',  '.join("$%s$ = %s" % item for 
        item in d.items()) 
    return d

def print_param(param):
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


## Saving and loading fire sims
def save_object(obj, filename):
    """
    Save object to pickle file

    Parameters:
    -----------
    obj : oject
        i.e. List of RCSR

    Sample usage:
        save_object(p, 'p.pkl')
    """
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object( filename):
    """
    Sample usage:
        d = load_object('p.pkl')
    """
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj



             