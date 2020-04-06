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
    Predict the max severity and min frequency to sustain biomass in 
    the upper canopy
    """

    print ("The minimum return interval with severity = {0:.3f} is {1:.2f} years".format(p.severity,p.min_RI_u()))
    print ("The maximum severity with RI = {0} years is {1:.4f}".format(p.RI, p.max_severity_u()))    



def dictionary_diff(a, b):
    """
    Compares two dictionaries
    Must have the same entries
    """
    value = { k : difference(a[k], b[k])  for k in set(a) 
        if a[k] != b[k]}
    return value
