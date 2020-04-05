class param_class:
    """    
    """
    def __init__(self, params = {}):
        """
        """
        for k, v in params.items():
             setattr(self, k, v)


# necessary?
# def set_defaults():
#     r_l = 1.5
#     r_u = 0.25
#     k_u = 20
#     k_l = 5
#     RI = 20
#     S = .21
#     alpha = 0.06
#     beta = 0.5
#     severity = 0.5
#     globals().update(locals())



def print_limits(p):
    """
    predict the max severity and min frequency to sustain biomass in 
    the upper canopy
    """
    print ("The minimum return interval with severity = {0} is {1:.2f} years".format(p.severity,p.min_RI_u()))
    print ("The maximum severity with RI = {0} years is {1:.4f}".format(p.RI, p.max_severity_u()))    
