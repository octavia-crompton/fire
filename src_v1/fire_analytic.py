from fire_model import *
"""
Stand alone functions only
"""  

##### Mean biomass, and biomass before/after fires ##### 
def G_prefire( r, k, RI, severity):
    """
    Computes the biomass immediately after each fire,
    once the system is in dynamic equilibrium
    """
    x = r*RI
    phi_R = 1 - severity # fraction of biomass remaining

    numer = phi_R - np.exp(-x)
    denom = 1 - np.exp(-x)
    G_max = k*numer/denom/phi_R

    return greater_than_zero(G_max)


def mean_G( r, k, RI, severity):
    """
    Returns mean(G) for general coefficients
    """
    G_mean = k*(1+np.log(1-severity)/(r*RI))        
    
    return greater_than_zero(G_mean)


def mean_G_l(r_l, r_u, k_l,  k_u, S, beta, alpha, RI, severity):
	"""
	Returns mean(G_l)
	"""
	r_up = r_u*S**beta

	G_u_mean = (k_u*(1+np.log(1-severity)/(r_up*RI)))        
	G_u_mean = greater_than_zero(G_u_mean)

	r_lp = r_l*S**beta-alpha*G_u_mean
	k_lp = k_l*r_lp/(r_l*S**beta)

	G_l_mean = k_lp*(1 + np.log(1- severity)/(r_lp*RI))

	return greater_than_zero(G_l_mean)


def mean_G_u( r_u,  k_u, S, beta, RI, severity):
	"""
	Returns mean(G_u)
	"""
	r_up = r_u*S**beta

	G_u_mean = greater_than_zero(k_u*(1+np.log(1-severity)/(r_up*RI)))
	    
	return greater_than_zero(G_u_mean)



def fix_G_l(r_l, r_u, k_l,  k_u, S, beta, alpha, RI, severity, gamma):
    """
    For lack of a better name, solves for mean(G_l) if we assume 
    mean(G_u) = gamma(k_u)
    """
    numer (alpha*k_u*gamma + (1-gamma)*r_u*S**beta)
    denom = (r_l*S**beta)

    return  1 - numer/denom
               

"""
Stability functions!
"""

def min_RI_u(r_u,S, beta, severity):
    """
    For the system's growth rate and severity, 
    find the minimuim return interval for which G_u>0
    """    
    r_up = r_u*S**beta
    severity = severity

    return -1./r_up*np.log(1-severity)

def max_RI_l(r_l, r_u, k_u, S, beta, alpha, RI, severity):
	"""
	Bounds the maximum return time for which G_l >0
	"""
	a = - np.log(1-severity)/(r_u*S**beta)
	numer = (alpha*k_u - r_u*S**beta)
	denom = (alpha*k_u - r_l*S**beta)
	return a*numer/denom

def min_RI_l(r_l, S, beta, severity):
    """
    Bounds the minimum return time for which G_l >0
    """
    r = r_l*S**beta
    return -1./r*np.log(1-severity)


def max_severity_u( r_u, RI):
    """
    For a given growth rate and return interval, 
    find the maximum severity for which G > 0.
    """
    return 1 - np.exp(-r_u*RI)

"""
Estimates related to G_l in equilibrium with G_max
"""
def max_severity_l(r_l, r_u, k_u, S, beta, alpha, RI):
    """
    Maximum fire severity;
    severity must be less than the returned value 
    to sustain lower canopy biomass

	# BUG? CONFUSED!
    """
    numer = alpha*k_u*np.exp(-r_u*S**beta*RI)
    denom = alpha*k_u - r_l*S**beta*(1 - np.exp(-r_u*S**beta*RI))
    return 1 - numer/denom



def G_l_equil(r_l, r_u, k_l,  k_u, S, beta, alpha, RI, severity):
    """
    Assumes G_l IS in steady state with G_u, 
    and computes G_l in equilibrium with G_u_max. 
    """

    r_up = r_u*S**beta 

    G_u_max = k_u/(1-severity)*(1 - severity - np.exp(-r_up*RI)) \
    							/(1- np.exp(-r_up*RI))
    G_u_max = greater_than_zero(G_u_max)

    r_lp = r_l*S**beta 
    G_l_eq = k_l*(1- alpha*G_u_max/r_lp)
    
    return greater_than_zero(G_l_eq)


def greater_than_zero(G_o):
    """
    set all values smaller than zero to zero
    
    compatible with floats, arrays and lists...  

    """
    if np.size(G_o) == 1:
        G_o = max(G_o, 0)
    else:
        G_o[G_o<0]= 0
    return G_o       