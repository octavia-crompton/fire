import os

import numpy as np
import pandas as pd
import scipy
import scipy.special

param_vars = ['alpha', 'beta', 'k_u', 'k_l', 'r_u', 'r_l', 'S', 'dt', 'dt_p',
              'seed', 'ti', 'tmax', 'RI', 'ignition_type', 'chi', 'severity_type',
              'severity', 'std_severity', 'r', 'a', 'b', 'init']


class RCSR:
    """
    Contains the two layer fire model 
    
    """

    # noinspection PyTypeChecker
    def __init__(self, params=None):
        """
        """
        params = default_params(params)

        for k, v in params.items():
            setattr(self, k, v)

        if self.init == "ICB":
            if self.veg == 1:
                self.G_uo = self.k_u * 0.9
                self.G_lo = self.k_l * 0.1

            elif self.veg == 2:
                self.G_uo = self.k_u * 0.1
                self.G_lo = self.k_l * 0.9

        else:
            self.G_uo = self.k_u / 10.
            self.G_lo = self.k_l / 10.

        self.G_u = self.G_uo
        self.G_l = self.G_lo
        self.G_u_list = [self.G_uo]
        self.G_l_list = [self.G_lo]

        self.fires = 0
        self.time_past_fire = 0
        self.cat_severity = 0
        self.t = 0
        self.RI = int(self.RI)

        self.p = 1 / self.RI
        self.times = np.arange(0, self.tmax + self.ti + self.dt)
        self.t_p = np.arange(self.ti, self.tmax + self.ti + self.dt_p,
                             self.dt_p)

        self.nt_steps = int(self.tmax / self.dt)
        self.nt_spinup = int(self.ti / self.dt)

        np.random.seed(int(self.seed))

        if self.ignition_type == "fixed":
            self.ignition_list = self.times[::self.RI][1:]

        elif self.ignition_type == "random":
            self.ignition_list = np.random.rand(int(self.tmax + self.ti) + 1)

        else:
            self.ignition_list = np.random.rand(int(self.tmax + self.ti) + 1)

        self.n_fires = len(self.ignition_list)

        if self.severity_type == "random":
            self.severity = np.round(self.severity, 4)

            self.severity_list = severity_sampler(n=self.n_fires,
                                                  std_severity=self.std_severity,
                                                  severity=self.severity,
                                                  a=self.a, b=self.b,
                                                  r=self.r, seed=0)
            np.random.seed(int(self.seed))
            np.random.shuffle(self.severity_list)

        elif self.severity_type == "fixed":
            self.severity = np.round(self.severity, 4)
            self.severity_list = np.ones([self.n_fires, 2]) * self.severity

        elif self.severity_type == "sample":
            model_dir = os.path.dirname(__file__)
            severity_csv = os.path.join(model_dir, "severity.csv")
            severities = np.loadtxt(severity_csv)
            severities = np.tile(severities, (2, 1)).T
            np.random.shuffle(severities[:, 1])
            
            self.severity_list = severities
            self.severity = np.mean(severities)

            np.random.seed(int(self.seed))
            np.random.shuffle(self.severity_list)
            print(self.severity_list[0],)


        self.record = pd.DataFrame(
            columns=["year", "time_past_fire",
                     "G_u_max", "G_u_mean",
                     "G_l_max", "G_l_mean",
                     "u_severity", "l_severity",
                     "G_u_mean_a", "G_l_mean_a"
                     ])

    def extract_params(self):
        """
        Extract parameter dictionary from class object
        """
        params = {}
        for k in list(default_params().keys()):
            params[k] = vars(self)[k]
        return params

    def __repr__(self):

        # attrs = vars(self)
        attrs = self.extract_params()
        d = '{' + ',\n  '.join("%s: %s" % item for
                               item in attrs.items()) + '}'
        return d

    def g_u(self):
        """
        Upper layer growth rate
        """
        G_u = self.G_u
        r_u = self.r_u * self.f_S()
        k_u = self.k_u
        return r_u * G_u * (1 - G_u / k_u)

    def g_l(self):
        """
        Lower layer growth rate
        """
        G_l = self.G_l
        G_u = self.G_u
        r_l = self.r_l * self.f_S()
        k_l = self.k_l
        alpha = self.alpha

        return r_l * G_l * (1 - G_l / k_l) - alpha * G_l * G_u

    def f_S(self):
        """
        vegetation growth limiting factor
        """
        return self.S ** self.beta

    def timestep(self):
        """
        Euler step forward a dt increments
        """
        self.t += self.dt
        self.time_past_fire += self.dt

        self.G_u = self.G_u + self.g_u() * self.dt
        self.G_l = self.G_l + self.g_l() * self.dt

        self.G_u = max(self.G_u, 0)
        self.G_l = max(self.G_l, 0)

        u_severity, l_severity = 0, 0

        t = round(self.t, 4)

        if self.ignition_type == "fixed":

            if t in self.ignition_list:
                u_severity, l_severity = self.severity_list[self.fires]

        else:
            if round(self.t, 0) == t:
                u_severity, l_severity = self.random_ignition()

        if (u_severity > 1e-3) or (l_severity > 1e-3):
            G_u = self.G_u * (1 - u_severity)
            G_l = self.G_l * (1 - l_severity)

            steps_since_fire = 1 + int(self.time_past_fire / self.dt_p)
            G_u_mean_c = np.mean(self.G_u_list[-steps_since_fire:])
            G_l_mean_c = np.mean(self.G_l_list[-steps_since_fire:])

            u_severity, l_severity = self.severity_list[self.fires]

            G_u_mean_a = self.mean_G_u(recomp=True)
            G_l_mean_a = self.mean_G_l(recomp=True)

            self.record = self.record.append({"year": self.t,
                                              "time_past_fire": self.time_past_fire,
                                              "G_u_max": self.G_u,
                                              "G_u_mean": G_u_mean_c,
                                              "G_l_max": self.G_l,
                                              "G_l_mean": G_l_mean_c,
                                              "u_severity": u_severity,
                                              "l_severity": l_severity,
                                              "G_u_mean_a": G_u_mean_a,
                                              "G_l_mean_a": G_l_mean_a
                                              }, ignore_index=True)

            self.fires += 1
            self.time_past_fire = 0.0
            self.G_l = G_l
            self.G_u = G_u

        mod = int(-np.log10(self.dt_p))

        if self.t >= self.ti:

            if round(self.t, 4) == round(self.t, mod):
                self.G_u_list.append(self.G_u)
                self.G_l_list.append(self.G_l)

    def ignite_threshold(self):
        """
        probability is the reciprocal of the prescribed RI
        """
        p = 1. / self.RI

        return p

    def G_l_ignite_threshold(self):
        """
        Ignition probability, prescribed as 1/RI
    
        Notes
        -----
        RI_l = RI + chi*RI - chi*RI/k_l*G_l
        RI_l(G_l = k_l)  is RI, that is, the minimum RI is the prescribed RI

        """
        RI_l = self.RI + self.chi * self.RI - \
               self.chi * self.RI / self.k_l * self.G_l

        p = 1 / RI_l

        return p

    def random_ignition(self):
        """
        Predict fire ignition with probability 'p'
        """
        ignite = self.ignition_list[int(self.t)]

        if self.ignition_type == "random":
            p = self.ignite_threshold()
        else:
            assert self.ignition_type == "G_l"
            p = self.G_l_ignite_threshold()

        if ignite < p:
            severity = self.severity_list[self.fires]
            u_severity, l_severity = severity[0], severity[1]
        else:
            l_severity, u_severity = 0, 0

        return u_severity, l_severity

    def run(self):
        """
        Run the model for ti + tmax timesteps, where ti is the spinup time
        """
        if self.ti > 0:

            for i in range(self.nt_spinup):
                self.timestep()

        # restart after spin-up
        self.G_u_list = [self.G_u]
        self.G_l_list = [self.G_l]

        self.record = pd.DataFrame(
            columns=["year", "time_past_fire",
                     "G_u_max", "G_u_mean",
                     "G_l_max", "G_l_mean",
                     "u_severity", "l_severity",
                     "G_u_mean_a", "G_l_mean_a"
                     ])

        self.spinup_fires = self.fires

        for i in range(self.nt_steps):
            self.timestep()
        self.G_u_list = np.array(self.G_u_list)
        self.G_l_list = np.array(self.G_l_list)
        self.t_p = self.t_p[:len(self.G_u_list)]

        self.RI_actual = np.mean(self.record.time_past_fire)
        if len(self.record) > 1:
            self.compute_statistics()

        self.G_u_mean_list = np.cumsum(self.G_u_list) / (
            np.arange(1, len(self.G_u_list) + 1))
        self.G_l_mean_list = np.cumsum(self.G_l_list) / (
            np.arange(1, len(self.G_l_list) + 1))

    def compute_statistics(self):
        """
        """
        to = np.where(self.t_p >= self.record.iloc[0]["year"])[0][0]

        G_u_list = self.G_u_list[to:to + int(self.RI / self.dt_p)]
        G_l_list = self.G_l_list[to:to + int(self.RI / self.dt_p)]

        self.G_u_min_c = np.min(G_u_list)
        self.G_u_mean_c = np.mean(G_u_list)

        self.G_l_min_c = np.min(G_l_list)
        self.G_l_max_c = np.min(G_l_list) / (1 - self.severity)
        self.G_l_mean_c = np.mean(G_l_list)

    # STABILITY FUNCTIONS
    def max_severity(self, r, RI):
        """
        For a given growth rate and return interval, 
        find the maximum severity for which G > 0.
        """
        return 1 - np.exp(-r * RI)

    def max_severity_u(self):
        """
        For the system's growth rate and return interval, 
        find the maximum severity for which G_u > 0
        """
        r = self.r_u * self.S ** self.beta
        RI = self.RI
        phi_S_max = 1 - np.exp(-r * RI)
        return phi_S_max

    def min_RI(self, r, severity):
        """
        For a given growth rate and severity, 
        find the minimuim return interval for which G>0
        """
        return -1. / r * np.log(1 - severity)

    def min_RI_u(self):
        """
        For the system's growth rate and severity, 
        find the minimuim return interval for which G_u>0
        """
        r_up = self.r_u * self.S ** self.beta
        severity = self.severity

        return -1. / r_up * np.log(1 - severity)

    def max_RI_l(self):
        a = - np.log(1 - self.severity) / (self.r_u * self.S ** self.beta)
        numer = self.alpha * self.k_u - self.r_u * self.S ** self.beta
        denom = (self.alpha * self.k_u - self.r_l * self.S ** self.beta)
        return a * numer / denom

    def min_RI_l(self):
        r = self.r_l * self.S ** self.beta
        return -1. / r * np.log(1 - self.severity)

    # AVERAGE BIOMASS
    def mean_G(self, k, r, RI, severity):
        """
        Returns mean(G) for general coefficients
        """
        G_mean = k * (1 + np.log(1 - severity) / (r * RI))

        return greater_than_zero(G_mean)

    def mean_G_u(self, recomp=False):
        """
        Returns the mean upper canopy biomass 
        """
        k = self.k_u
        r = self.r_u * self.S ** self.beta

        if not recomp:
            RI = self.RI
        else:
            RI = self.record.time_past_fire.mean()

        if not recomp:
            if self.severity_type == "random":
                severity = predict_truncated_mean(
                    self.severity, self.std_severity, self.a, self.b)
            else:
                severity = self.severity
        else:
            severity = self.record.u_severity.mean()

        G_u_mean = k * (1 + np.log(1 - severity) / (r * RI))

        return greater_than_zero(G_u_mean)

    def r_lp(self, recomp):
        """
        Computes the modified lower canopy growth rate as:

            r_lp = r_l*S^beta - alpha*G_u_mean
        """

        G_u_mean = self.mean_G_u(recomp)

        r_lp = self.r_l * self.S ** self.beta - self.alpha * G_u_mean

        return r_lp

    def k_lp(self, recomp):
        """
        Computes the modified lower canopy carrying capacity (kg/m2) as:

            k_lp  = k_l r_lp / (r_l S**beta)

        """
        r_lp = self.r_lp(recomp)

        return self.k_l * r_lp / (self.r_l * self.S ** self.beta)

    def mean_G_l(self, recomp=False):
        """
        Estimates the mean lower canopy biomass 
    
        Parameters:
        ----------
        recomp: bool
            If true, recompute the mean RI from the fire record

        Returns:
        --------
        G_l_mean: float 
            The predicted mean G_l
        
        Notes:
        ------
        (a) approximating G_u as constant (G_u_mean) 
        (b) rewriting the G_l equation in logistic form (r_lp, k_lp)
        (c) using the same approach as used to solve for G_u_mean
        
        The approximation is given as

          G_l_mean =   k_lp ( 1 +  log(1- severity)/(r_lp*RI))
        """

        if not recomp:
            RI = self.RI
        else:
            RI = self.record.time_past_fire.mean()

        r_lp = self.r_lp(recomp)
        k_lp = self.k_lp(recomp)

        if not recomp:
            if self.severity_type == "random":
                severity = predict_truncated_mean(
                    self.severity, self.std_severity, self.a, self.b)
            else:
                severity = self.severity
        else:
            severity = self.record.l_severity.mean()

        G_l_mean = k_lp * (1 + np.log(1 - severity) / (r_lp * RI))

        return greater_than_zero(G_l_mean)

    def fix_G_l(self, gamma):
        """
        Solves for mean(G_l) given the assumption:
            mean(G_u) = gamma*k_u

        Parameters:
        ----------
        gamma: float
            defined as mean(G_u) = gamma*k_u
        """
        numer = self.alpha * self.k_u * gamma + \
                (1 - gamma) * self.r_u * self.S ** self.beta
        denom = (self.r_l * self.S ** self.beta)

        return 1 - numer / denom

    def G_postfire(self, r, k, RI, severity):
        """        
        Computes the biomass immediately after each fire,
        assuming the system is in dynamic equilibrium

        Parameters:
        -----------
        r : float
            generalized growth rate
        k : float
            generalized growth rate            
        """
        x = r * RI
        phi_R = 1 - severity  # fraction of biomass remaining

        numer = phi_R - np.exp(-x)
        denom = 1 - np.exp(-x)
        G_o = k * numer / denom

        return greater_than_zero(G_o)

    def G_prefire(self, r, k, RI, severity):
        """
        Computes the biomass immediately after each fire,
        once the system is in dynamic equilibrium
        """
        x = r * RI
        phi_R = 1 - severity  # fraction of biomass remaining

        numer = phi_R - np.exp(-x)
        denom = 1 - np.exp(-x)
        G_o = k * numer / denom / phi_R

        return greater_than_zero(G_o)

    def G_u_postfire(self):
        """
        Computes G_u immediately after each fire.
        """
        RI = self.RI
        severity = self.severity
        r = self.r_u * self.S ** self.beta
        k = self.k_u
        G_u_min = self.G_postfire(r, k, RI, severity)

        return greater_than_zero(G_u_min)

    def G_l_postfire(self, recomp=False):
        """
        Returns the mean lower canopy biomass by
        (i) approximating G_u as constant (G_u_max) 
        (ii) rewriting the equation for G_l in logistic form,
        (iii) using the same approach to solve for G_l_max/G_l_min
        
        """

        r_lp = self.r_lp(recomp)
        k_lp = self.k_lp(recomp)
        RI = self.RI
        G_l_min = self.G_postfire(r_lp, k_lp, RI, self.severity)

        return greater_than_zero(G_l_min)

    def G_l_equil(self):
        """
        Assume G_l in steady state with G_u, and computes G_l in equilibrium with G_u_max. 

        """
        G_u_min = self.G_u_postfire()
        G_u_max = G_u_min / (1 - self.severity)

        r_lp = self.r_l * self.S ** self.beta
        k_l = self.k_l

        G_l_eq = k_l * (1 - self.alpha * G_u_max / r_lp)

        return max(G_l_eq, 0)

    def G_analytic(self, G_o, r, k, ts):
        """
        Returns an analytic solution to the logistic equation 

        Parameters:
        ----------
        r : float 
            growth rate
        k : float
            carrying capacity
        G_o : float
            initial biomass
        ts : list
            list of times

        Returns:
        -------
        G : float
            biomass at times `ts`
        """

        denom = G_o + (k - G_o) * np.exp(-r * ts)

        return k * G_o / denom

    def G_u_analytic(self, times, G_o=None):
        """
        Compute the analytic solution for upper canopy, for a 
        given IC and time range

        Parameters:
        ----------
        times : list
            
        G_o : float
            initial biomass
        """
        if G_o is None:
            G_o = self.G_uo

        k = self.k_u
        r = self.r_u * self.S ** self.beta

        return self.G_analytic(G_o, r, k, times)

    def integrate_G_analytic(self, G_o, r, k, t):
        """
        Returns the integral of the logistic equation solution
        
        Parameters:
        ----------
        G_o : float
            initial biomass
        r : float 
            growth rate
        k : float
            carrying capacity
        t : float
            final time (to=0 assumed)
        """
        dum = np.abs(k - G_o + G_o * np.exp(r * t))
        return k / r * np.log(dum)

    def integrate_G_u_analytic(self, G_o, tf, t0=0):
        """
        Returns the mean upper canopy biomass by integrating the 
        analytic solution
        """
        k = self.k_u
        r = self.r_u * self.S ** self.beta
        upper = self.integrate_G_analytic(G_o, r, k, tf)
        lower = self.integrate_G_analytic(G_o, r, k, t0)

        return (upper - lower) / (tf - t0)

    def predict_G_l(self):
        """
        helper function to obtain G_l_min and G_l_max as a tuple
        """
        G_l_min_a = self.G_l_postfire()
        G_l_max_a = G_l_min_a / (1 - self.severity)

        return G_l_min_a, G_l_max_a

    def predict_G_u(self):
        """
        Helper function to obtain G_u_min and G_u_max as a tuple
        """
        G_u_min_a = self.G_u_postfire()
        G_u_max_a = G_u_min_a / (1 - self.severity)

        return G_u_min_a, G_u_max_a

    def equil_time(self, gamma):
        """
        Time for the upper canopy biomass to reach gamma*something?
        """
        r = self.r_u * self.S ** self.beta

        is_nonzero = (1 - self.severity - np.exp(-r * self.RI)) > 0
        if is_nonzero:
            t_eq = 1 / r * np.log(gamma / (1 - gamma) * self.severity / (
                    1 - self.severity - np.exp(-r * self.RI)))
        else:
            t_eq = 0

        return t_eq

    def G_u_deriv_c(self):
        """
        Compute the minimum derivative from the G_u timeseries
        """
        deriv = np.diff(self.G_u_list)
        deriv[deriv < 0] = np.nan

        deriv = deriv / self.dt_p
        return np.nanmin(deriv)

    def G_u_deriv_a(self):
        """
        Compute the minimum derivative from the derivative 
        of the G_u(t) solution.
        ** not the smartest approach! **
        """
        G = self.G_u_postfire()  # biomass after fires
        k = self.k_u
        r = self.r_u * self.S ** self.beta
        t = np.arange(0, self.RI, self.dt_p)

        deriv = (G * k * (-G + k) * r * np.exp(r * t)) / (k + G * (-1 + np.exp(r * t))) ** 2

        return np.nanmin(deriv)


def default_params(update=None):
    """
    Contains default parameters values, which the dictionary 
    `update` will overwrite.
    
    Useful for rapidly re-initializing stuff
    RCSR uses this function to initialize parameter values
    """

    params = {
        "alpha": 0.05,
        "beta": 0.5,
        "k_u": 60.,
        "k_l": 5.,
        "r_u": 0.25,
        "r_l": 1.5,
        "S": 0.5,
        "dt": 0.01,
        "dt_p": 0.1,
        "seed": 0,
        "ti": 0,
        "tmax": 3000,
        "RI": 60,
        "ignition_type": "fixed",
        "severity_type": "fixed",
        "severity": 0.5,
        "std_severity": 0.1,
        "init": 0,
        "r": 0.5,
        "a": 0.01,
        "b": 0.99,
    }
    if update is not None:
        params.update(update)
    return params


def runmodel(param):
    p = RCSR(param)
    p.run()
    return p


"""
 Code to sample severity
"""


def severity_sampler(n=1e5, severity=0.3, std_severity=0.01,
                     r=0.5, seed=0, include=False,
                     a=0, b=1):
    """
    Multivariate sampler to specify how the probability densities of 
    lower and upper fire severity are correlated. 
     
    """
    n = int(n)
    np.random.seed(seed)

    sigma_phi = std_severity ** 2

    K_0 = np.array([[1, r],
                    [r, 1]]) * sigma_phi

    x = np.random.multivariate_normal(
        [severity, severity], cov=K_0, size=2 * n)
    y = x.copy()
    zero_inds = np.where(1 - (x < a).sum(axis=1) > 0)[0]
    if len(zero_inds) > 0:
        x = x[zero_inds]
    one_inds = np.where(1 - (x > b).sum(axis=1) > 0)[0]

    if len(one_inds) > 0:
        x = x[one_inds]

    x = x[:n]
    y = y[:n]

    if include:
        return x, y
    else:
        return x


def phi_norm(d):
    """
    Helper function to predict truncated Gaussian moments
    """
    phi = 1 / (np.sqrt(2. * np.pi)) * np.exp(-1 / 2. * d ** 2)
    return phi


def Phi_norm(d):
    """
    Helper function to predict truncated Gaussian moments
    """
    return 0.5 * (1 + scipy.special.erf(d / np.sqrt(2.)))


def predict_truncated_mean(severity, std_severity, a, b):
    """
    Predict the mean of a truncated normal distribution 
    """
    alpha_norm = (a - severity) / std_severity
    beta_norm = (b - severity) / std_severity

    numer = (phi_norm(alpha_norm) - phi_norm(beta_norm))
    denom = (Phi_norm(beta_norm) - Phi_norm(alpha_norm))

    return severity + std_severity * numer / denom


def predict_truncated_std(severity, std_severity, a, b):
    """
    Predict the standard deviation of a truncated normal
    distribution 
    """
    alpha_norm = (a - severity) / std_severity
    beta_norm = (b - severity) / std_severity

    # Term 1
    numer = alpha_norm * phi_norm(alpha_norm) - \
            beta_norm * phi_norm(beta_norm)
    denom = (Phi_norm(beta_norm) - Phi_norm(alpha_norm))
    term1 = numer / denom

    # Term 2
    numer = phi_norm(alpha_norm) - phi_norm(beta_norm)
    denom = (Phi_norm(beta_norm) - Phi_norm(alpha_norm))
    term2 = (numer / denom) ** 2

    return std_severity * (1 + term1 - term2) ** 0.5


"""
 Post-processing
 """


def compute_errors(p):
    """
    Compares analytic predictions and simulations results 
    """
    to = np.where(p.t_p >= p.record.iloc[0]["year"])[0][0]

    G_u_list = p.G_u_list[to:to + int(p.RI / p.dt_p)]
    gamma = 0.8

    G_u_min_c = np.min(G_u_list)
    G_u_mean_c = np.mean(G_u_list)

    G_l_min_c = np.min(p.G_l_list[to:to + int(p.RI / p.dt_p)])
    G_l_mean_c = np.mean(p.G_l_list[to:to + int(p.RI / p.dt_p)])

    G_u_min_a, G_u_max_a = p.predict_G_u()

    df = pd.DataFrame({
        "analytic": {"G_u_min": G_u_min_a,
                     "G_u_max": G_u_min_a / (1 - p.severity),
                     "G_u_mean": p.mean_G_u(),
                     "G_u_deriv": p.G_u_deriv_a(),
                     "G_l_min": p.G_l_postfire(),
                     "G_l_max": p.G_l_postfire() / (1 - p.severity),
                     "G_l_mean": p.mean_G_l()
                     },
        "computed": {"G_u_min": G_u_min_c,
                     "G_u_max": G_u_min_c / (1 - p.severity),
                     "G_u_mean": G_u_mean_c,
                     "G_u_deriv": p.G_u_deriv_c(),
                     "G_l_min": G_l_min_c,
                     "G_l_max": G_l_min_c / (1 - p.severity),
                     "G_l_mean": G_l_mean_c
                     },
    })

    df["errors"] = df["computed"] - df["analytic"]
    df["errors_percent"] = (df["computed"]
                            - df["analytic"]) / df["computed"] * 100

    too_small = (np.all([(df["computed"] < 1,
                          df["analytic"] < 1)], axis=0))[0]
    df["errors_percent"][too_small] = 0

    a = pd.Series(df["analytic"]).copy()
    a.index = [d + "_a" for d in df.index]

    c = pd.Series(df["computed"]).copy()
    c.index = [d + "_c" for d in df.index]

    e = pd.Series(df["errors"]).copy()
    e.index = [d + "_e" for d in df.index]

    pe = pd.Series(df["errors_percent"]).copy()
    pe.index = [d + "_pe" for d in df.index]

    dfl = a.append(c).append(e).append(pe)

    return df, dfl


def compute_errors_mean(p):
    """
    Compares analytic predictions and simulations results     
    
    Notes:
    ------
    Unlike 'compute_errors', this function uses the entire series
    Useful for random ignition and random severity cases
    
    """

    RI = p.record.time_past_fire.mean()
    RI_std = p.record.time_past_fire.std()

    phi = p.record.u_severity.mean()
    phi_std = p.record.u_severity.std()

    phi_tr_m = predict_truncated_mean(p.severity,
                                      p.std_severity, p.a, p.b)
    phi_tr_std = predict_truncated_std(p.severity,
                                       p.std_severity, p.a, p.b)

    df = pd.DataFrame({
        "analytic": {
            "G_u_mean": p.mean_G_u(recomp=True),
            "G_l_mean": p.mean_G_l(recomp=True),
            "RI": p.RI,
            "RI_std": p.RI,
            "severity": phi_tr_m,
            "severity_std": phi_tr_std,
        },
        "computed": {
            "G_u_mean": p.G_u_list.mean(),
            "G_l_mean": p.G_l_list.mean(),
            "RI": RI,
            "severity": phi,
            "RI_std": RI_std,
            "severity_std": phi_std
        },
    })

    df["errors"] = df["computed"] - df["analytic"]
    df["errors_percent"] = df["errors"] / df["computed"] * 100

    a = pd.Series(df["analytic"]).copy()
    a.index = [d + "_a" for d in df.index]

    c = pd.Series(df["computed"]).copy()
    c.index = [d + "_c" for d in df.index]

    e = pd.Series(df["errors"]).copy()
    e.index = [d + "_e" for d in df.index]

    pe = pd.Series(df["errors_percent"]).copy()
    pe.index = [d + "_pe" for d in df.index]

    dfl = a.append(c).append(e).append(pe)

    return df, dfl


def summary(p):
    """
    Summarize a simulation instance
    """
    RI = p.record.time_past_fire.mean()
    RI_std = p.record.time_past_fire.std()

    phi = p.record.u_severity.mean()
    phi_std = p.record.u_severity.std()

    df = pd.Series({
        "G_u_mean": p.G_u_list.mean(),
        "G_l_mean": p.G_l_list.mean(),
        "RI_c": RI,
        "severity_c": phi,
        "RI_std_c": RI_std,
        "severity_std_c": phi_std,
        "p": 1 / p.RI
    })

    return df


def compute_all_summary(all_sims, output_dir=None):
    """
    Compute the errors for a list of RCSR instances, with
    regular ignition and severities
    
    """
    res = pd.DataFrame()

    for key in all_sims.index:
        p = all_sims.loc[key][0]

        var_list = list(default_params().keys())
        param = pd.Series(vars(p))[var_list]

        df = summary(p)

        df = df.append(param)

        res = res.append(df, ignore_index=True)
    res.index = all_sims.index

    if output_dir:
        res.to_pickle(output_dir + "/summary.pkl")
    return res


"""
Utility
"""


def greater_than_zero(G_o):
    """
    set all values smaller than zero to zero
    
    compatible with floats, arrays and lists...  

    """
    if np.size(G_o) == 1:
        G_o = max(G_o, 0)
    else:
        G_o[G_o < 0] = 0
    return G_o


def less_than_one(x):
    """
    Set all values smaller than zero to zero
    
    compatible with floats, arrays and lists...  

    """
    if np.size(x) == 1:
        x = min(1, x)
    else:
        x[x > 1] = 1
    return x


def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))


def difference(a, b):
    """
    Returns a-b for floats and ints,
    returns "a/b" for strings
    """
    if type(a) is not str:
        # return round(a-b, 8)
        return '{0:.2f}; default={1:.2f}'.format(a, b)
    else:
        return '; default='.join([a, b])


def diff_from_default(params):
    """
    Compares a parameter dictionary 
    """
    value = {k: difference(params[k], default_params()[k]) for k in set(params)
             if params[k] != default_params()[k]}
    return value
