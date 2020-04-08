import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import time
import matplotlib.colors as colors
import copy
import math

"""
Plotting functions
"""  
def canopy_plot(p, nfire = 100, title = ""):
    """
    Plot lower and upper canopy biomass on the same axis

    Parameters:
    p  : RCSR instance
       RCSR simulation
    nfire : int
        number of fires
    title: str
        plot title  
    """
    fig, ax = plt.subplots(1, figsize = (14,4) )

    to = int(-nfire*p.RI/p.dt_p)
    
    line_u = ax.plot(p.t_p[to:-1:int(1/p.dt_p)], 
        p.G_u_list[to:-1:int(1/p.dt_p)], '-', lw = 1, 
        label ="upper canopy")        
    line_l = ax.plot(p.t_p[to:-1:int(1/p.dt_p)], 
        p.G_l_list[to:-1:int(1/p.dt_p)], 
        '-', lw = 1, label = "lower canopy")       
    
    c_u = line_u[0].get_color()
    c_l = line_l[0].get_color()
    
    ax.set_ylabel("biomass")
    ax.set_ylim(0,)
    ax.legend()

    severity = p.severity

    G_u_min_a = p.G_u_postfire()
    G_u_max_a = G_u_min_a/(1-severity)
    
    if G_u_min_a > 0:
        plt.axhline(G_u_min_a, ls = "-", lw = 1, alpha = 0.7, c = c_u)    
        plt.axhline(G_u_max_a, ls = "-", lw = 1, alpha = 0.7, c = c_u)    

    G_l_postfire = p.G_l_postfire()
    if G_l_postfire > 0:
        plt.axhline(p.G_l_postfire(), ls = '--', lw = 1, c = c_l)
        plt.axhline(p.G_l_postfire()/(1-severity), 
            ls = '--', lw = 1, c = c_l)
    ax.set_title(title)

"""
Canopy plot for "paired simulations"
"""

def canopy_compare(p_wf, p_nf, nfire = 1000, title = ""):
    """
    Plot comparison of two cases: with and without G_l feedback


    """
    fig, axes = plt.subplots(2, 1, figsize = (14,6) )
    ax = axes[0] 

    to = int(-nfire*p_nf.RI/p_nf.dt_p)
    
    line_l = ax.plot(p_nf.t_p[to:-1:int(1/p_nf.dt_p)], 
        p_nf.G_u_list[to:-1:int(1/p_nf.dt_p)], 
        '-', lw = 1, label = "no feedback")  
    c_l = line_l[0].get_color()
    
    ax.plot(p_wf.t_p[to:-1:int(1/p_nf.dt_p)], 
            p_wf.G_u_list[to:-1:int(1/p_nf.dt_p)], '--',
            label ="$G_l$ feedback")        
    ax.set_ylabel("$G_u$ n")
    ax.set_ylim(0,)
    ax.legend()

    severity = p_nf.severity

    G_u_postfire = p_nf.G_u_postfire()
    if G_u_postfire > 0:
        ax.axhline(p_nf.mean_G_u(), ls = '--', lw = 1, c = c_l)

    ax.set_title(title)
    
   
    ax = axes[1]
    to = int(-nfire*p_nf.RI/p_nf.dt_p)
  
    line_l = ax.plot(p_nf.t_p[to:-1:int(1/p_nf.dt_p)], 
        p_nf.G_l_list[to:-1:int(1/p_nf.dt_p)], 
        '-', lw = 1, label = "no feedback")  
    c_l = line_l[0].get_color()
    
    ax.plot(p_wf.t_p[to:-1:int(1/p_nf.dt_p)], 
            p_wf.G_l_list[to:-1:int(1/p_nf.dt_p)], '--',
            label ="$G_l$ feedback")        
    ax.set_ylabel("$G_l$")
    ax.set_ylim(0,)
    ax.legend()

    severity = p_nf.severity

    G_l_postfire = p_nf.G_l_postfire()
    if G_l_postfire > 0:
        ax.axhline(p_nf.mean_G_l(), ls = '--', lw = 1, c = c_l)

    ax.set_title(title)
    
    

def colormap(xc,yc, array, ax = '',
             colorbar = True,             
             bounds = '', clabel = '',
             cmin = False, cmax = False,           
             cround = '', cfontsize = 14,
             cmap = "Blues", ax_ticks = True):

    """
    Recycled from the SVE code, but used by grid plots 
    """  
    if ax == '':      
        fig = plt.figure(figsize = (5,5))
        ax = fig.add_subplot(111)
    else:
        fig = plt.gcf()
  
    if bounds == '':
    
        scale_vals =  array[:-1, :-1].ravel()
        
        if type(cmin) == bool:            
            cmin = np.nanmin(scale_vals)

        if type(cmax) == bool:            
            cmax = np.nanmax(scale_vals)
            
        bounds = np.linspace(cmin, cmax, 100)
        
        if cround != '':
            cmin = np.round(cmin, cround)
            bounds = np.arange(cmin, cmax, 1/10.**cround)

        if np.sum(array.astype(int) - array) == 0:
             bounds = np.arange(cmin-1, cmax+1.1,1)
            
        if np.nanstd(array) < 1e-5:
            bounds = np.linspace(cmin-1, cmax+1, 10)

    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)    

    zplot = ax.pcolormesh( xc,yc, array,
                           norm = norm,
                           cmap=cmap, alpha= 1);
                                               
    if colorbar == True:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib import ticker

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        cbh = fig.colorbar(zplot,cax = cax,shrink=1)
        cbh.set_label(clabel, fontsize = cfontsize)

        tick_locator = ticker.MaxNLocator(nbins=6)
        cbh.locator = tick_locator
        cbh.update_ticks()
        cbh.ax.tick_params(labelsize=cfontsize) 
    ax.set_ylim(yc.min(), yc.max())
    ax.set_xlim(xc.min(), xc.max())
    if ax_ticks == False:
        ax.set_xticks([], []);
        ax.set_yticks([], []);
    else:
        ax.set_xticks(xc[0,::int(xc.shape[1]/5)]);
        ax.set_yticks(yc[::int(yc.shape[0]/5), 0]);    
    return ax, zplot

    


def check_convergence(p, to = 0, slice = 1000):
    """
    """
    to = 0
    G_u_mean_list = p.G_u_mean_list
    G_l_mean_list = p.G_l_mean_list
         
    t_p = p.t_p[to:]

    fig, axes = plt.subplots(1, 2, figsize = (13, 4))

    ax = axes[0]
    line = ax.plot(t_p[::slice], G_u_mean_list[::slice], label = "$G_u$")
    line = ax.plot(p.record.year, p.record.G_u_mean_a, 
        label =  "predicted $G_u$")
    ax.set_xlabel("time (year)")
    ax.set_ylabel("$G_u$")

    ax.legend()

    ax = axes[1]
    ax.plot(t_p[::slice], G_l_mean_list[::slice] , label = "$G_l$")
    line = ax.plot(p.record.year, p.record.G_l_mean_a, label =
      "predicted $G_l$")

    ax.set_xlabel("time (year)")
    ax.set_ylabel("$G_l$")

    ax.legend()


###  Stability
def plot_G_u_a(p,ax):
    """
    """
    G_u_min_a, G_u_max_a = p.predict_G_u()
    
    if G_u_min_a > 0:
    
        ax.axhline(G_u_min_a, ls = "--", lw = 1)
        ax.axhline(G_u_max_a, ls = "--", lw = 1)

        t_int = np.arange(0, p.RI, .1)
        
        G_u_analytic = p.G_u_analytic(t_int, G_u_min_a)
        
        plt.plot(t_int +p.record.iloc[0]["year"], G_u_analytic)
    
        plt.axhline(p.mean_G_u(), ls = "-.", lw = 1)

def plot_G_l_a(p,ax):
    """
    """
    G_l_min_a, G_l_max_a = p.predict_G_l()
    
    if G_l_min_a > 0 :
    
        ax.axhline(G_l_min_a, ls = "--", lw = 1, c= 'k')
        ax.axhline(G_l_max_a, ls = "--", lw = 1, c= 'k')

def plot_fire_events(p, ax):
    """
    plot the fire events
    """
    for i in range(len(p.record))[:5]:
        ax.axvline(p.record.year[i], ls = "--", lw = 1)           


### see CRUR_analytic
def expand(x):
    
    x = np.insert(x, x.shape[1],2*x[:, -1]-x[:, -2] , axis = 1)
    x = np.insert(x, x.shape[0],2*x[-1]-x[-2] , axis = 0)
    return x

def expand_z(x):
    
    x = np.insert(x, x.shape[1],0 , axis = 1)
    x = np.insert(x, x.shape[0],0 , axis = 0)
    return x


def reformat_to_grid(subset, x_var, y_var):
    maps = {}
    for fld in  subset.columns:#["ignition_freq", "" ]:
        Nx = len(np.unique(subset[x_var]))
        Ny = len(np.unique(subset[y_var]))    
        maps[fld] = (np.reshape(np.array(subset[fld]), [Ny,Nx]))
    return maps


def plot_G_grid(subset, x_var, y_var):
    """
    Gridded plot of G_u and G_l predicted mean and errors, 
    over a 2D grid of parameter values (`x_var` and `y_var`)
    
    """
    maps = reformat_to_grid(subset,x_var, y_var)

    y = expand(maps[y_var].copy())
    x = expand(maps[x_var].copy())

    fig, axes = plt.subplots(2,2, figsize = (12, 8))

    ax = axes[0, 0]
    z = expand_z(np.array(maps["G_u_mean_c"]))
    ax.set_xticks([], []);    
    ax, zplot = colormap(x,y, z, ax= ax)
    ax.set_ylabel("Severity")
    ax.set_title(r"$\bar G_u$  ")
    
    ax = axes[0, 1]
    z = expand_z(np.array(maps["G_l_mean_c"]))
    ax, zplot = colormap(x,y, z, ax= ax)
    ax.set_yticks([], []);
    ax.set_xticks([], []);        
    ax.set_title(r"$\bar G_l$  ")
    
    ax = axes[1, 0]
    z = expand_z(np.array(maps["G_u_mean_e"]))
    ax, zplot = colormap(x,y, z, ax= ax)
    ax.set_ylabel("Severity")
    ax.set_title(r"$\bar G_u$ error")
    ax.set_xlabel("Return interval")    

    ax = axes[1, 1]
    z = expand_z(np.array(maps["G_l_mean_e"]))
    ax, zplot = colormap(x,y, z, ax= ax)
    ax.set_yticks([], []);
    ax.set_title(r"$\bar G_l$ error")
    ax.set_xlabel("Return interval")
    
    return axes

