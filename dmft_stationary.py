#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 18:46:53 2018

Stationary cavity analysis. Also enclose the localization of the expected transitions:
    - the chaotic transition (exact)
    - the Unbounded Growth transition (approximate)
    - the possible Response Transition, might be aging with immigration (doubts about that)

@author: froy
"""

import numpy as np
from scipy.stats import norm
import scipy.optimize as opt

###################################################################
# Useful functions
def w2(x):
    return (x**2+1)*norm.cdf(x)+x*norm.pdf(x)
def w1(x):
    return x*norm.cdf(x) + norm.pdf(x)
def w0(x):
    return norm.cdf(x)
###################################################################
# GAMMA = 0
def findDelta_zeroGam(sig):
    def findRootOf(delt):
        return w2(delt)-sig**(-2)    
    return opt.newton(findRootOf,0)
def fullCav_zeroGam(mu,sig):
    delt = findDelta_zeroGam(sig)
    phi_K = w0(delt)
    m = w1(delt) / (mu*w1(delt) + delt)
    q = ( 1/sig / (mu*w1(delt) + delt) )**2
    return phi_K, m, q, delt 

# Unbounded growth transition: when <N> and <N2> diverges
def UG_delta_zeroGam(mu):
    def findRootOf(delta):
        return delta+mu*w1(delta)    
    return opt.newton(findRootOf,0)
def UG_sigma_zeroGam(mu):
    return 1/np.sqrt(w2(UG_delta_zeroGam(mu)))
###################################################################
# GAMMA != 0
def findDelta_withGam(sig, gam):
    def findRootOf(x):
        # I went to log space in order to have better numerical convergence. Gam=-1 seems problematic.
        correctedGam = np.maximum(gam, -1+1e-8)
        return 2*np.log(sig) + 2*np.log( np.abs( 1 + correctedGam * w0(x)/w2(x) ) ) + np.log( w2(x) )
    return opt.newton(findRootOf,0)
def fullCav_withGam(mu,sig,gam):
    delt = findDelta_withGam(sig, gam)
    m = w1(delt)/(  delt*sig**2*( w2(delt)+gam*w0(delt) ) + mu*w1(delt)  )
    q = ( sig*( w2(delt)+gam*w0(delt) ) / ( delt*sig**2*(w2(delt)+gam*w0(delt)) + mu*w1(delt) ) )**2
    if gam==0:
        K = w0(delt)
    else:
        K = (1/sig**2 - w2(delt) )/gam - w0(delt)
    phi = w0(delt)
    return (phi, K, m, q, delt)

###################################################################
#  UG line: when <N> and <N2> diverges
def UG_delta_withGam(mu,sig):
    def findRootOf(x):
        # I went to log space in order to have better numerical convergence.
        if mu==0:
            return x
        else:
            return 2*np.log( sig/mu ) + np.log( x**2*w2(x) ) -2*np.log( w1(x) ) 
    return opt.newton(findRootOf,1)
def UG_gam_withGam(mu,sig):
    if mu == 0:
        return np.sqrt(2)/sig -1
    else:
        delt = UG_delta_withGam(mu,sig)
        return -delt/mu/w1(delt)*w2(delt)/w0(delt) 
    return 0

def UG_mu_withGam(sig, gam):
    delt = findDelta_withGam(sig,gam)
    return -sig**2*delt*(w2(delt)+gam*w0(delt))/w1(delt)

# Inverting the implicit relation is not so good. Better stay with UG_mu
def UG_sig_withGam(mu, gam):
    def findRootOf(x):
        return UG_mu_withGam(x,gam)-mu
    return opt.newton(findRootOf,10, maxiter=400)


###################################################################
#  Possible aging line: where the response function behaviour might change
def find_sigma_possibleAgingTr(gam):
    def findRootOf(delt):
        return w2(delt)-gam*w0(delt)    
    return 1/4/w2( opt.newton(findRootOf,0) )    
