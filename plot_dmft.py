#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 12:40:38 2018

Clean code for plotting the results of dmft with gam.

@author: froy
"""

from dmft_stationary import fullCav_withGam

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
    

def check_convergence(iterFinalAutocorA, iterFinalMeanA, iterNormDifA, paramL, iterIntegratedChiAA=1):
    [mu, sig, gam, lamda, tMax, threshold, soft, tau, nIte] = paramL
    titleParam = (r'$(\mu, \sigma, \gamma, \lambda) =( %.1f, %.1f, %.1f, %.0e)$' %(mu,sig, gam, lamda))
        
    _, statK, statMean, statSquare, _ = fullCav_withGam(mu,sig,gam)

    sizeIte = iterFinalMeanA.shape[0]
    iterLastPlot=plt.figure()#figsize=(11.0, 8.0))
    iterLastAx = iterLastPlot.add_subplot(111)
    iterLastAx.plot(iterFinalMeanA, label=r'final mean $m(t_f)$')
    iterLastAx.plot([0,sizeIte-1],[statMean,statMean],'--', label='m static cavity')
    iterLastAx.plot(iterFinalAutocorA, label=r'final same-time correlation $C(t_f,t_f)$')
    iterLastAx.plot([0,sizeIte-1],[statSquare,statSquare],'--', label='q static cavity')
    if np.size(iterIntegratedChiAA) !=1:
        iterLastAx.plot(iterIntegratedChiAA, label='integrated response $\chi_{int}$')    
        iterLastAx.plot([0,sizeIte-1],[statK,statK],'--', label=r'$\chi_{int}$ static cavity')
    iterLastAx.set_xlabel('Number of iterations')
    iterLastAx.set_ylabel('Observables')
    iterLastAx.set_title(titleParam)
    plt.axhline(y=1e-9, color='k', alpha=0.3)
    plt.axvline(x=0, color='k', alpha=0.3)
    iterLastAx.legend(loc=1)
    plt.tight_layout()
    
    iterNorm=plt.figure()
    plt.plot(iterNormDifA)
    plt.xlabel('Number of iterations')
    plt.ylabel('Amplitude of each iteration')
    plt.yscale('log')
    plt.title(titleParam)
    plt.axhline(y=1e-9, color='k', alpha=0.3)
    plt.axvline(x=0, color='k', alpha=0.3)
    plt.tight_layout()
    
    return (iterLastPlot,iterNorm)

def plot_observables(corAA,meanA,paramL, chiAA=1):
    [mu, sig, gam, lamda, tMax, threshold, soft, tau, nIte] = paramL

    nTime= int(tMax/tau)
    titleParam = (r'$(\mu, \sigma, \gamma, \lambda) =( %.1f, %.1f, %.1f, %.0e)$' %(mu,sig, gam, lamda))
    
    timeA = np.linspace(0,(nTime-1)*tau,nTime)
    
    # 3d plot of the last correlator
    cor3dPlot = plt.figure()#figsize=(15.0, 8.0))
    ax = cor3dPlot.gca(projection='3d')
    
    # Make data
    X = np.copy( timeA )
    Y = np.copy( timeA )
    X, Y = np.meshgrid(X, Y)
    Z = corAA.T
    ax.set_xlabel('t')
    ax.set_ylabel('t\'')
    ax.set_zlabel('Correlator C(t,t\')')
        
    # Plot the surface.
    ax.plot_surface(X, Y, Z, cmap=cm.plasma,
                           linewidth=0)
    cor3dPlot.tight_layout()
    ax.set_title(titleParam)
    
    # Plot the mean and square
    _, statK, statMean, statSquare, _ = fullCav_withGam(mu,sig,gam)
    
    obsPlot=plt.figure()#figsize=(15.0, 8.0))
    obsAx = obsPlot.add_subplot(111)
    obsAx.set_title(titleParam)
    obsAx.set_ylabel('Observables')
    obsAx.set_xlabel('Time')
    obsAx.plot(timeA,meanA,label='mean m(t) DMFT')
    obsAx.plot([0,tau*nTime],[statMean,statMean],'--', label='m static cavity')
    obsAx.plot(timeA,np.diag(corAA),label='same-time correlation C(t,t) DMFT')
    obsAx.plot([0,tau*nTime],[statSquare,statSquare],'--', label='q static cavity')
    if np.size(chiAA) != 1:
        obsAx.plot(timeA,[tau*chiAA[:t,int(nTime/8)].sum() for t in range(nTime)],label=r'integrated response $\chi_{int}$ DMFT')
        obsAx.plot([0,tau*nTime],[statK,statK],'--', label=r'$\chi_{int}$ static cavity')


    obsAx.grid()
    obsAx.legend(loc=1)
    
#    # Plot the 3d response
    if np.size(chiAA) != 1:
        chi3dPlot = plt.figure()   #figsize=(11.0, 8.0) for bigger
        axK = chi3dPlot.gca(projection='3d')
        
        X = np.copy( timeA )
        Y = np.copy( timeA )
        X, Y = np.meshgrid(X, Y)
        axK.set_xlabel(r'$t$')
        axK.set_ylabel(r"$t'$")
        axK.set_zlabel(r"Response $\chi(t,t')$")
    #        axK.set_title(titleParam)
        axK.set_title(titleParam)
        axK.plot_surface(X, Y, chiAA.T, cmap=cm.coolwarm,
                               linewidth=0)

    else:
        chi3dPlot = False
    
    return cor3dPlot, obsPlot, chi3dPlot


##############################################################################################################
