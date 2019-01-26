#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 16:37:59 2019

@author: froy
"""

from dmft_withoutGamma import writeArraysDMFT_withoutGam
from dmft_withGamma import writeArraysDMFT_withGam
from plot_dmft import check_convergence, plot_observables
import os
import numpy as np
import matplotlib.pyplot as plt

########################################################################################################################

# Example of solutions without gamma (ie no response function)

print('\nDMFT solver, with gamma=0, in the chaotic phase')

mu = 10
sig = 2
gam = 0
lamda = 1e-10
soft = 0.3 	          	# coefficient for the loop
tau = 0.1	                	# discrete time step
tMax = 40           
threshold = 1e-9           # threshold for convergence in correlators

nIteL, nEtaL = [20, 10, 10], [1e1, 1e2, 2e2]

nIte = np.sum(nIteL)

paramL = [mu, sig, gam, lamda, tMax, soft, tau, nIte, threshold]

currentPath = os.path.dirname(os.path.abspath(__file__))
pathToSave = currentPath + '/solutionExamples'

os.makedirs(pathToSave)

writeArraysDMFT_withoutGam(paramL, nIteL, nEtaL, pathToSave)

pathForThisInstance = pathToSave + '/mu%.1f_sig%.2f_gam%.2f_lam%.0e_tMax%.0e' %(mu, sig, gam, lamda, tMax )
os.chdir( pathForThisInstance + '/arrays' )

corAA = np.load('corAA.npy')
meanA = np.load('meanA.npy')
iterFinalAutocorA = np.load('iterFinalAutocorA.npy')
iterFinalMeanA = np.load('iterFinalMeanA.npy')
iterNormDifA = np.load('iterNormDifA.npy')
paramA = np.load('paramA.npy')

os.makedirs( pathForThisInstance + '/plots' )
os.chdir( pathForThisInstance + '/plots' )

print('Convergence check plots')
iterLastPlot,iterNorm = check_convergence(iterFinalAutocorA, iterFinalMeanA, 
    iterNormDifA, paramA)
iterLastPlot.savefig('observables_iterations.png')
iterNorm.savefig('iterations_step_amplitude.png')

plt.show(block=False)
input("Press Enter to continue...")
plt.close()

print('Plot the results')
cor3dPlot, obsPlot, _ = plot_observables(corAA,meanA,paramA)
cor3dPlot.savefig('3d_correlator.png')
obsPlot.savefig('observables_solution.png')

plt.show(block=False)
input("Press Enter to continue...")
plt.close()

########################################################################################################################

# Example of solutions with gamma (ie with response function)

print('\nDMFT solver, with gamma!=0, in the 1eq phase')

mu = 10
sig = 1
gam = -0.5
lamda = 1e-10
soft = 0.3 	          	# coefficient for the loop
tau = 0.1	                	# discrete time step
tMax = 20           
threshold = 1e-9           # threshold for convergence in correlators

nIteL, nEtaL = [20, 10, 10], [1e1, 1e2, 2e2]

nIte = np.sum(nIteL)

paramL = [mu, sig, gam, lamda, tMax, soft, tau, nIte, threshold]

writeArraysDMFT_withGam(paramL, nIteL, nEtaL, pathToSave)

pathForThisInstance = pathToSave + '/mu%.1f_sig%.2f_gam%.2f_lam%.0e_tMax%.0e' %(mu, sig, gam, lamda, tMax )
os.chdir( pathForThisInstance + '/arrays' )

corAA = np.load('corAA.npy')
meanA = np.load('meanA.npy')
chiAA = np.load('chiAA.npy')
iterFinalAutocorA = np.load('iterFinalAutocorA.npy')
iterFinalMeanA = np.load('iterFinalMeanA.npy')
iterIntegratedChiA = np.load('iterIntegratedChiA.npy')
iterNormDifA = np.load('iterNormDifA.npy')
paramA = np.load('paramA.npy')

os.makedirs( pathForThisInstance + '/plots' )
os.chdir( pathForThisInstance + '/plots' )



print('Convergence check plots')
iterLastPlot,iterNorm = check_convergence(iterFinalAutocorA, iterFinalMeanA, 
    iterNormDifA, paramA, iterIntegratedChiA)
iterLastPlot.savefig('observables_iterations.png')
iterNorm.savefig('iterations_step_amplitude.png')

plt.show(block=False)
input("Press Enter to continue...")
plt.close()

print('Plot the results')
cor3dPlot, obsPlot, chi3dPlot = plot_observables(corAA,meanA,paramA, chiAA)
cor3dPlot.savefig('3d_correlator.png')
obsPlot.savefig('observables_solution.png')
chi3dPlot.savefig('3d_response.png')

plt.show(block=False)
input("Press Enter to finish...")
plt.close()

print('\nAll plots are saved in folders solutionExamples')
