#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 28th, 2018

DMFT for the random Lotka-Volterra model, with perfect asymmetry gamma=0. In this case,
I don't have to compute the response function chi along the iterations.

Contains the functions:
    iterateDMFT_withoutGam: iterates once the correlator and the mean, averaging over a given number nEta of trajectories
    loopDMFT_withoutGam: performs many iterations, with a convergence criterion to stop the iterations
    writeArraysDMFT_withoutGam: does everything and stores in folder

@author: froy
"""

import numpy as np
import os
from datetime import datetime
import time

########################################################################################################################

# Function to iterate once the correlator and the mean, averaging over a given number nEta of trajectories

def iterateDMFT_withoutGam(corAA,meanA,paramL,nEta):
    
    # The arguments of the function:
        # corAA, meanA: the correlation and mean at the beginning of this iteration
        # paramL: list of the scalar parameters for the model and the algorithm (details below)
        # nEta: number of trajectories to average upon
            
    # The function returns:
        # newCorAA, newMeanA: the new correlation and mean after this iteration
        # stop_because_UG: boolean variable, True if there was a divergence in this iteration
            # This is to check that all went well in the iteration. It should be False.
        # maxLogAb, minLogAb: extremal scalar of the log of the populations encountered in this iteration
            # These are to check that all went well in the iteration.
        # eigenValues: the array of the eigenValues of the correlation at the beginning of the iteration
            # These are to check that all went well in the iteration. They should all be positive.
        
    [mu, sig, gam, lamda, tMax, soft, tau, nIte, threshold] = paramL
    nEta, nTime = int(nEta), int( tMax/tau )
    # Define the parameters of the DMFT
        # mu, sig, gam and lamda: parameters of the rLV model
        # tMax: upper time boundary wanted for the DMFT solution
        # tau: discrete time step
        # soft: strength of the merging of observables when iterating 
        # nIte: maximal number of iterations (not needed in this function)
        # threshold: threshold under which the iterations are considered to have converged
    
    # Declare a threshold value over which I'll consider that the system has exploded (UG)
    threshold_abundance_for_UG, stop_because_UG = 1e2, False    # Initialize the error divergence as False
    
    # Sample the gaussian paths, starting with iid variables, and mixing them in the proper basis
    eigenValues, orthogonalBasis = np.linalg.eigh(corAA)
    # Check for eigenValues properties of the correlator. They should always be positive.
    positiveEigenValues = eigenValues*( eigenValues>0 )  # Impose positive eigenvalues
    tempGaussian = np.random.normal(0,1,(nTime,nEta))
    pathAA = orthogonalBasis.dot( np.diag(np.sqrt(positiveEigenValues)).dot(tempGaussian) )
    
    
    # Now enters the model
    
    # Initial conditions for all trajectories, depends on the model
    logTrajAA = np.zeros((nTime,nEta))
    if lamda != 0:      # be careful to prevent the initial explosion, imposing no IC below lamda
        logLamdaA =  np.log( lamda*np.ones(nEta) )
        logTempA =  np.maximum( np.log( np.random.uniform(0,1,nEta) ), logLamdaA )
    else:
        logTempA =  np.log( np.random.uniform(0,1,nEta) )
    logTrajAA[0,:] = np.copy(logTempA)
    
    # Temporal log integration, depends on the model. I use a basic Euler's scheme.
    for i in range(nTime-1):
        # check for divergence, in the UG phase
        if logTempA.max() > np.log( threshold_abundance_for_UG ):
            stop_because_UG = True
            return 1,1, stop_because_UG, np.log( threshold_abundance_for_UG ), logTempA.min(), eigenValues
        trajA = np.exp( logTempA )
        if lamda == 0:
            logDerivativeA = 1 - mu*meanA[i] - trajA - sig*pathAA[i,:]
        else:
            logDerivativeA = 1 - mu*meanA[i] - trajA - sig*pathAA[i,:] + np.exp( logLamdaA - logTempA )
        logTempA += tau*logDerivativeA
        logTrajAA[i+1,:]  =  np.copy( logTempA )
        
    # Update softly the observables
    trajAA = np.exp( logTrajAA )
    newCorAA = (1-soft)*corAA + soft*trajAA.dot(trajAA.T)/nEta   
    newMeanA = (1-soft)*meanA + soft*trajAA.mean(axis = 1)
    
    # Keep track of the max and min log abundances among trajectories
    maxLogAb, minLogAb = logTrajAA.max(), logTrajAA.min()
    
    return newCorAA, newMeanA, stop_because_UG, maxLogAb, minLogAb, eigenValues


########################################################################################################################

# Perform many iterations, with a convergence criterion to stop the iterations.
# It can either start with a given correlator, or the constant sationary limit.
    
def loopDMFT_withoutGam(paramL, nIteL, nEtaL, pathForLogFile, initialCorAA = 1, initialMeanA = 1):
    
    # The arguments of the function:
        # paramL: list of the scalar parameters for the model and the algorithm (details below)
        # nIteL, nEtaL: the iteration process. 
            # For example nIteL=[20,10] and nEtaL=[1e3,1e4] will launch
            # 20 iterations with 1e3 trajectories, then 10 iterations with 1e4 trajectories.
        # pathForLogFile: the path on the computer where to write the console output
        # initialCorAA, initialMeanA: the initial guesses for the correlation matrix
            # and the average vector. If not specified, they will be taken as detailed below
            
    # The function returns:
        # finalAutocorA: array of C(tMax,tMax) for each iteration
        # finalMeanA: array of m(tMax) for each iteration
        # normDifA: array of the size of each iteration step
        # corAA, meanA: the numerical solutions of C and m after all iterations
    
    [mu, sig, gam, lamda, tMax, soft, tau, nIte, threshold] = paramL
    nTime = int( tMax/tau )         # nTime: the number of discrete time steps
    # Define the parameters of the DMFT
        # mu, sig, gam and lamda: parameters of the rLV model
        # tMax: upper time boundary wanted for the DMFT solution
        # tau: discrete time step
        # soft: strength of the merging of observables when iterating 
        # nIte: maximal number of iterations (not needed in this function)
        # threshold: threshold under which the iterations are considered to have converged
    
    # I keep track of C(tMax,tMax) and m(tMax) over iterations
    finalAutocorA, finalMeanA = np.zeros(nIte+1), np.zeros(nIte+1)
    
    # this duplicates outputs from the console into the log file
    def printLog(*args, **kwargs):
        print(*args, **kwargs)
        with open(pathForLogFile, 'a') as file:
            print(*args, **kwargs, file=file)
        
    # Keep track of the needed time for each iteration
    start = time.time()
    chrono = time.time()
    
    parameterStr = ( '\nParameters:' + '\n' +
          'mu=%i, sig=%.1f, gam=%.1f, lamda=%.0e, tMax=%.0e | soft=%.1f, tau=%.1f, nIte=%i, threshold_norm=%.0e'
          %(mu, sig, gam, lamda, tMax, soft, tau, nIte, threshold) )            # Parameters string, to be printed
    iterationsStr =( '\n' + 'Number of iterations: ' + str(nIteL) + 
          '\n' + 'Corresponding number of paths: ' + str(nEtaL) )                # Iteration strategy string, to be printed
    printLog( '\n', datetime.now(), '\n' + parameterStr + '\n' + iterationsStr + '\n')
    
    # Initialize with a white noise correlation and zero mean, if IC for the correlation and mean are not given
    if type(initialCorAA) == int:
        corAA  =  np.diag( np.ones(nTime)  )
    else:
        corAA  =  np.copy(initialCorAA)
    if type(initialMeanA) == int:
        meanA = np.zeros(nTime)
    else:
        meanA  =  np.copy(initialMeanA)
        
    # Declare the array that will store the size of the iterations steps
    normDifA  =  np.zeros(nIte)
    
    # Record the observable C(tMax,tMax) and m(tMax) of the initial guess
    finalAutocorA[0], finalMeanA[0] = corAA[-1,-1], meanA[-1]
            
    # Iterate the update of observables
    comptIte=0                      # count the # of iterations
    nIteSteps=np.size(nIteL)        # number of bundles of iterations
    for i in range(nIteSteps):      # loops over the iteration strategy
        nIte_fixEta, nEta = int( nIteL[i] ), nEtaL[i]       # # of iterations with a given # of trajectories
        for j in range(nIte_fixEta):                        # loop over the requested iterations with the given # of trajectories
            
            # Perform an iteration of the correlator and the mean, calls the function "iterateDMFT_withoutGam"
            (tempCorAA,tempMeanA,stop_because_UG, maxLogAb, minLogAb, eigenValues)  =  iterateDMFT_withoutGam(corAA, meanA, paramL, nEta)
            
            # If there was a diverging trajectory, record it, and stop the algorithm
            if stop_because_UG:
                printLog( '\nStopped after %i iterations because of divergence (%i seconds).' 
                      %(comptIte, time.time()-start) )                 
                return (finalAutocorA[0:comptIte], finalMeanA[0:comptIte], normDifA[0:comptIte-1], corAA, meanA)
            
            # Check the properties of the correlator eigenvalues. They should all be positive for proper sampling.
            if (eigenValues<1e-10).sum() > 0:
                printLog( 'Eigenvalues of the correlator: %i negative ones and %i positive below 1e-10 /%i.'
                      %( (eigenValues<0).sum(), (eigenValues<1e-10).sum() - (eigenValues<0).sum() , nTime ) )
            
            comptIte += 1
            
            # Record the observable C(tMax,tMax) and m(tMax) of the new iteration
            finalAutocorA[comptIte], finalMeanA[comptIte] = tempCorAA[-1,-1], tempMeanA[-1]
            
            # Evaluate the amplitude of the iteration step, and store into normDifA.
            # Here I use the Frobenius norm of the correlator difference.
            normDif  =  np.trace( (tempCorAA-corAA).dot( (tempCorAA-corAA).T )   ) / nTime**2
            normDifA[comptIte-1]  =  np.copy( normDif ) 
            
            # If the step is small enough, stop the loop
            if normDif<threshold and comptIte>4:
                printLog('\nConverged after %i iterations, with threshold %.1e (%i seconds).' 
                      %(comptIte,threshold, time.time()-start))
                return (finalAutocorA[0:comptIte+1], finalMeanA[0:comptIte+1], normDifA[0:comptIte], tempCorAA, tempMeanA)
            
            # Else, copy the new correlation and mean, print the characteristics of the iteration, and continue new iterations
            corAA = np.copy( tempCorAA )
            meanA = np.copy( tempMeanA )
            tempTime=time.time()
            printLog('Iteration %.2i, nEta = %.0e, %i seconds, stepAmplitude %.1e, log10Abundances between %i and %i.'
                  %(comptIte, nEta, tempTime-chrono, normDif, minLogAb/np.log(10), maxLogAb/np.log(10)) )
    
    # The algorithm reaches this point if after all iteration strategy, it didn't converge below the threshold size of step.
    printLog('\nDid not converge under threshold %.1e after %i iterations (%i seconds). \n'
          %(threshold,comptIte,time.time()-start)) 

    return (finalAutocorA, finalMeanA, normDifA, corAA, meanA)
########################################################################################################################

# Do everything and store in folder

def writeArraysDMFT_withoutGam(paramL, nIteL, nEtaL, pathToSave, initialCorAA = 1, initialMeanA = 1):
    
    # The arguments of the function:
        # paramL: list of the scalar parameters for the model and the algorithm (details below)
        # nIteL, nEtaL: the iteration process. 
            # For example nIteL=[20,10] and nEtaL=[1e3,1e4] will launch
            # 20 iterations with 1e3 trajectories, then 10 iterations with 1e4 trajectories.
        # pathToSave: the path on the computer where to store the results and log file
        # initialCorAA, initialMeanA: the initial guesses for the correlation matrix
            # and the average vector. If not specified, they will be taken as detailed
            # in the function "loopDMFT_withoutGam"
            
    # The function returns: nothing.
    
    [mu, sig, gam, lamda, tMax, soft, tau, nIte, threshold] = paramL
    # Define the parameters of the DMFT
        # mu, sig, gam and lamda: parameters of the rLV model
        # tMax: upper time boundary wanted for the DMFT solution
        # tau: discrete time step
        # soft: strength of the merging of observables when iterating 
        # nIte: maximal number of iterations (not needed in this function)
        # threshold: threshold under which the iterations are considered to have converged
    
    # Create the folder where to save the results
    pathFolder = '/mu%.1f_sig%.2f_gam%.2f_lam%.0e_tMax%.0e' %(mu, sig, gam, lamda, tMax )
    os.makedirs( pathToSave + pathFolder + '/arrays' )
    
    # Add the console output to a log file as well
    pathForLogFile = pathToSave + pathFolder + '/logfile.log'
    
    # Perform the iterations, calls the function "loopDMFT_withoutGam")
    (iterFinalAutocorA, iterFinalMeanA, iterNormDifA, corAA, meanA) = loopDMFT_withoutGam(paramL, nIteL, 
        nEtaL, pathForLogFile, initialCorAA=initialCorAA, initialMeanA=initialMeanA)
    
    # Write the arrays in the folder
    paramForSaveA = np.array( [ mu, sig, gam, lamda, tMax, threshold, soft, tau, nIte ] )
    os.chdir( pathToSave + pathFolder + '/arrays' )
    np.save('corAA',corAA)
    np.save('meanA',meanA)
    np.save('iterFinalAutocorA',iterFinalAutocorA)
    np.save('iterFinalMeanA', iterFinalMeanA)
    np.save('iterNormDifA',iterNormDifA)
    np.save('paramA',paramForSaveA)
    np.save('nIteL',nIteL)
    np.save('nEtaL',nEtaL)
    
    return 0    
