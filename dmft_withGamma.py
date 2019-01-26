#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 28th, 2018

DMFT for the random Lotka-Volterra model, with any symmetry parameter gamma. In this case,
I do have to compute the response function chi along the iterations.

Contains the functions:
    integrate_chiAA: Function to integrate the response function for each trajectory, and average over trajectories. 
    iterateDMFT_withGam: iterates once the correlator, mean and response, averaging over a given number nEta of trajectories
    loopDMFT_withGam: performs many iterations, with a convergence criterion to stop the iterations
    writeArraysDMFT_withGam: does everything and stores in folder

@author: froy
"""

import numpy as np
import os
from datetime import datetime
import time


########################################################################################################################

# Function to integrate the response function for each trajectory, and average over trajectories. 
# Depends on the model.

def integrate_chiAA( trajAA, logDerivAA, chiAA, sig, gam, tau ):
    
    # The arguments of the function:
        # trajAA: nEta trajectories of the populations
        # logDerivAA: an intermediate computation that I can reuse to compute the response
        # chiAA: response function before iteration
        # sig, gam: parameters of the rLV model
        # tau: discrete time step
        
    # The function returns:
        # the averaged response function over these trajectories

    nTime, nEta = trajAA.shape
    chiTrajAAA = np.zeros( (nTime, nTime, nEta) )
    for t2 in range(nTime-1):
        tempChiA = np.copy( trajAA[ t2, : ] )
        chiTrajAAA[ t2+1, t2, : ] = np.copy( tempChiA )
        t1 = t2+1
        while t1<nTime-1:
            derivMultiplicativeA = logDerivAA[t1, :] - trajAA[ t1, : ]
            deriveAdditiveA = trajAA[ t1, : ]* gam*sig**2*tau*chiAA[t1,t2:t1].dot( chiTrajAAA[ t2:t1, t2, :] )
            tempChiA = np.copy( tempChiA*np.exp( tau*derivMultiplicativeA ) + tau*deriveAdditiveA )
            chiTrajAAA[t1+1,t2,:] = np.copy( tempChiA )
            t1+=1
        
    return chiTrajAAA.sum( axis=2 )


########################################################################################################################
    
# Function to iterate once the correlator and the mean, averaging over a given number nEta of trajectories

def iterateDMFT_withGam(corAA,meanA,chiAA,paramL,nEta):
    
    # The arguments of the function:
        # corAA, meanA, chiAA: the correlation, mean and response at the beginning of this iteration
        # paramL: list of the scalar parameters for the model and the algorithm (details below)
        # nEta: number of trajectories to average upon
            
    # The function returns:
        # newCorAA, newMeanA, newChiAA: the new correlation, mean and response after this iteration
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
    threshold_abundance_for_UG, stop_because_UG = 1e3, False    # Initialize the error divergence as False
    
    # Sample the gaussian paths, starting with iid variables, and mixing them in the proper basis
    eigenValues, orthogonalBasis = np.linalg.eigh(corAA)
    # Check for eigenValues properties of the correlator. They should always be positive.
    positiveEigenValues = eigenValues*( eigenValues>0 )  #impose positive eigenvalues
    tempGaussian = np.random.normal(0,1,(nTime,nEta))
    pathAA = orthogonalBasis.dot( np.diag(np.sqrt(positiveEigenValues)).dot(tempGaussian) )

    # Now enters the model
    
    # trajAA will contain each trajectory
    # I also keep track of some part of the derivative to use it again in the response computation
    trajAA = np.zeros((nTime,nEta))
    logDeriv_nonLam_AA = np.zeros( (nTime-1,nEta) )
    
    # Initial conditions
    tempA = np.random.uniform( 0, 1, nEta )
    if lamda != 0:
        tempA = np.maximum( np.random.uniform( 0, 1, nEta ), lamda*np.ones(nEta) )
    trajAA[0, :] = np.copy( tempA )
    
    # Temporal semi-log integration
    for i in range(nTime-1):
        # check for divergence
        if tempA.max() >  threshold_abundance_for_UG :
            stop_because_UG = True
            return 1,1,1, stop_because_UG, threshold_abundance_for_UG, tempA.min(), eigenValues
        logDerivA = ( 1 - mu*meanA[i] - tempA - sig*pathAA[i,:] 
                + gam*sig**2*chiAA[i,:i].dot( trajAA[:i,:] )*tau )
        logDeriv_nonLam_AA[i,:] = np.copy( logDerivA  )
        tempA = tempA*np.exp( tau* ( logDerivA ) ) + tau*lamda
        trajAA[i+1,:]  =  np.copy( tempA )
        
    # Update softly the observables
    newCorAA = (1-soft)*corAA + soft*trajAA.dot(trajAA.T)/nEta   
    newMeanA = (1-soft)*meanA + soft*trajAA.mean(axis = 1)
    
    # Response function: for memory issues sometimes I need to bundle together the paths for the response function
    tempChiAA = np.zeros( (nTime, nTime) )
    nEtaMaxPerBundle = int(1e9/nTime**2)
    nBundles = int( nEta / nEtaMaxPerBundle )   # nEta = nBundles*nEtaMaxPerBundle + remainingPaths
    for i in range(nBundles):
        tempChiAA += integrate_chiAA( trajAA[:, i*nEtaMaxPerBundle:(i+1)*nEtaMaxPerBundle ], 
                                           logDeriv_nonLam_AA[:, i*nEtaMaxPerBundle:(i+1)*nEtaMaxPerBundle ], 
                                           chiAA, sig, gam, tau ) / nEta
    tempChiAA += integrate_chiAA( trajAA[:, nBundles*nEtaMaxPerBundle: ], 
                                           logDeriv_nonLam_AA[:, nBundles*nEtaMaxPerBundle: ], 
                                           chiAA, sig, gam, tau ) / nEta
    newChiAA = (1-soft)*chiAA + soft*tempChiAA

    # Keep track of the max and min abundances among trajectories
    maxAb, minAb = trajAA.max(), trajAA.min()
    
    return newCorAA, newMeanA, newChiAA, stop_because_UG, maxAb, minAb, eigenValues
    
########################################################################################################################

# Perform many iterations, with a convergence criterion to stop the iterations.
# It can either start with a given correlator, or the constant sationary limit.

def loopDMFT_withGam(paramL, nIteL, nEtaL, pathForLogFile,
                        initialCorAA = 1, initialMeanA = 1, initialChiAA =1):
    
    # The arguments of the function:
        # paramL: list of the scalar parameters for the model and the algorithm (details below)
        # nIteL, nEtaL: the iteration process. 
            # For example nIteL=[20,10] and nEtaL=[1e3,1e4] will launch
            # 20 iterations with 1e3 trajectories, then 10 iterations with 1e4 trajectories.
        # pathForLogFile: the path on the computer where to write the console output
        # initialCorAA, initialMeanA, initialChiAA: the initial guesses for the correlation matrix,
            # the average vector and the response matrix. If not specified, they will be taken as detailed below
            
    # The function returns:
        # finalAutocorA: array of C(tMax,tMax) for each iteration
        # finalMeanA: array of m(tMax) for each iteration
        # integratedChiA: 
        # normDifA: array of the size of each iteration step
        # corAA, meanA, chiAA: the numerical solutions of C, m and chi after all iterations

    [mu, sig, gam, lamda, tMax, soft, tau, nIte, threshold] = paramL
    nTime = int( tMax/tau )         # nTime: the number of discrete time steps
    # Define the parameters of the DMFT
        # mu, sig, gam and lamda: parameters of the rLV model
        # tMax: upper time boundary wanted for the DMFT solution
        # tau: discrete time step
        # soft: strength of the merging of observables when iterating 
        # nIte: maximal number of iterations (not needed in this function)
        # threshold: threshold under which the iterations are considered to have converged
    
    # I keep track of C(tMax,tMax), m(tMax) and chi_int over iterations
    finalAutocorA, finalMeanA, integratedChiA = np.zeros(nIte+1), np.zeros(nIte+1), np.zeros(nIte+1)
    
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
    
    # Initialize with a white noise correlation, zero mean and a diagonal response
    # if IC for the correlation and mean are not given.
    if type(initialCorAA) == int:
        corAA  =  np.diag( np.ones(nTime)  )
    else:
        corAA  =  np.copy(initialCorAA)
    if type(initialMeanA) == int:
        meanA = np.zeros(nTime)
    else:
        meanA  =  np.copy(initialMeanA)
    if type(initialChiAA) == int:
        chiAA = np.diag( np.ones(nTime)  )
    else:
        chiAA  =  np.copy(initialChiAA)
        
    # Declare the array that will store the size of the iterations steps
    normDifA  =  np.zeros(nIte)
    
    # Record the observable C(tMax,tMax), m(tMax) and chi_int of the initial guess
    finalAutocorA[0], finalMeanA[0], integratedChiA[0] = corAA[-1,-1], meanA[-1], chiAA[:,int(nTime/4)].sum()*tau
            
    # Iterate the update of observables
    comptIte=0                      # count the # of iterations
    nIteSteps=np.size(nIteL)        # number of bundles of iterations
    for i in range(nIteSteps):      # loops over the iteration strategy
        nIte_fixEta, nEta = int( nIteL[i] ), nEtaL[i]       # # of iterations with a given # of trajectories
        for j in range(nIte_fixEta):                        # loop over the requested iterations with the given # of trajectories
            
            # Perform an iteration of the correlator, mean and response, calls the function "iterateDMFT_withGam"
            (tempCorAA,tempMeanA,tempChiAA,stop_because_UG, maxAb,
             minAb, eigenValues)  =  iterateDMFT_withGam(corAA, meanA, chiAA, paramL, nEta)
            
            # If there was a diverging trajectory, record it, and stop the algorithm
            if stop_because_UG:
                printLog( '\nStopped after %i iterations because of divergence (%i seconds).' 
                      %(comptIte, time.time()-start) )                 
                return (finalAutocorA[0:comptIte], finalMeanA[0:comptIte],
                        integratedChiA[0:comptIte], normDifA[0:comptIte-1], corAA, meanA, chiAA)
            
            # Check the properties of the correlator eigenvalues. They should all be positive for proper sampling.
            if (eigenValues<1e-10).sum() > 0:
                printLog( 'Eigenvalues of the correlator: %i negative ones and %i positive below 1e-10 /%i.'
                      %( (eigenValues<0).sum(), (eigenValues<1e-10).sum() - (eigenValues<0).sum() , nTime ) )
            
            comptIte += 1
            
            # Record the observable C(tMax,tMax), m(tMax) and chi_int of the new iteration

            (finalAutocorA[comptIte], finalMeanA[comptIte], 
             integratedChiA[comptIte]) = tempCorAA[-1,-1], tempMeanA[-1], tempChiAA[:,int(nTime/4)].sum()*tau
            
            # Evaluate the amplitude of the iteration step, and store into normDifA.
            # Here I use the Frobenius norm of the correlator difference.
            normDif  =  np.trace( (tempCorAA-corAA).dot( (tempCorAA-corAA).T )   ) / nTime**2
            normDifA[comptIte-1]  =  np.copy( normDif ) 
            
            # If the step is small enough, stop the loop
            if normDif<threshold and comptIte>4:
                printLog('\nConverged after %i iterations, with threshold %.1e (%i seconds).' 
                      %(comptIte,threshold, time.time()-start))
                return (finalAutocorA[0:comptIte+1], finalMeanA[0:comptIte+1],
                        integratedChiA[0:comptIte+1], normDifA[0:comptIte], tempCorAA, tempMeanA, tempChiAA)
            
            # Else, copy the new correlation and mean, print the characteristics of the iteration, and continue new iterations
            corAA = np.copy( tempCorAA )
            meanA = np.copy( tempMeanA )
            chiAA = np.copy( tempChiAA )
            tempTime=time.time()
            printLog('Iteration %.2i, nEta = %.0e, %i seconds, stepAmplitude %.1e, abundances between %.1e and %.1e.'
                  %(comptIte, nEta, tempTime-chrono, normDif, minAb, maxAb) )

    # The algorithm reaches this point if after all iteration strategy, it didn't converge below the threshold size of step.
    printLog('\nDid not converge under threshold %.1e after %i iterations (%i seconds). \n'
          %(threshold,comptIte,time.time()-start)) 

    return (finalAutocorA, finalMeanA, integratedChiA, normDifA, corAA, meanA, chiAA)


########################################################################################################################

# Do everything and store in folder

def writeArraysDMFT_withGam(paramL, nIteL, nEtaL, pathToSave, 
                initialCorAA = 1, initialMeanA = 1, initialChiAA = 1):
    
    # The arguments of the function:
        # paramL: list of the scalar parameters for the model and the algorithm (details below)
        # nIteL, nEtaL: the iteration process. 
            # For example nIteL=[20,10] and nEtaL=[1e3,1e4] will launch
            # 20 iterations with 1e3 trajectories, then 10 iterations with 1e4 trajectories.
        # pathToSave: the path on the computer where to store the results and log file
        # initialCorAA, initialMeanA, initialChiAA: the initial guesses for the correlation matrix,
            # the average vector and the response matrix. If not specified, they will be taken as detailed
            # in the function "iterations_with_stop"
            
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
    
    # Perform the iterations, calls the function "iterations_with_stop")
    (iterFinalAutocorA, iterFinalMeanA, iterIntegratedChiA, 
     iterNormDifA, corAA, meanA, chiAA) = loopDMFT_withGam(paramL, nIteL, 
        nEtaL, pathForLogFile, initialCorAA=initialCorAA,
        initialMeanA=initialMeanA, initialChiAA=initialChiAA)
    
    # Write the arrays in the folder
    paramForSaveA = np.array( [ mu, sig, gam, lamda, tMax, threshold, soft, tau, nIte ] )
    os.chdir( pathToSave + pathFolder + '/arrays' )
    np.save('corAA',corAA)
    np.save('meanA',meanA)
    np.save('chiAA', chiAA)
    np.save('iterFinalAutocorA',iterFinalAutocorA)
    np.save('iterFinalMeanA', iterFinalMeanA)
    np.save('iterIntegratedChiA', iterIntegratedChiA)
    np.save('iterNormDifA',iterNormDifA)
    np.save('paramA',paramForSaveA)
    np.save('nIteL',nIteL)
    np.save('nEtaL',nEtaL)
    
    return 0    
