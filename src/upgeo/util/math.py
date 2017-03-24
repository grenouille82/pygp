'''
Created on Jul 26, 2012

@author: marcel
'''
import numpy as np

def fmin_ern(f, x0, fprime, args=None, maxiter=None, reduc=1):
    '''
    '''
    INT = 0.1    # don't reevaluate within 0.1 of the limit of the current bracket
    EXT = 3.0    # extrapolate maximum 3 times the current step-size
    MAX = 20     # max 20 function evaluations per line search
    RATIO = 10   # maximum allowed slope ratio
    SIG = 0.1 
    RHO = SIG/2 
# SIG and RHO are the constants controlling the Wolfe-
# Powell conditions. SIG is the maximum allowed absolute ratio between
# previous and new slopes (derivatives in the search direction), thus setting
# SIG to low (positive) values forces higher precision in the line-searches.
# RHO is the minimum allowed fraction of the expected (from the slope at the
# initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
# Tuning of SIG (depending on the nature of the function to be optimized) may
# speed up the minimization; it is probably not worth playing much with RHO.

    
    i = 0               #zero the run length counter
    ls_failed = False   #no previous line search has failed
    f0 = f(x0)          #get function value
    df0 = fprime(x0)    #get gradient
    i = i + 1           #count epoch (todo)
    fX = f0
    s = -df0            #initial search direction (steepest) and slope
    d0 = np.dot(-s,s)
    x3 = reduc/(1.0-d0) #initial step is red/(|s|+1)
    
    while i < np.abs(maxiter):
        i = i + 1                   #count iterations?!
        
        X0 = x0; F0 = f0; dF0 = df0 #make a copy of current values
        M = MAX
        while True: #keep extrapolating as long as necessary
            x2 = 0; f2 = f0; d2 = d0; f3 = f0; df3 = df0
            success = False
            while not success and M > 0:
                try:
                    M = M - 1
                    i = i + 1
                    
                    f3 = f(x0+x3*s)
                    df3 = fprime(x0+x3*s)
                    
                    if np.isnan(f3) or np.isinf(f3) or np.any(np.logical_or(np.isnan(df3), np.isinf(df3))):
                        raise
                    success = True 
                except:                 #catch any error which occured in f
                    print 'schwanz'
                    x3 = (x2+x3)/2.0    #bisect and try again
                    
            if f3 < F0:
                print 'best={0}'.format(f3)
                X0 = x0+x3*s
                F0 = f3
                dF0 = df3
            
            d3 = np.dot(df3,s)      #new slope
                
            if d3 > SIG or f3 > f0+x3*RHO*d0 or M == 0: #are we done extrapolating?
                break
                
            x1 = x2; f1 = f2; d1 = d2   #move point 2 to point 1
            x2 = x3; f2 = f3; d2 = d3   #move point 3 to point 2
            A = 6*(f1-f2)+3*(d2+d1)*(x2-x1) #make cubic extrapolation
            B = 3*(f2-f1)-(2*d1+d2)*(x2-x1)
            x3 = x1-d1*(x2-x1)**2.0/(B+np.sqrt(B*B-A*d1*(x2-x1)))           #num. error possible, ok!
            if not np.isreal(x3) or np.isnan(x3) or np.isinf(x3) or x3 < 0:  # num prob | wrong sign?
                x3 = x2*EXT     #extrapolate maximum amount
            elif x3 > x2*EXT:   #new point beyond extrapolation limit?
                x3 = x2*EXT     #extrapolate maximum amount
            elif x3 < x2+INT*(x2-x1): #new point too close to previous point?
                x3 = x2+INT*(x2-x1)
                    
        while (np.abs(d3) > -SIG*d0 or f3 > f0+x3*RHO*d0) and M > 0:  # keep interpolating
            #choose subinterval
            if d3 > 0 or f3 > f0+x3*RHO*d0:
                x4 = x3; f4 = f3; d4 = d3
            else:
                x2 = x3; f2 = f3; d2 = d3
            
            if f4 > f0: #f4 may be not initialized (same problem as in Rasmussen)
                x3 = x2-(0.5*d2*(x4-x2)**2)/(f4-f2-d2*(x4-x2)) #quadratic interpolation
            else:
                A = 6.0*(f2-f4)/(x4-x2)+3*(d4+d2) #cubic interpolation
                B = 3.0*(f4-f2)-(2.0*d2+d4)*(x4-x2)
                x3 = x2+(np.sqrt(B*B-A*d2*(x4-x2)**2)-B)/A
            
            if np.isnan(x3) or np.isinf(x3):
                x3 = (x2+x4)/2.0    #if we had a numerical problem then bisect
                
            x3 = max(min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2)) 
            f3 = f(x0+x3*s)
            df3 = fprime(x0+x3*s)
            
            if f3 < F0: #keep best values
                X0 = x0+x3*s
                F0 = f3
                dF0 = df3
            
            M = M - 1
            i = i + 1           #count epochs?!
            d3 = np.dot(df3,s)  #new slope
        
        if np.abs(d3) < -SIG*d0 and f3 < f0+x3*RHO*d0:  #if line search suceeded
            print 'print succed'
            x0 = x0+x3*s; f0 = f3; fX = np.r_[fX, f0]   #update variables
            #checke ob die klammer bei s korrekt ist
            s = (np.dot(df3,df3)-np.dot(df0,df3))/np.dot(df0,df0)*s - df3 #Polack-Ribiere CG direction
            df0 = df3                                   #swap derivates
            d3 = d0; d0 = np.dot(df0,s)
            if d0 > 0: #new slope must be negative
                s = -df0; d0 = np.dot(-s,s)
            
            x3 = x3 * np.min([RATIO, d3/(d0-np.finfo(np.float64).eps)])    #slope ratio but max RATIO
            ls_failed = False
        else:
            print 'print fsail'
            x0 = X0; f0 = F0; df0 = dF0;                 #restore best point so far
            if ls_failed or i > np.abs(maxiter):         #line search failed twice in a row
                print 'brek'
                break                                    #or we ran out of time, so we give up
            s = -df0
            d0 = np.dot(-s,s);
            ls_failed = True
    
    print 'number of iterations: {0}'.format(i)
    print 'func values: {0}'.format(fX)
    
    return x0
              
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
#
#
#This file contains a Python version of Carl Rasmussen's Matlab-function 
#minimize.m
#
#minimize.m is copyright (C) 1999 - 2006, Carl Edward Rasmussen.
#Python adaptation by Roland Memisevic 2008.
#
#
#The following is the original copyright notice that comes with the 
#function minimize.m
#(from http://www.kyb.tuebingen.mpg.de/bs/people/carl/code/minimize/Copyright):
#
#
#"(C) Copyright 1999 - 2006, Carl Edward Rasmussen
#
#Permission is granted for anyone to copy, use, or modify these
#programs and accompanying documents for purposes of research or
#education, provided this copyright notice is retained, and note is
#made of any changes that have been made.
#
#These programs and documents are distributed without any warranty,
#express or implied.  As the programs were written for research
#purposes only, they have not been tested to the degree that would be
#advisable in any important application.  All use of these programs is
#entirely at the user's own risk."


"""minimize.py 

This module contains a function 'minimize' that performs unconstrained
gradient based optimization using nonlinear conjugate gradients. 

The function is a straightforward Python-translation of Carl Rasmussen's
Matlab-function minimize.m

"""


#from numpy import dot, isinf, isnan, any, sqrt, isreal, real, nan, inf

def fmin_ras(X, f, grad, args, maxnumlinesearch=None, maxnumfuneval=None, red=1.0, verbose=True):
    INT = 0.1;# don't reevaluate within 0.1 of the limit of the current bracket
    EXT = 3.0;              # extrapolate maximum 3 times the current step-size
    MAX = 20;                     # max 20 function evaluations per line search
    RATIO = 10;                                   # maximum allowed slope ratio
    SIG = 0.1;RHO = SIG/2;# SIG and RHO are the constants controlling the Wolfe-
    #Powell conditions. SIG is the maximum allowed absolute ratio between
    #previous and new slopes (derivatives in the search direction), thus setting
    #SIG to low (positive) values forces higher precision in the line-searches.
    #RHO is the minimum allowed fraction of the expected (from the slope at the
    #initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
    #Tuning of SIG (depending on the nature of the function to be optimized) may
    #speed up the minimization; it is probably not worth playing much with RHO.

    SMALL = 10.**-16                    #minimize.m uses matlab's realmin 
    
    if maxnumlinesearch == None:
        if maxnumfuneval == None:
            raise "Specify maxnumlinesearch or maxnumfuneval"
        else:
            S = 'Function evaluation'
            length = maxnumfuneval
    else:
        if maxnumfuneval != None:
            raise "Specify either maxnumlinesearch or maxnumfuneval (not both)"
        else: 
            S = 'Linesearch'
            length = maxnumlinesearch

    i = 0                                         # zero the run length counter
    ls_failed = 0                          # no previous line search has failed
    f0 = f(X, *args)                          # get function value and gradient
    df0 = grad(X, *args)  
    fX = [f0]
    i = i + (length<0)                                         # count epochs?!
    s = -df0; d0 = -np.dot(s,s)    # initial search direction (steepest) and slope
    x3 = red/(1.0-d0)                             # initial step is red/(|s|+1)

    while i < abs(length):                                 # while not finished
        i = i + (length>0)                                 # count iterations?!

        X0 = X; F0 = f0; dF0 = df0              # make a copy of current values
        if length>0:
            M = MAX
        else: 
            M = min(MAX, -length-i)
        while 1:                      # keep extrapolating as long as necessary
            x2 = 0; f2 = f0; d2 = d0; f3 = f0; df3 = df0
            success = 0
            while (not success) and (M > 0):
                try:
                    M = M - 1; i = i + (length<0)              # count epochs?!
                    f3 = f(X+x3*s, *args)
                    df3 = grad(X+x3*s, *args)
                    if np.isnan(f3) or np.isinf(f3) or any(np.isnan(df3)+np.isinf(df3)):
                        print "error"
                        return
                    success = 1
                except:                    # catch any error which occured in f
                    x3 = (x2+x3)/2                       # bisect and try again
            if f3 < F0:
                X0 = X+x3*s; F0 = f3; dF0 = df3   # keep best values
            d3 = np.dot(df3,s)                                         # new slope
            if d3 > SIG*d0 or f3 > f0+x3*RHO*d0 or M == 0:  
                                                   # are we done extrapolating?
                break
            x1 = x2; f1 = f2; d1 = d2                 # move point 2 to point 1
            x2 = x3; f2 = f3; d2 = d3                 # move point 3 to point 2
            A = 6*(f1-f2)+3*(d2+d1)*(x2-x1)          # make cubic extrapolation
            B = 3*(f2-f1)-(2*d1+d2)*(x2-x1)
            Z = B+np.sqrt(complex(B*B-A*d1*(x2-x1)))
            if Z != 0.0:
                x3 = x1-d1*(x2-x1)**2/Z              # num. error possible, ok!
            else: 
                x3 = np.inf
            if (not np.isreal(x3)) or np.isnan(x3) or np.isinf(x3) or (x3 < 0): 
                                                       # num prob | wrong sign?
                x3 = x2*EXT                        # extrapolate maximum amount
            elif x3 > x2*EXT:           # new point beyond extrapolation limit?
                x3 = x2*EXT                        # extrapolate maximum amount
            elif x3 < x2+INT*(x2-x1):  # new point too close to previous point?
                x3 = x2+INT*(x2-x1)
            x3 = np.real(x3)

        while (abs(d3) > -SIG*d0 or f3 > f0+x3*RHO*d0) and M > 0:  
                                                           # keep interpolating
            if (d3 > 0) or (f3 > f0+x3*RHO*d0):            # choose subinterval
                x4 = x3; f4 = f3; d4 = d3             # move point 3 to point 4
            else:
                x2 = x3; f2 = f3; d2 = d3             # move point 3 to point 2
            if f4 > f0:           
                x3 = x2-(0.5*d2*(x4-x2)**2)/(f4-f2-d2*(x4-x2))
                                                      # quadratic interpolation
            else:
                A = 6*(f2-f4)/(x4-x2)+3*(d4+d2)           # cubic interpolation
                B = 3*(f4-f2)-(2*d2+d4)*(x4-x2)
                if A != 0:
                    x3=x2+(np.sqrt(B*B-A*d2*(x4-x2)**2)-B)/A
                                                     # num. error possible, ok!
                else:
                    x3 = np.inf
            if np.isnan(x3) or np.isinf(x3):
                x3 = (x2+x4)/2      # if we had a numerical problem then bisect
            x3 = max(min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2))  
                                                       # don't accept too close
            f3 = f(X+x3*s, *args)
            df3 = grad(X+x3*s, *args)
            if f3 < F0:
                X0 = X+x3*s; F0 = f3; dF0 = df3              # keep best values
            M = M - 1; i = i + (length<0)                      # count epochs?!
            d3 = np.dot(df3,s)                                         # new slope

        if abs(d3) < -SIG*d0 and f3 < f0+x3*RHO*d0:  # if line search succeeded
            X = X+x3*s; f0 = f3; fX.append(f0)               # update variables
            if verbose: print '%s %6i;  Value %4.6e\r' % (S, i, f0)
            s = (np.dot(df3,df3)-np.dot(df0,df3))/np.dot(df0,df0)*s - df3
                                                  # Polack-Ribiere CG direction
            df0 = df3                                        # swap derivatives
            d3 = d0; d0 = np.dot(df0,s)
            if d0 > 0:                             # new slope must be negative
                s = -df0; d0 = -np.dot(s,s)     # otherwise use steepest direction
            x3 = x3 * min(RATIO, d3/(d0-SMALL))     # slope ratio but max RATIO
            ls_failed = 0                       # this line search did not fail
        else:
            X = X0; f0 = F0; df0 = dF0              # restore best point so far
            if ls_failed or (i>abs(length)):# line search failed twice in a row
                break                    # or we ran out of time, so we give up
            s = -df0; d0 = -np.dot(s,s)                             # try steepest
            x3 = 1/(1-d0)                     
            ls_failed = 1                             # this line search failed
    if verbose: print "\n"
    return X, fX, i

