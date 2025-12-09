"""
Performs an optimization to estimate the minimum p_e sufficient for asymptotic extraction 
using the XOR extractor for a given value of delta

"""


#%% IMPORT PACKAGES


import numpy as np
from scipy.optimize import minimize
from itertools import product
import Utils as ut


#%% XOR OPTIMIZATION


# Error tolerance for the optimization
errorTolerance = 1e-10

# Value of delta we want to test
delta = 0.001

# Range of values for p_e we want to test
p_e_vals = np.linspace(0.0001,1,1000)

# Behavior
p0 = delta
p1 = 1
behavior = np.array([[p0,p1],[1-p0,1-p1]]) # behavior[a][x] = p(a|x)

# Quantum states for the optimization
rho0 = np.array([[delta, np.sqrt(delta)*np.sqrt(1-delta)],[np.sqrt(delta)*np.sqrt(1-delta), 1-delta]])
rho1 = np.array([[1,0],[0,0]])

for p_e in p_e_vals:
    p_r = 1 - p_e
    
    # Definition of constraints
    constraints = []
    ut.include_constraints(constraints,rho0,rho1,p_e,p_r,'XOR') # Select 'XOR' or 'MBIT'

    numOfattemps=100
    np.random.seed(123) # Reproducibility
    
    # Calculation of minimum p_e
    found = False
    for _ in range(numOfattemps):
        initialGuess = np.random.uniform(-10,10,size=17)
        
        result = minimize(ut.objective_asym_XOR,
                        x0=initialGuess,
                        args=(behavior, p_e, p_r),
                        constraints=constraints,
                        method="SLSQP",
                        options={'ftol': errorTolerance,"maxiter":1000,"disp": False}
                        )
        if result.success and result.fun<=0 and not np.isclose(result.fun,0): # We look for the suitable result closest to 0
            beta,alphas,nus,_ = ut.unpack_params(result.x)
            if ut.check_valid_nus(delta,nus):
                viols=[]
                for a,x in product([0,1],[0,1]):
                   viols.append(np.abs(-p_r*np.sqrt(2)**(beta-1)*(4*nus[a][x]-1) - p_e*np.sqrt(2)**alphas[a][x] + 1))

                print("For delta=",delta, "a value of p_e sufficient for extraction is",p_e)
                print("optimization's obj value=",-result.fun,"nonlinear constraint's violations:",viols)
                found=True
                break
    if found:
        break
