"""
Performs an optimization to estimate the output length of the SDI seedless extractor
both asymptotically and in the finite-saze regime in terms of the number of rounds n.
Includes the possibility to add white noise to the behavior of the device.

Results are saved as xlsx files that can then be read to retrieve data for plotting.
"""


#%% IMPORT PACKAGES


import picos as pic
import numpy as np
from numpy.linalg import eigh
from scipy.optimize import minimize
from tqdm import tqdm
import timeit
import pandas as pd
import os


#%% DEFINE FUNCTIONS


def setNumericalPrecisionForSolver(problem):
    """
    Sets the numerical precission for the SDP solver
    
        problem -- sdp optimization problem
    """
    
    tol=1e-9
    problem.options["rel_ipm_opt_tol"]=tol
    problem.options["rel_prim_fsb_tol"]=tol
    problem.options["rel_dual_fsb_tol"]=tol
    problem.options["max_footprints"]=None

def kronecker_delta(i, j):
    """
    Defines the Kronecker delta function between two indices
    
        i,j -- variable indices
    """
    
    return np.equal(i, j).astype(int)

def dual(delta,behavior):
    """
    Solves the dual form of the optimization problem for the pguess (fixed input)
    
        delta -- lower bound to the fidelity between input states
        behavior -- expected conditional probability distribution for inputs/outputs
    """
    
    # We start by defining an object for the optimization problem
    prob=pic.Problem()
    nInputs = 2
    nOutputs = 2
    
    # We define some real variables for the "nu" parameters
    nu00=pic.RealVariable('nu00')
    nu01=pic.RealVariable('nu01')
    nu10=pic.RealVariable('nu10')
    nu11=pic.RealVariable('nu11')
    nu=[[nu00,nu01],[nu10,nu11]]
    
    # We define some 2x2 Hermitian variables for the "H" parameters
    H0=pic.HermitianVariable('H0',(2,2))
    H1=pic.HermitianVariable('H1',(2,2))
    H=[H0,H1]
    
    # We define some arbitrary pure states that saturate the bound on the fidelity
    rho0 = np.array([[delta**2,delta*np.sqrt(1-delta**2)],[delta*np.sqrt(1-delta**2),1-delta**2]])
    rho1 = np.array([[1,0],[0,0]])
    rhos = [rho0,rho1]

    # Add constraints
    for a in range(nOutputs):
        for lambda_value in [0, 1]:
            constr = H[lambda_value] - 0.5 * pic.trace(H[lambda_value]) * np.eye(2)

            constr += rhos[0]*kronecker_delta(lambda_value, a)
            for x in range(nInputs):
                constr += -rhos[x] * nu[a][x]

            prob.add_constraint(constr<<0)

    # We define the objective function for the minimization
    obj = sum([nu[a][x]*behavior[a][x] for a in range(nOutputs) for x in range(nInputs)])
    
    # We set the problem as a minimization, set the numerical precision and solve using the Mosek solver
    prob.set_objective('min',obj)
    setNumericalPrecisionForSolver(prob)
    prob.solve(solver = "mosek", verbosity = 0)
    
    optimal_nus = [nu[a][x].value for a in range(nOutputs) for x in range(nInputs)]   
    
    return prob.value,optimal_nus

def check_valid_nus(delta,opt_nus):
    """
    Checks the validity of obtained optimal parameters for the pguess dual optimization
    
        delta -- lower bound to the fidelity between input states
        opt_nus -- optimal values for the "nu" parameters
    """
    
    # We start by defining an object for the optimization problem
    prob=pic.Problem()
    nInputs = 2
    nOutputs = 2
    
    # We define some real variables for the "nu" parameters
    nu00=pic.RealVariable('nu00')
    nu01=pic.RealVariable('nu01')
    nu10=pic.RealVariable('nu10')
    nu11=pic.RealVariable('nu11')
    nu=[[nu00,nu01],[nu10,nu11]]
    
    # We define some 2x2 Hermitian variables for the "H" parameters
    H0=pic.HermitianVariable('H0',(2,2))
    H1=pic.HermitianVariable('H1',(2,2))
    H=[H0,H1]
    
    # We define some arbitrary pure states that saturate the bound on the fidelity
    rho0 = np.array([[delta**2,delta*np.sqrt(1-delta**2)],[delta*np.sqrt(1-delta**2),1-delta**2]])
    rho1 = np.array([[1,0],[0,0]])
    rhos = [rho0,rho1]

    # Add constraints
    for a in range(nOutputs):
        for lambda_value in [0, 1]:
            constr = H[lambda_value] - 0.5 * pic.trace(H[lambda_value]) * np.eye(2)

            constr += rhos[0]*kronecker_delta(lambda_value, a)
            for x in range(nInputs):
                constr += -rhos[x] * nu[a][x]

            prob.add_constraint(constr<<0)

    prob.add_list_of_constraints([nu[a][x]==opt_nus[a][x] for a in range(nOutputs) for x in range(nInputs)])
    
    # We set the problem as a minimization, set the numerical precision and solve using the Mosek solver
    prob.set_objective('min',0*nu[0][0])
    setNumericalPrecisionForSolver(prob)
    prob.solve(solver = "mosek", verbosity = 0)
    
    return prob.status=="optimal"

def unpack_params(params):
    """
    Unpacks the list of parameters for the optimization to calculate the length of the output of the extraction protocol
        
        params -- list of parameters [beta,{alpha[a,x]},{nu[a,x]},{gamma[i]}]
    """
    beta = params[0]
    alphas = np.array([[params[1], params[2]], [params[3], params[4]]])
    nus = np.array([[params[5], params[6]], [params[7], params[8]]])
    gammas = np.array(params[9:17]).reshape((2,4))
    
    return beta, alphas, nus, gammas

def objective_asym(params):
    """ 
    Objective function to minimize for the calculation of the asymptotic efficiency rate
    Args:
        params (list): List of parameters [beta,{alpha[a,x]},{nu[a,x]},{gamma[i]}]
    """
    beta,alphas,_,_ = unpack_params(params)
    
    return -(p_e*0.5*sum(behavior[a][x]*alphas[a][x] for a in range(2) for x in range(2)) + p_r*beta) 

def objective(params):
    """ 
    Objective function to minimize for the calculation of the finite size efficiency rate
    Args:
        params (list): List of parameters [beta,{alpha[a,x]},{nu[a,x]},{gamma[i]}]
    """
    beta, alphas, _, _ = unpack_params(params)
    
    return -escale*(sum(counts[a][x]*alphas[a][x] for a in range(2) for x in range(2))
                    + beta*n_r - 2*np.log2(1/eps) - 4*np.log2(n_r))

def linear_constraints(a,lam,params):
    """ 
    Linear constraints for the optimization problem

        a -- output index (0 or 1)
        lam -- strategy index (0 or 1)
        params -- list of parameters [beta,{alpha[a,x]},{nu[a,x]},{gamma[i]}]
    """
    #Since we cannot impose a matricial constraint, we just need to impose that the eigenvalues of
    #M (the constraint) are negative since the constraint must be negative semidefinite
    
    _,_,nus,gammas = unpack_params(params)
    
    H_lam = [[gammas[lam][0],gammas[lam][1]+gammas[lam][2]*1j],[gammas[lam][1]-gammas[lam][2]*1j,gammas[lam][3]]]
    
    M = rho0*((1 if a==lam else 0) - nus[a][0])
    M += -rho1*nus[a][1]
    M = M.astype(np.complex128)
    M += H_lam - 0.5*np.trace(H_lam)*np.eye(2)
    eigenvals,_ = eigh(M)
    
    return -eigenvals[0].real,-eigenvals[1].real #The - sign is because, by default, scipy's ineq are >= 0, but we want <= 0

def nonlinear_constraints(a,x,params):
    """ 
    Nonlinear constraints for the optimization problem

        params -- list of parameters [beta,{alpha[a,x]},{nu[a,x]},{gamma[i]}]
    """
    beta,alphas,nus,_ = unpack_params(params)
    
    return -p_r*np.sqrt(2)**(beta-1)*(4*nus[a][x]) - p_e*np.sqrt(2)**alphas[a][x] + 1 

def include_constraints(constraints):
    """ 
    Fill the list of constraints

        constraints -- empty list of constraints
    """
    
    # Linear constraints
    for a in [0, 1]:
        for lam in [0, 1]:
            constraints.append({
                'type': 'ineq',
                'fun': lambda params, a=a, lam=lam: linear_constraints(a, lam, params)[0]
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda params, a=a, lam=lam: linear_constraints(a, lam, params)[1]
            })

    # Nonlinear constraints
    for a in [0, 1]:
        for x in [0, 1]:
            constraints.append({
                'type': 'eq',
                'fun': lambda params, a=a, x=x: nonlinear_constraints(a, x, params)
            })
            
    return constraints


#%% CALCULATE NOISY ASYMPTOTIC RATE


# Generate output file
output_file = "Noisy_Results_Asymptotic.xlsx"

# Time the optimization
start_time = timeit.default_timer()

# Probability of estimation and key generation rounds
p_e = 0.955
p_r = 1 - p_e

# Lower bound on the fidelity (square root)
delta = 0.500004

# Ideal behavior
p0_id = delta
p1_id = 0.999999
behavior_id = np.array([[p0_id, p1_id], [1 - p0_id, 1 - p1_id]])

# Quantum states for the optimization
rho0 = np.array([[delta, np.sqrt(delta)*np.sqrt(1-delta)],[np.sqrt(delta)*np.sqrt(1-delta), 1-delta]])
rho1 = np.array([[1,0],[0,0]])

# Noise values to consider
noise_vals = np.linspace(0.0, 0.001, num=11)

for noise_val in noise_vals:
    try:    
        
        # Noisy behavior
        noise_prob = noise_val
        behavior = (1 - noise_prob) * behavior_id + noise_prob * np.array([[0.5, 0.5], [0.5, 0.5]])
        p0, p1 = behavior[0][0], behavior[0][1]
        
        # Dual value
        dual_value, _ = dual(np.sqrt(delta), behavior)
    
        # Definition of constraints
        constraints = []
        include_constraints(constraints)
        
        # Initialization of optimization parameters
        x0 = np.zeros(17)
        x0[5:9] = 0.5  # initialize nus ~0.5
        
        # Calculation of asymptotic rate
        result = minimize(objective_asym,
                        x0=x0,
                        constraints=constraints,
                        method="SLSQP",
                        options={'ftol': 1e-8,"maxiter":3000,"disp": False}
                        )
        if result.success and result.fun<=0:
            beta,alphas,nus,_ = unpack_params(result.x)
            try:
                if check_valid_nus(np.sqrt(delta), nus):
                    print(f"p_e={p_e}, p0={p0}, dual={dual_value}, Reff={-result.fun}", '\n')
            except Exception as e:
                print(f"Skipping p0={p0} due to solver failure: {e}")
        else:
            print(f"Optimization failed for p_e={p_e}, p0={p0}: {result.message}", '\n')
    
        # Save to Excel (crash-safe incremental writing with timestamp)
        row_data = {
            "Timestamp": pd.Timestamp.now(),
            "Reff": -result.fun,
            "Dual value": dual_value,
            "Delta": delta,
            "p0": p0,
            "p1": p1,
            "Noise probability": noise_prob,
            "p_e": p_e,
            "maxiter": 3000
        }
        if os.path.exists(output_file):
            df_existing = pd.read_excel(output_file)
            df_new = pd.DataFrame([row_data])
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = pd.DataFrame([row_data])

        df_combined.to_excel(output_file, index=False, engine="openpyxl")
        print(f"✅ Results saved to {output_file}\n")
    
    
    except Exception as e:
        print(f"⚠️ Exception at iteration i={noise_vals.index(noise_val)}: {e}")
        continue
    
elapsed = timeit.default_timer() - start_time
print(f"\nTotal time taken for optimization: {elapsed:.2f} s\n")


#%% CALCULATE NOISY FINITE SIZE RATE


# Generate output file
output_file = "Noisy_Results_Finite.xlsx"

# Time the optimization
start_time = timeit.default_timer()

# Probability of estimation and key generation rounds
p_e = 0.955
p_r = 1 - p_e

# Lower bound on the fidelity (square root)
delta = 0.500004

# Ideal behavior
p0_id = delta
p1_id = 0.999999
behavior_id = np.array([[p0_id, p1_id], [1 - p0_id, 1 - p1_id]])

# Quantum states for the optimization
rho0 = np.array([[delta, np.sqrt(delta)*np.sqrt(1-delta)],[np.sqrt(delta)*np.sqrt(1-delta), 1-delta]])
rho1 = np.array([[1,0],[0,0]])

# Security parameter
eps = 1e-10

# Scale parameter (required for optimization)
escale = 1e-5

# Number of rounds to consider
n_values = np.array([1e4,3e4,1e5,1e6])

# Noise values to consider
noise_vals = np.linspace(0.0, 0.001, num=11)

# Number of iterations for expectation value
n_iterations = 100

for n in n_values:
    for noise_val in noise_vals:  
        try:
            
            # Define lists to store relevant values and error counts       
            optvalue, reffvalue, rextvalue, dualvalue = [], [], [], []
            errvalue = invalidnus = exceptioncheck = exceptdual = 0
            
            # Noisy behavior
            noise_prob = noise_val
            behavior = (1 - noise_prob) * behavior_id + noise_prob * np.array([[0.5, 0.5], [0.5, 0.5]])
            p0, p1 = behavior[0][0], behavior[0][1]
    
            for j in tqdm(range(n_iterations), desc=f"n={int(n)}"):
                
                # Number of estimation and raw key rounds
                n_e = int(n * p_e)
                n_r = n - n_e
                
                # Randomly sample from noisy behavior for estimation rounds
                x = np.random.choice([0, 1], size=n_e, p=[0.5, 0.5])
                p_a0_given_x = np.where(x == 0, p0, p1)
                a = (np.random.rand(n_e) >= p_a0_given_x).astype(int)
                counts = np.array([[np.sum((a == a_i) & (x == x_j)) for x_j in (0, 1)] for a_i in (0, 1)])
                
                # Definition of constraints
                constraints = []
                include_constraints(constraints)
                
                # Initialization of optimization parameters
                x0 = np.zeros(17)
                x0[5:9] = 0.5  # initialize nus ~0.5
    
                # Calculation of asymptotic rate
                result = minimize(objective, x0=x0, constraints=constraints, method="SLSQP",
                                  options={'ftol': 1e-8, "maxiter": 3000, "disp": False})
                if result.success and result.fun <= 0:
                    beta, alphas, nus, _ = unpack_params(result.x)
                    try:
                        dual_value, _ = dual(np.sqrt(delta), behavior)
                        dualvalue.append(dual_value)
                    except Exception as e:
                        exceptdual += 1
                        print(f"Skipping due to exception in dual: {e}", '\n')
                    try:
                        if check_valid_nus(np.sqrt(delta), nus):
                            optvalue.append(-result.fun/escale) # total output length
                            reffvalue.append(-result.fun/(escale*n)) # efficiency rate
                            rextvalue.append(-result.fun/(escale*n_r)) # extraction rate
                            print(f"p_e={p_e}, p0={p0}, p1={p1}, dual={dual_value}, m(z,t):{-result.fun/(escale)}, Rext={-result.fun/(escale*n_r)}, Reff={-result.fun/(escale*n)}", '\n')
                        else:
                            invalidnus += 1
                            print(f"Invalid nus for p_e={p_e}, p0={p0}, p1={p1}", '\n')
                    except Exception as e:
                        exceptioncheck += 1
                        print(f"Skipping due to exception in check_valid_nus at p_e={p_e}, p0={p0}, p1={p1}: {e}", '\n')
                        continue
                else:
                    errvalue += 1
                    print(f"Optimization failed for p_e={p_e}, p0={p0}, p1={p1}: {result.message}", '\n')
    
            # Aggregate results and calculate error rates
            avgoptvalue = np.mean(optvalue) if len(optvalue) > 0 else 0
            avgreffvalue = np.mean(reffvalue) if len(reffvalue) > 0 else 0
            avgrextvalue = np.mean(rextvalue) if len(rextvalue) > 0 else 0
            stdoptvalue = np.std(optvalue) if len(optvalue) > 0 else 0
            stdreffvalue = np.std(reffvalue) if len(reffvalue) > 0 else 0
            stdrextvalue = np.std(rextvalue) if len(rextvalue) > 0 else 0
            avgdualvalue = np.mean(dualvalue) if len(dualvalue) > 0 else 0
            err_prop = errvalue / n_iterations
            invnus_prop = invalidnus / n_iterations
            except_prop = exceptioncheck / n_iterations
            exceptdual_prop = exceptdual / n_iterations
    
            print(f"\n=== Results for n={n} ===")
            print(f"Avg optimal value: {avgoptvalue} ± {stdoptvalue}")
            print(f"Avg Reff: {avgreffvalue} ± {stdreffvalue}")
            print(f"Avg Rext: {avgrextvalue} ± {stdrextvalue}")
            print(f"Error rate: {err_prop}, Invalid nus: {invnus_prop}, Exception rate: {except_prop}")
            print(f"Avg dual value: {avgdualvalue}, Dual exceptions: {exceptdual_prop}")
    
            # Save to Excel (crash-safe incremental writing with timestamp)
            row_data = {
                "Timestamp": pd.Timestamp.now(),
                "n": n,
                "Avg optimal value": avgoptvalue,
                "Dev": stdoptvalue,
                "Avg Reff": avgreffvalue,
                "Dev.1": stdreffvalue,
                "Avg Rext": avgrextvalue,
                "Dev.2": stdrextvalue,
                "Err Rate": err_prop,
                "N(avg)": n_iterations,
                "Dual value": avgdualvalue,
                "Delta": delta,
                "p0": p0,
                "p1": p1,
                "Noise probability": noise_prob,
                "p_e": p_e,
                "eps": eps,
                "escale": escale,
                "maxiter": 3000
            }
            if os.path.exists(output_file):
                df_existing = pd.read_excel(output_file)
                df_new = pd.DataFrame([row_data])
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_combined = pd.DataFrame([row_data])
    
            df_combined.to_excel(output_file, index=False, engine="openpyxl")
            print(f"✅ Results saved to {output_file}\n")
    
        except Exception as e:
            print(f"⚠️ Exception at iteration i={n_values.index(n)}, n={n}: {e}")
            continue

elapsed = timeit.default_timer() - start_time
print(f"\nTotal time taken for optimization: {elapsed:.2f} s\n")