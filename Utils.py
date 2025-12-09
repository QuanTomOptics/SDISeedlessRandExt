"""
Includes all the functions used throughout the rest of the codes

"""


import picos as pic
import numpy as np
from numpy.linalg import eigh


#------------------------------------------------------------#
# GENERAL PURPOSE
#------------------------------------------------------------#

def kronecker_delta(i, j):
    """
    Defines the Kronecker delta function between two indices
    
        i,j -- variable indices
    """
    
    return np.equal(i, j).astype(int)



def sci_label(x):
    """
    Function to replace python scientific notation with physics scientific notation
    
        x -- float to convert to physics scientific notation
    """
    
    # Format in scientific notation, e.g. "1.23e-4"
    s = f"{x:.0e}"
    coeff, exp = s.split("e")
    exp = int(exp)  # remove + sign, leading zeros
    
    return rf"${coeff} \times 10^{{{exp}}}$"


#------------------------------------------------------------#
# SDP OPTIMIZATION PARAMETERS
#------------------------------------------------------------#


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
    

#------------------------------------------------------------#
# PGUESS CALCULATION
#------------------------------------------------------------#    


def dual(delta,behavior):
    """
    Solves the dual form of the optimization problem for the pguess (fixed input) defined in Eq.(3)
    
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
    
    # We set the problem as a minimization, set the numerical precision and solve using the mosek/qics solver
    prob.set_objective('min',obj)
    setNumericalPrecisionForSolver(prob)
    prob.solve(solver = "qics", verbosity = 0)
    
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
    
    # We set the problem as a minimization, set the numerical precision and solve using the mosek/qics solver
    prob.set_objective('min',0*nu[0][0])
    setNumericalPrecisionForSolver(prob)
    prob.solve(solver = "qics", verbosity = 0)
    
    return prob.status=="optimal"


#------------------------------------------------------------#
# EXTRACTOR RATES
#------------------------------------------------------------#


def unpack_params(params):
    """
    Unpacks the list of parameters for the optimizations of m(t,z) in Eqs.(12) and (14) 
        
        params -- list of parameters [beta,{alpha[a,x]},{nu[a,x]},{gamma[i]}]
    """
    beta = params[0]
    alphas = np.array([[params[1], params[2]], [params[3], params[4]]])
    nus = np.array([[params[5], params[6]], [params[7], params[8]]])
    gammas = np.array(params[9:17]).reshape((2,4))
    
    return beta, alphas, nus, gammas


def objective_asym_XOR(params,behavior,p_e,p_r):
    """ 
    Objective function to minimize to check if there is extraction for the XOR extractor
    Args:
        params -- list of parameters [beta,{alpha[a,x]},{nu[a,x]},{gamma[i]}]
        behavior -- expected conditional probability distribution for inputs/outputs
        p_e -- probability of an estimation round
        p_r -- probability of a raw key generation round
    """
    beta,alphas,_,_ = unpack_params(params)
    
    return -(p_e/2*sum(behavior[a][x]*alphas[a][x] for a in range(2) for x in range(2)) + p_r*(beta-1)) 


def objective_asym(params,behavior,p_e,p_r):
    """ 
    Objective function to minimize for the calculation of the asymptotic efficiency rate for the multi-bit extractors
    Args:
        params (list): List of parameters [beta,{alpha[a,x]},{nu[a,x]},{gamma[i]}]
        behavior -- expected conditional probability distribution for inputs/outputs
        p_e -- probability of an estimation round
        p_r -- probability of a raw key generation round
    """
    beta,alphas,_,_ = unpack_params(params)
    
    return -(p_e/2*sum(behavior[a][x]*alphas[a][x] for a in range(2) for x in range(2)) + p_r*beta) 


def objective(params,behavior,p_e,p_r,n_r,counts,escale,eps):
    """ 
    Objective function to minimize for the calculation of the finite size efficiency rate for the multi-bit extractors
    Args:
        params -- list of parameters [beta,{alpha[a,x]},{nu[a,x]},{gamma[i]}]
        behavior -- expected conditional probability distribution for inputs/outputs
        p_e -- probability of an estimation round
        p_r -- probability of a raw key generation round
        n_r -- length of the raw key
        counts -- finite-size sample of size n_e of the behavior
        escale -- scale factor for numerical stability
        eps -- security parameter of the protocol
    """
    beta, alphas, _, _ = unpack_params(params)
    
    return -escale*(sum(counts[a][x]*alphas[a][x] for a in range(2) for x in range(2))
                    + beta*n_r - 2*np.log2(1/eps) - 4*np.log2(n_r))


def linear_constraints(a,lam,params,rho0,rho1):
    """ 
    Linear constraints for the optimization problem for both kinds of extractors

        a -- output index (0 or 1)
        lam -- strategy index (0 or 1)
        params -- list of parameters [beta,{alpha[a,x]},{nu[a,x]},{gamma[i]}]
        rho0, rho1 -- arbitrary pure quantum states with a fixed overlap
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


def nonlinear_constraints_XOR(a,x,params,p_e,p_r):
    """ 
    Nonlinear constraints for the optimization problem for the XOR extractor

        a -- output index (0 or 1)
        x -- input index (0 or 1)
        params -- list of parameters [beta,{alpha[a,x]},{nu[a,x]},{gamma[i]}]
        p_e -- probability of an estimation round
        p_r -- probability of a raw key generation round
    """
    beta,alphas,nus,_ = unpack_params(params)
    
    return -p_r*np.sqrt(2)**(beta-1)*(4*nus[a][x]-1) - p_e*np.sqrt(2)**alphas[a][x] + 1


def nonlinear_constraints(a,x,params,p_e,p_r):
    """ 
    Nonlinear constraints for the optimization problem for the multi-bit extractors

         a -- output index (0 or 1)
         x -- input index (0 or 1)
         params -- list of parameters [beta,{alpha[a,x]},{nu[a,x]},{gamma[i]}]
         p_e -- probability of an estimation round
         p_r -- probability of a raw key generation round
    """
    beta,alphas,nus,_ = unpack_params(params)
    
    return -p_r*np.sqrt(2)**(beta-1)*(4*nus[a][x]) - p_e*np.sqrt(2)**alphas[a][x] + 1 


def include_constraints(constraints,rho0,rho1,p_e,p_r,ext_type):
    """ 
    Function to construct the list of constraints for both types of extractors

        constraints -- empty list of constraints
        rho0, rho1 -- arbitrary pure quantum states with a fixed overlap
        p_e -- probability of an estimation round
        p_r -- probability of a raw key generation round
        ext_type -- type of extractor ({'XOR', 'MBIT'})
    """
    
    # Linear constraints
    for a in [0, 1]:
        for lam in [0, 1]:
            constraints.append({
                'type': 'ineq',
                'fun': lambda params, a=a, lam=lam: linear_constraints(a,lam,params,rho0,rho1)[0]
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda params, a=a, lam=lam: linear_constraints(a,lam,params,rho0,rho1)[1]
            })

    # Nonlinear constraints
    if ext_type == 'XOR':
        for a in [0, 1]:
            for x in [0, 1]:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda params, a=a, x=x: nonlinear_constraints_XOR(a,x,params,p_e,p_r)
                })
    elif ext_type == 'MBIT':
        for a in [0, 1]:
            for x in [0, 1]:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda params, a=a, x=x: nonlinear_constraints(a,x,params,p_e,p_r)
                })
            
    return constraints
