"""
Calculates and plots the guessing probability for Eve in terms of the lower bound on the fidelity.
"""


#%% IMPORT PACKAGES


import numpy as np
import picos as pic
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn style and context
sns.set(
    style="whitegrid",      # axes style
    context="notebook",     # font sizes / scaling
    rc={
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'text.usetex': True
    },
    font_scale=1.2
)


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

#%% DUAL CALCULATION


# Vector to store the values of pguess and delta
pguessvect = np.empty(1001)
deltavect = np.empty(1001)

for i in range(1001):
    
    # Define delta and behavior
    delta = 1/1000*i
    p0 = delta
    p1 = 0.99999
    behavior = [[p0,p1],[1-p0,1-p1]]
    
    # Helper variable (to avoid numerical problems)
    delta_aux = p0 * p1
    
    # Calculate guessing probability
    pguess,_ = dual(np.sqrt(delta_aux), behavior)

    # Store values
    deltavect[i] = delta
    pguessvect[i] = pguess


#%% PLOT


# Plot the results
fig, ax2 = plt.subplots(figsize=(9, 6), constrained_layout=True)
fig.patch.set_facecolor("#fdfdfd")

colors = sns.color_palette("Set2", 3)

ax2.plot(deltavect, pguessvect, color=colors[2], lw=2)
plt.xticks([0.0,0.25,0.5,0.75,1.0], fontsize=16)
plt.yticks([0.5,0.6,0.7,0.8,0.9,1.0], fontsize=16)
ax2.set_ylabel(r"Guessing probability, $P_{\mathrm{guess}}$", size=22)
ax2.set_xlabel(r"Fidelity lower bound, $\delta$", size=22)
plt.minorticks_on()

# Save as PDF (vectorized, with LaTeX fonts)
plt.savefig("Dual_vs_Delta.pdf", format="pdf", bbox_inches="tight")
plt.show()
plt.close()
