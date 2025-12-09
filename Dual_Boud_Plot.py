"""
Calculates and plots the guessing probability for Eve in terms of the lower bound on the fidelity
"""


#%% IMPORT PACKAGES


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Utils as ut

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
    pguess,_ = ut.dual(np.sqrt(delta_aux), behavior)

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
