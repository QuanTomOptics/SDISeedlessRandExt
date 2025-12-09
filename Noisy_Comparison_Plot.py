"""
Loads results data from xlsx files and plots the rates comparison for different values of the noise
"""


#%% IMPORT PACKAGES


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
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


#%% LOAD DATA
 

# Load Excel files
df = pd.read_excel("Noisy_Results_Finite.xlsx", sheet_name="Sheet1", header=0)
df2 = pd.read_excel("Noisy_Results_Asymptotic.xlsx", sheet_name="Sheet1",header=0)

# Ensure numeric types
df["n"] = pd.to_numeric(df["n"], errors="coerce")
df["Avg Reff"] = pd.to_numeric(df["Avg Reff"], errors="coerce")
df["Dev.1"] = pd.to_numeric(df["Dev.1"], errors="coerce")
df["Noise probability"] = pd.to_numeric(df["Noise probability"], errors="coerce")
df2["Reff"] = pd.to_numeric(df2["Reff"], errors="coerce")
df2["Noise probability"] = pd.to_numeric(df2["Noise probability"], errors="coerce")

# Drop empty rows
df = df.dropna(subset=["n", "Avg Reff", "Dev.1", "Noise probability"])
df2 = df2.dropna(subset=["Reff", "Noise probability"])

# Identify blocks for finite rates; We'll assume each block starts where 'n' decreases or repeats previous starting values
blocks = []
start_idx = 0
for i in range(1, len(df)):
    if df["Noise probability"].iloc[i] <= df["Noise probability"].iloc[i-1]:
        blocks.append(df.iloc[start_idx:i])
        start_idx = i
blocks.append(df.iloc[start_idx:])  # last block


#%% PLOT


# Main plot
fig, ax = plt.subplots(figsize=(9,6))

# Select colors
colors = sns.color_palette("Set2", 5)

# Extract x-axis values
noise_vals = (df2["Noise probability"].values * 100)

# Extract asymptotic rate
asymptotic_rate = (df2["Reff"].values)

for i, block in enumerate(blocks[:5]):  # Take first five blocks
    n_vals = block["n"].values
    avg_reff = block["Avg Reff"].values
    dev_reff = block["Dev.1"].values
    
    # Plot finite rates with standard deviation
    ax.plot(noise_vals, avg_reff, label=f"$n$ = {ut.sci_label(n_vals[0])}", color=colors[i], linewidth=2)
    ax.fill_between(noise_vals, avg_reff - dev_reff, avg_reff + dev_reff, color=colors[i], alpha=0.3)

# Plot asymptotic rate
ax.plot(noise_vals, asymptotic_rate, label=r"asymptotic", color="black", linewidth=2, linestyle='--')

# Grid & labels
# X-axis: major step = 0.02, so minor step = 0.02/4 = 0.005
ax.xaxis.set_minor_locator(MultipleLocator(0.005))
# Y-axis: major step = 0.004, so minor step = 0.004/4 = 0.001
ax.yaxis.set_minor_locator(MultipleLocator(0.001))
ax.grid(True, which="major", linestyle='-', linewidth=0.8, alpha=0.7)
ax.grid(True, which="minor", linestyle=':', linewidth=0.5, alpha=0.5)
ax.set_xlabel(r"Noise rate, $\gamma$ in \%", fontsize=22)
ax.set_ylabel(r"Expected rate, $\overline{\mathrm{rate}}_\mathrm{mul}$", fontsize=22)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1],loc="lower left", fontsize=18)

# Fix plot ticks
ticks = [0, 0.02, 0.04, 0.06, 0.08, 0.1]
ax.set_xticks(ticks)
ax.set_xticklabels([r"$0$", r"$0.02$", r"$0.04$", r"$0.06$", r"$0.08$", r"$0.1$"], fontsize=16)
ticks2 = [0.004,0.008,0.012,0.016]
ax.set_yticks(ticks2)
ax.set_yticklabels([r"$0.004$",r"$0.008$",r"$0.012$",r"$0.016$"],fontsize=16)

plt.tight_layout()
plt.savefig("Noisy_Comparison.pdf", format="pdf", bbox_inches="tight")
plt.show()
