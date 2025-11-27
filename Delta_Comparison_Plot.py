"""
Loads results data from xlsx files and plots the rates comparison for different values of delta.
"""


#%% IMPORT PACKAGES


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LogLocator, NullFormatter

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
df = pd.read_excel("Delta_Results_Finite.xlsx", sheet_name="Sheet1", header=0)
df2 = pd.read_excel("Delta_Results_Asymptotic.xlsx", sheet_name="Sheet1",header=0)

# Ensure numeric types
df["n"] = pd.to_numeric(df["n"], errors="coerce")
df["Avg Reff"] = pd.to_numeric(df["Avg Reff"], errors="coerce")
df["Dev"] = pd.to_numeric(df["Dev"], errors="coerce")
df2["Reff"] = pd.to_numeric(df2["Reff"], errors="coerce")
df2["Delta"] = pd.to_numeric(df2["Delta"], errors="coerce")

# Drop empty rows
df = df.dropna(subset=["n", "Avg Reff", "Dev"])
df2 = df2.dropna(subset=["Reff", "Delta"])

# Identify blocks for finite rates; We'll assume each block starts where 'n' decreases or repeats previous starting values
blocks = []
start_idx = 0
for i in range(1, len(df)):
    if df["n"].iloc[i] <= df["n"].iloc[i-1]:
        blocks.append(df.iloc[start_idx:i])
        start_idx = i
blocks.append(df.iloc[start_idx:])  # last block


#%% PLOT


# Main plot
fig, ax = plt.subplots(figsize=(9,6))

# Select colors
colors = sns.color_palette("Set2", 4)


for i, block in enumerate(blocks[:4]):  # Take first four blocks
    n_vals = block["n"].values
    avg_reff = block["Avg Reff"].values
    dev_reff = block["Dev1"].values
    asym_reff = df2.iloc[i]["Reff"]
    delta = df2.iloc[i]["Delta"]
    
    # Plot finite rates with standard deviation + asymptotic rates
    ax.plot(n_vals, avg_reff, label=f"$\delta = {delta:.2f}$", color=colors[3-i], linewidth=2)
    ax.fill_between(n_vals, avg_reff - dev_reff, avg_reff + dev_reff, color=colors[3-i], alpha=0.3)
    ax.axhline(y=asym_reff, color=colors[3-i], linestyle='--', linewidth=1.5)

# Find minimum n and plot a vertical line
min_n = min(block["n"].min() for block in blocks)
ax.axvline(x=min_n, color="black", linestyle=":", linewidth=1.5, label=f"$n={7e3:.0f}$")

# Log scale, grid & labels
ax.set_xscale("log")
ax.xaxis.set_minor_locator(LogLocator(base=10, subs='auto', numticks=100))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.grid(True, which="major", linestyle='-', linewidth=0.8, alpha=0.7)
ax.grid(True, which="minor", linestyle=':', linewidth=0.5, alpha=0.5)
ax.set_xlabel(r"Number of rounds, $n$", fontsize=22)
ax.set_ylabel(r"Expected rate, $\overline{\mathrm{rate}}_\mathrm{mul}$", fontsize=22)
ax.legend(loc="upper left", fontsize=18)

# Fix plot ticks
ticks = [1e4,1e5,1e6,1e7,1e8]
ax.set_xticks(ticks)
ax.set_xticklabels([r"$1\times10^4$", r"$1\times10^5$", r"$1\times10^6$", r"$1\times10^7$", r"$1\times10^8$"], fontsize=16)
ticks2 = [0.000,0.002,0.004,0.006,0.008,0.010,0.012,0.014,0.016]
ax.set_yticks(ticks2)
ax.set_yticklabels([r"$0.000$",r"$0.002$", r"$0.004$", r"$0.006$", r"$0.008$", r"$0.010$", r"$0.012$", r"$0.014$", r"$0.016$"],fontsize=16)

plt.tight_layout()
plt.savefig("Delta_Comparison.pdf", format="pdf", bbox_inches="tight")
plt.show()

