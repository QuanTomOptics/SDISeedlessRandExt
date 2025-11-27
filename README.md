# SDISeedlessRandExt
This repository contains the relevant codes for the manuscript titled "Deterministic randomness extraction for semi-device-independent quantum random number generation". It is structured as follows:

1. **`Delta_Comparison_Plot`** - Computes and plots the single-round guessing probability $P\_{{\rm guess}}(A|E,X=0,\mathbf{p}\_\delta,\delta)$ defined in Eq. (3) in terms of the behaviors $\mathbf{p}\_\delta$ defined in Eq. (17). This code is used to generate the plot presented in Figure 2 in the manuscript.
2. **`Mbit_Rates.py`** - Computes the expected rate $\overline{\mathrm{rate}}\_\mathrm{mul}(n)$ and the expected asymptotic rate $\overline{\mathrm{rate}}^\infty\_\mathrm{mul}$, as defined in Eq. (18) and Eq. (19), respectively, for any noisy behavior $p_\delta^\mathrm{noisy}(\mathbf{a}|\mathbf{x})$ defined in Eq. (20).
3. **`Delta_Comparison_Plot`** - Plots the rates obtained from **Mbit_Rates.py** in terms of $n$ for different values of the lower bound on the fidelity, $\delta$. This code is used to generate the plot presented in Figure 3 in the manuscript.
4. **`Noisy_Comparison_Plot`** - Plots the rates obtained from **Mbit_Rates.py** in terms of the noise probability, $p$, for different values of $n$. This code is used to generate the plot presented in Figure 4 in the manuscript.


## Library Requirements

- `matplotlib` 3.7.5
- `mosek` 10.0.43
- `numpy` 1.24.4
- `pandas` 2.0.3
- `PICOS` 2.6.2 
- `scipy` 1.10.1
- `seaborn` 0.13.2
- `tqdm` 4.66.4
