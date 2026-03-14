Here is the README content prepared based on the design project document:

# Coupled-Line Bandpass Filter Design Project

## Project Overview

This project involves the design, simulation, and implementation of a 7th-order bandpass filter (BPF). The design progresses from an initial lumped element model to a coupled-line filter using ideal transmission lines, and finally to a physical microstrip implementation.

## Design Specifications

The filter was designed to meet the following performance criteria:
* **Passband:** 7.6 GHz ($f_{L}$) to 8.4 GHz ($f_{H}$).
* **Return Loss (RL):** $\ge$ 22 dB within the passband.
* **Insertion Loss (IL):** $\le$ 0.5 dB within the passband.
* **Stopband Attenuation (Lower):** $IL \ge$ 40 dB for $f \le$ 7.25 GHz.
* **Stopband Attenuation (Upper):** $IL \ge$ 50 dB for $f \ge$ 9.0 GHz.
* **Impedance:** 50&Omega; source and load impedances.

## Project Structure
**code**          → MATLAB scripts for design calculations. 

**ads**           → ADS simulation files and schematics.

**output**        → Performance plots and S-parameter results.

**documentation** → Final project report and design specs.

## Implementation Details

### 1. Lumped Element BPF

* Designed as a 7th-order chebyshev filter ($N=7$) based on hand calculations and MATLAB simulations.
* Simulated over a frequency range of 7 GHz to 9 GHz.

### 2. Coupled-Line BPF (Ideal)

* Converted the lumped element design into a coupled-line configuration.
* Included feed transmission lines with an electrical length of 90° at both the input and output ports.

### 3. Microstrip Implementation

* **Substrate:** Duroid 6010.
* **Dielectric Constant ($\epsilon_{r}$):** 10.7.
* **Substrate Thickness:** 0.635 mm.
* **Loss Tangent (tan $\delta$):** 0.0023.
* **Conductor:** Copper ($t = 18~\mu m$, $\sigma = 5.8 \times 10^{+7} S/m$).

## Tools Used

* **MATLAB:** Used for initial hand calculations, parameter determination (width, spacing, and length of lines), and ideal element simulations.


* **ADS (Advanced Design System):** Utilized for simulating the microstrip coupled-line filter using MCFIL elements and for final tuning (V2) to match performance targets.


## Results Summary
The project concludes with a comparative analysis of the ideal transmission line model versus the tuned microstrip implementation, verified through $|S_{21}|$ and $|S_{11}|$ plots across the 7 GHz to 9 GHz range.

