# Egalitarian Gradient Descent (EGD)

This repository contains code to reproduce the experimental results from the paper:

**_Egalitarian Gradient Descent: A Simple Approach to Accelerated Grokking_**  
Accepted at **ICLR 2026**

---

## Repository Structure

### `main-results/`
Code to reproduce the **main experimental results** in the paper.

This folder includes implementations and experiments for:
- **EGD**
- **EGD + Randomized SVD (RSVD)**
- **Column Normalization**
- **Vanilla SGD**

Tasks covered:
- Modulo addition  
- Modulo multiplication  
- Parity  

Each experiment is configured to match the settings reported in the paper.

---

### `grokfast-comparison/`
Code to reproduce the **comparison experiments between Grokfast and EGD**.

- Includes runnable `.sh` scripts demonstrating how to launch the experiments.
- The core Grokfast implementation is adapted from the official repository:

> **Grokfast: Accelerated Grokking by Amplifying Slow Gradients**  
> https://github.com/ironjr/grokfast  
> (cloned on September 10, 2025)

Modifications were made only as necessary to ensure a fair and consistent comparison with EGD.

---

## Notes
- All experiments are designed for reproducibility.
- Hyperparameters and settings follow those described in the paper unless stated otherwise.
