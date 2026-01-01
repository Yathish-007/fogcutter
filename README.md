# Fogcutter 

**Fogcutter** is a Python library for cutting through the uncertainty of Large Language Models (LLMs). It implements state-of-the-art quantification metrics from 2022-2025 research.

## Features
- **White-Box (Logits):** 
  - Token Probability & Entropy
  - Perplexity
  - LogU (Chen et al., 2025)
- **Black-Box (Sampling):** 
  - Self-Consistency (Wang et al., 2022)
  - Semantic Entropy (Kuhn et al., 2023)
- **Verbalized:** 
  - Explicit Scoring & Epistemic Markers

## Installation
bash
pip install fogcutter

## Quick Start
python
import fogcutter.whitebox as white
# Calculate entropy from logits
entropy = white.token_entropy(logits)

