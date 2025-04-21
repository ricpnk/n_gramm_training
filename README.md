
# N-Gram Language Modeling

This project implements a collection of count-based language models using the Penn Treebank dataset. It includes Unigram and Bigram models, with and without Laplace smoothing, and allows for evaluation using sentence log-probability and perplexity.

## Features

- Unigram model with optional `<unk>` token replacement
- Bigram model with:
  - raw frequency-based probabilities
  - Laplace-smoothing support
- Sentence log-probability computation
- Perplexity calculation for test datasets
- Token cleanup using frequency threshold
- Optional: Byte-Pair Encoding (BPE) (see below)

## Project Structure

- `n-gram.py`: main logic and execution entry point
- `pyproject.toml`: project metadata and uv-compatible dependencies
- `README.md`: this file

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for lightweight Python environment management.


### 1. Install dependencies
```bash
uv venv
uv sync
```

uv will resolve packages from pyproject.toml.

### 2. Run the project
```bash
uv run n-gram.py
```
This will output log-probabilities and perplexities for each model.

## How It Works

### 1. Preprocessing
- Loads the Penn Treebank dataset
- Adds <stop> tokens and removes sentences shorter than tokens
- Splits data into train/test sets
- Optionally replaces rare words with <unk>

### 2. Language Models
- Unigram: word-level frequency
- Bigram: context-aware with P(w2 | w1)
- Smoothing: Laplace-based adjustments using configurable alpha

### 3. Evaluation
- Sentence log-probability
- Test-set perplexity

### 4. Optional Extension (Not Required)

You may implement Byte-Pair Encoding (BPE) as a class with train() and apply() methods to segment words using subword units.

## Output Examples

Example output during execution:

Unigram Model:
logprob of 'the the the <stop>': -17.052292292248378
logprob of 'ilove computer science <stop>': inf
Perplexity of Testset: 694.6988206842844
Perplexity of Clean-Testset: 360.0584646186809

Bigram Model:
logprob for bigram of 'the the the <stop>': -37.26824939975674
logprob for bigram of 'ilove computer science <stop>': inf
Perplexity of Bigram-model with Testset: 22.149080717090335
The Bigram-model has: 7609 sentences with a probability of zero

Smoothed Bigram Model:
Perplexity of Smoothed-Bigram-model with Testset: 246.71472890788124
The Smoothed-Bigram-model has: 2359 sentences with a probability of zero


