# Matrix Multiplication Performance Benchmark for AI API Cost Simulation

## Objective

This project demonstrates the application of matrix multiplication and numerical computing techniques to model AI API resource consumption in a higher education environment.

A synthetic dataset representing 2,000 students is generated using reproducible probabilistic simulations. The notebook compares multiple matrix multiplication algorithms, evaluates their computational performance, and applies matrix-based operations to estimate AI API usage and associated institutional costs.

The experiment integrates mathematical foundations, algorithm implementation, performance benchmarking, and scientific visualization to illustrate how numerical methods support large-scale computational analysis.

---

## Business Problem & Impact

Universities are increasingly adopting Large Language Models (LLMs) for teaching, research, and administrative workflows. As AI adoption grows across thousands of students and multiple academic departments, monitoring token consumption and controlling API costs becomes critical.

Institutional IT teams require efficient computational methods to:

- **Estimate AI API Costs:** Project expenditures across massive student populations.
- **Analyze Consumption:** Pinpoint power users and track token allocation.
- **Compare Pricing Strategies:** Evaluate costs dynamically across shifting provider rates.
- **Support Data-Driven Budgeting:** Forecast infrastructure demand using matrix equations.

### Potential Applications

- Estimating institution-wide AI API expenditures.
- Evaluating pricing scenarios across multiple LLM providers (e.g., ChatGPT, Gemini, Claude).
- Supporting budget forecasting through reproducible probabilistic simulation.

---

## Data Science Approach & Workflow

The analysis flows from synthetic data formulation to algorithmic benchmarking:

1. **Probabilistic Simulation:** Generating student usage metrics via heavy-tailed distributions.
2. **Tabular-to-Matrix Mapping:** Transforming structured Pandas data into optimized NumPy array representations.
3. **From-Scratch Algorithmic Implementation:** Writing 3-loop Naive and 6-loop Cache-Aware Tiled algorithms from first principles.
4. **Hardware Vectorization Analysis:** Benchmarking custom loops against highly optimized NumPy vector engines.
5. **Scientific Visualization:** Visualizing cost structures and execution scaling.

---

## Project Structure

```text
Matrix_Multiplication/
├── output/
│   ├── ai_api_performance_dashboard.png
│   ├── benchmark_scaling.png
│   └── institutional_costs.png
├── LICENSE
├── README.md
├── dashboard.py
├── matrix_multiplication_benchmark.ipynb
└── requirements.txt
```

---

## Tech Stack & Core Concepts

### Tech Stack

- **Programming Language:** Python 3.12+
- **Core Libraries:** NumPy, Pandas, Matplotlib
- **Execution Profiling:** `time.perf_counter`

### Numerical Computing Concepts

- **Linear Algebra Foundations:** Matrix projections and tensor dimension transformations.
- **Vectorization & SIMD:** Single Instruction, Multiple Data (SIMD) operations utilizing underlying hardware.
- **Probabilistic Distributions:** Modeling user spikes via log-normal statistics.
- **Computational Complexity:** Empirical tracking of asymptotic (O(N^3)$) behavior.
- **Cache-Aware Optimization:** Spatial memory locality and CPU cache optimization.

---

## Mathematical & Algorithmic Design

### The Linear Algebra Engine

Given $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$, their matrix product is $C = AB$, where each element is defined as: $$c_{ij} = \sum_{k=1}^{n} a_{ik}b_{kj}$$ The resulting matrix has dimensions $C \in \mathbb{R}^{m \times p}$.

### Core Algorithmic Implementations

- **Naive Algorithm (O(N^3)$)** – Standard three-loop implementation.
- **Tiled Algorithm (O(N^3)$)\n** – Cache-aware matrix blocking to improve memory locality.
- **NumPy Matrix Engine** – Highly optimized BLAS/LAPACK-backed vectorized implementation.

---

## Performance Benchmarking & Insights

### Quantitative Performance Metrics

The system was benchmarked across increasing square matrix dimensions:

| Matrix Size | Naive (s) | Tiled (s) | NumPy (s) | Naive Speedup | Tiled Speedup |
|------------:|----------:|----------:|----------:|--------------:|--------------:|
| 10 | 0.0007 | 0.0011 | 0.0001 | 7.8464 | 11.5618 |
| 30 | 0.0159 | 0.0213 | 0.0002 | 74.1024 | 99.0475 |
| 60 | 0.1579 | 0.1797 | 0.0004 | 383.6596 | 436.5858 |
| 100 | 0.7393 | 0.8243 | 0.0005 | 1412.9930 | 1575.4841 |
| 150 | 2.4113 | 2.6894 | 0.0009 | 2726.1553 | 3040.5449 |
| 220 | 7.7118 | 8.1557 | 0.0013 | 6105.9250 | 6457.3730 |

### Technical Note on the Python Tiling Paradox

As observed for matrix dimensions \(N \ge 150\), the tiled implementation executes slightly slower than the naive implementation.

Although matrix tiling significantly reduces cache misses in compiled languages such as C and C++, implementing the algorithm in pure Python expands the execution from **three nested loops to six nested loops**. The additional interpreter overhead outweighs the cache locality benefits, causing the tiled implementation to underperform.

A compiled implementation using C, Cython, or Numba would be expected to demonstrate the performance advantages typically associated with cache-aware tiling.

---

## Setup & Execution

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Project

Launch the Jupyter notebook:

```bash
jupyter notebook matrix_multiplication_benchmark.ipynb
```

The notebook will:

- Generate the synthetic dataset.
- Execute all three matrix multiplication algorithms.
- Benchmark execution times.
- Compute AI API cost projections.
- Generate visualization dashboards.

---

## License

This project is released under the MIT License.

Ethical Use Notice: This project is intended for educational, research, and professional learning purposes. Misrepresentation of authorship or deceptive use is unethical. Attribution is required under the MIT License.
