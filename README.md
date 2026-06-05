# DPBench

**A benchmark for evaluating coordination in multi-agent LLM systems under simultaneous resource contention.**

<p align="center">
  <a href="https://arxiv.org/abs/2602.13255">
    <img src="https://raw.githubusercontent.com/najmulhasan-code/dpbench/main/experiments/figures/teaser.png" alt="DPBench" width="720"/>
  </a>
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2602.13255-b31b1b.svg)](https://arxiv.org/abs/2602.13255)
[![PyPI version](https://img.shields.io/pypi/v/dpbench.svg)](https://pypi.org/project/dpbench/)
[![Downloads](https://static.pepy.tech/badge/dpbench)](https://pepy.tech/project/dpbench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

DPBench adapts the Dining Philosophers problem into a controlled testbed where the action protocol, the communication structure, and the group size each vary independently. Each episode reports four metrics (deadlock rate, throughput, fairness, message-action consistency) with Wilson and *t*-based 95% confidence intervals. On a single model, the benchmark captures what changes when the protocol changes.

Read the full [paper](https://arxiv.org/abs/2602.13255) on arXiv.

## Installation

```bash
pip install dpbench
```

Optional provider extras for the example experiments:

```bash
pip install "dpbench[openai]"
pip install "dpbench[anthropic]"
pip install "dpbench[google]"
pip install "dpbench[xai]"
```

## Quickstart

```python
from dpbench import Benchmark

def my_model(system_prompt: str, user_prompt: str) -> str:
    """Your LLM call here. Returns the agent's response as a string."""
    ...

results = Benchmark.run(
    model_fn=my_model,
    system_prompt="...",
    decision_prompt="...",
    philosophers=5,
    episodes=30,
    mode="simultaneous",
    communication=False,
)

print(f"Deadlock rate: {results['deadlock_rate']:.1%}")
print(f"Throughput:    {results['avg_throughput']:.3f}")
print(f"Fairness:      {results['avg_fairness']:.3f}")
```

Prompt templates used in the paper are in [`experiments/prompts/`](experiments/prompts/). Full parameter documentation is in the `Benchmark.run` docstring.

## Reproducing the paper experiments

```bash
git clone https://github.com/najmulhasan-code/dpbench.git
cd dpbench
pip install -e .

# Provider API keys for the LLMs you want to evaluate
cp .env.example .env

python experiments/scripts/run.py
python experiments/scripts/aggregate.py
python experiments/scripts/generate_figures.py
```

Experiments are configured by [`experiments/configs/conditions.yaml`](experiments/configs/conditions.yaml) and [`experiments/configs/models.yaml`](experiments/configs/models.yaml).

## Citation

If you use DPBench in your work, please cite:

```bibtex
@misc{hasan2026dpbenchstructuraldeterminantsmultiagent,
      title={DPBench: Structural Determinants of Multi-Agent LLM Coordination Under Simultaneous Resource Contention},
      author={Najmul Hasan and Prashanth BusiReddyGari},
      year={2026},
      eprint={2602.13255},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.13255},
}
```

## License

Licensed under the MIT License.
