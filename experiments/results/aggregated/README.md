# DPBench aggregated results

Total cells: 36

## Files

- `master.csv` — every (model, condition) cell with confidence intervals.
- `per_analysis/` — filtered views aligned to paper figures and tables.
- `per_episode/` — one CSV per cell with per-episode outcomes.
- `manifest.yaml` — generation metadata and counts.

## Confidence intervals

- Deadlock rate: Wilson score 95% interval.
- Throughput, fairness: t-based 95% interval (`t.ppf(0.975, df=n-1) * std / sqrt(n)`).
- Time-to-deadlock: not interval-estimated (small subsample).

## Schema

Master CSV columns: source_dir, model, condition_canonical, condition_original, prompt_variant, mode, communication, philosophers, episodes, max_timesteps, temperature, memory_window, communication_rounds, deadlock_terminal, deadlock_rate, deadlock_count, deadlock_ci_low, deadlock_ci_high, avg_throughput, std_throughput, throughput_ci_half, avg_fairness, std_fairness, fairness_ci_half, avg_starvation_count, avg_time_to_deadlock, avg_timesteps, total_deadlock_events, message_action_consistency, total_llm_calls, avg_latency_ms, total_tokens_in, total_tokens_out, total_tokens, runtime_seconds, dpbench_version.

## Regeneration

Run `python experiments/scripts/aggregate.py` from the repository root.
