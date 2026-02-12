import json
from pathlib import Path


def main() -> None:
    data = json.loads(Path("metrics.json").read_text())

    users = data.get("meta", {}).get("users_evaluated", "unknown")
    k = data.get("meta", {}).get("k", "unknown")

    pop_p = data["popularity"]["precision@k"]
    pop_r = data["popularity"]["recall@k"]
    rnd_p = data["random"]["precision@k"]
    rnd_r = data["random"]["recall@k"]

    md = f"""# Evaluation Results

- Users evaluated: **{users}**
- K: **{k}**

## Baselines

| Baseline | Precision@K | Recall@K |
|---|---:|---:|
| Popularity | {pop_p:.6f} | {pop_r:.6f} |
| Random | {rnd_p:.6f} | {rnd_r:.6f} |

## Notes
- Split: per-user holdout (leave-one-out)
- Candidates: global train universe, excluding items seen in train per user
- Random baseline is deterministic via seed
"""

    Path("metrics.md").write_text(md)
    print("Wrote metrics.md")


if __name__ == "__main__":
    main()
