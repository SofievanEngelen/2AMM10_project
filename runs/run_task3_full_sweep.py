import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

NOTEBOOK = Path("ass1_task3_optimized.ipynb")
SUMMARY_PATH = Path("task3_full_sweep_run_summary.json")
TASK2_MODEL_PATH = Path("saved_models/task2_embedding_model.pth")


def execute_cell(nb, idx, namespace, replacements=None):
    source = "".join(nb["cells"][idx].get("source", []))
    if replacements:
        for old, new in replacements.items():
            if old not in source:
                raise RuntimeError(f"Expected replacement text not found in cell {idx}: {old!r}")
            source = source.replace(old, new)

    started = time.perf_counter()
    print(f"\n===== Executing cell {idx} =====", flush=True)
    exec(compile(source, f"{NOTEBOOK}:cell_{idx}", "exec"), namespace)
    elapsed_minutes = (time.perf_counter() - started) / 60
    print(f"===== Finished cell {idx} in {elapsed_minutes:.2f} min =====", flush=True)
    return elapsed_minutes


def main():
    started = time.perf_counter()
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    cell_minutes = {}

    for idx in [5, 6, 7, 8, 9, 11, 14]:
        cell_minutes[str(idx)] = execute_cell(nb, idx, namespace)

    load_started = time.perf_counter()
    torch = namespace["torch"]
    EmbeddingCNN = namespace["EmbeddingCNN"]
    EMBED_DIM = namespace["EMBED_DIM"]
    device = namespace["device"]

    if not TASK2_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing Task 2 model: {TASK2_MODEL_PATH}")

    namespace["model"] = EmbeddingCNN(embedding_dim=EMBED_DIM).to(device)
    namespace["model"].load_state_dict(torch.load(TASK2_MODEL_PATH, map_location=device))
    namespace["model"].eval()
    cell_minutes["load_task2_model"] = (time.perf_counter() - load_started) / 60
    print(f"Loaded Task 2 model from {TASK2_MODEL_PATH}", flush=True)

    cell_minutes["23"] = execute_cell(
        nb,
        23,
        namespace,
        replacements={
            'TASK3_RUN_MODE = "deadline_safe"': 'TASK3_RUN_MODE = "full_sweep"',
        },
    )

    for idx in [24, 25, 26, 27, 28]:
        cell_minutes[str(idx)] = execute_cell(nb, idx, namespace)

    total_minutes = (time.perf_counter() - started) / 60
    summary = {
        "mode": "full_sweep",
        "total_minutes": total_minutes,
        "cell_minutes": cell_minutes,
        "best_metadata": namespace["best_metadata"],
        "best_result_bundle": namespace["best_result_bundle"],
        "best_training_history": namespace["best_history"],
        "all_results": namespace["task3_all_results"],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {SUMMARY_PATH}", flush=True)
    print(f"Total runtime: {total_minutes:.2f} min", flush=True)


if __name__ == "__main__":
    main()
