import copy
import gc
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import kagglehub
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

os.environ.setdefault("MPLBACKEND", "Agg")

NOTEBOOK = Path("ass1_task3_optimized.ipynb")
TASK2_MODEL_PATH = Path("saved_models/task2_embedding_model.pth")
TASK3_SAVE_PATH = Path("saved_models/task3_best_joint_embedding_model.pth")
TASK3_FULL_SWEEP_SAVE_PATH = Path("saved_models/task3_best_joint_embedding_model_full_sweep.pth")
SUMMARY_PATH = Path("task3_full_sweep_fast_run_summary.json")

TASK3_SEED = 6
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 1024
KNN_QUERY_BATCH_SIZE = 1024
IMAGE_SIZE = 64
EMBED_DIM = 128
TASK3_ALPHA_VALUES = [0.25, 0.5, 1.0]
TASK3_K_VALUES = [1, 3, 5]
TASK3_EPOCHS = 15


def log(message=""):
    print(message, flush=True)


def timed(label, timings, fn):
    started = time.perf_counter()
    log(f"\n===== {label} =====")
    result = fn()
    minutes = (time.perf_counter() - started) / 60
    timings[label] = minutes
    log(f"===== Finished {label} in {minutes:.2f} min =====")
    return result


def load_notebook_definitions():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {
        "__name__": "__main__",
        "os": os,
        "re": re,
        "Path": Path,
        "Dataset": Dataset,
        "Image": Image,
        "kagglehub": kagglehub,
        "torch": torch,
        "np": np,
        "defaultdict": defaultdict,
        "nn": nn,
        "F": F,
    }

    for idx in [8, 9]:
        source = "".join(nb["cells"][idx].get("source", []))
        exec(compile(source, f"{NOTEBOOK}:cell_{idx}", "exec"), namespace)

    garden_source = "".join(nb["cells"][14].get("source", []))
    garden_source = garden_source.split("\ndataset = GardenDataset()", 1)[0]
    exec(compile(garden_source, f"{NOTEBOOK}:cell_14_class_only", "exec"), namespace)
    return namespace["EmbeddingCNN"], namespace["batch_hard_triplet_loss"], namespace["GardenDataset"]


def require_cuda():
    log(torch.__version__)
    cuda_available = torch.cuda.is_available()
    log(cuda_available)
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "No GPU"
    log(gpu_name)
    if not cuda_available:
        raise RuntimeError("Full sweep stopped because CUDA is not available.")
    return gpu_name


def set_seed(seed=TASK3_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preload_dataset(dataset, device, name):
    n = len(dataset)
    images_cpu = torch.empty((n, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
    family_labels = torch.empty(n, dtype=torch.long)
    item_labels = torch.empty(n, dtype=torch.long)
    item_names = []
    family_names = []

    for i in range(n):
        image, family_label, item_label = dataset[i]
        images_cpu[i].copy_(image)
        family_labels[i] = int(family_label)
        item_labels[i] = int(item_label)
        item_name = Path(dataset.image_paths[i]).parent.name
        item_names.append(item_name)
        family_names.append(dataset.item_to_family[item_name])
        if (i + 1) % 5000 == 0 or i + 1 == n:
            log(f"{name}: preloaded {i + 1}/{n} images")

    images_gpu = images_cpu.to(device, non_blocking=True)
    del images_cpu
    gc.collect()

    return {
        "images": images_gpu,
        "family_labels": family_labels.to(device, non_blocking=True),
        "item_labels": item_labels.to(device, non_blocking=True),
        "item_names": item_names,
        "family_names": family_names,
        "paths": [str(path) for path in dataset.image_paths],
        "dataset": dataset,
    }


def make_mask(data, *, family_subset, item_subset):
    new_families = set(data["dataset"].new_families)
    new_items = set(data["dataset"].new_items)
    mask = []
    for item_name, family_name in zip(data["item_names"], data["family_names"]):
        if family_subset == "main" and family_name in new_families:
            mask.append(False)
            continue
        if family_subset == "new" and family_name not in new_families:
            mask.append(False)
            continue
        if item_subset == "main" and item_name in new_items:
            mask.append(False)
            continue
        if item_subset == "new" and item_name not in new_items:
            mask.append(False)
            continue
        mask.append(True)
    return torch.tensor(mask, dtype=torch.bool)


def mask_to_indices(mask, device):
    return torch.nonzero(mask, as_tuple=False).flatten().to(device)


def summarize_labels(name, labels):
    labels_cpu = labels.detach().cpu().tolist()
    counts = Counter(labels_cpu)
    log(
        f"{name}: {len(labels_cpu)} images, {len(counts)} labels, "
        f"min/max per label {min(counts.values())}/{max(counts.values())}"
    )


def build_scenarios(train_data, test_data, device):
    train_main_main = mask_to_indices(make_mask(train_data, family_subset="main", item_subset="main"), device)
    train_main_all = mask_to_indices(make_mask(train_data, family_subset="main", item_subset="all"), device)
    train_all_main = mask_to_indices(make_mask(train_data, family_subset="all", item_subset="main"), device)
    test_main_main = mask_to_indices(make_mask(test_data, family_subset="main", item_subset="main"), device)
    test_main_new = mask_to_indices(make_mask(test_data, family_subset="main", item_subset="new"), device)
    test_new_main = mask_to_indices(make_mask(test_data, family_subset="new", item_subset="main"), device)

    scenarios = [
        {
            "name": "scenario_1_items_main",
            "classification_level": "Items",
            "support_source": "train",
            "support_indices": train_main_main,
            "support_label_key": "item_labels",
            "test_source": "test",
            "test_indices": test_main_main,
            "test_label_key": "item_labels",
            "task2_key": "scenario_1_items_main",
        },
        {
            "name": "scenario_2_families_main",
            "classification_level": "Families",
            "support_source": "train",
            "support_indices": train_main_main,
            "support_label_key": "family_labels",
            "test_source": "test",
            "test_indices": test_main_main,
            "test_label_key": "family_labels",
            "task2_key": "scenario_2_families_main",
        },
        {
            "name": "scenario_3_new_items",
            "classification_level": "Items",
            "support_source": "train",
            "support_indices": train_main_all,
            "support_label_key": "item_labels",
            "test_source": "test",
            "test_indices": test_main_new,
            "test_label_key": "item_labels",
            "task2_key": "scenario_3_new_items",
        },
        {
            "name": "scenario_4_new_families",
            "classification_level": "Families",
            "support_source": "train",
            "support_indices": train_all_main,
            "support_label_key": "family_labels",
            "test_source": "test",
            "test_indices": test_new_main,
            "test_label_key": "family_labels",
            "task2_key": "scenario_4_new_families",
        },
    ]

    for scenario in scenarios:
        support_data = train_data if scenario["support_source"] == "train" else test_data
        test_source_data = train_data if scenario["test_source"] == "train" else test_data
        support_labels = support_data[scenario["support_label_key"]][scenario["support_indices"]]
        test_labels = test_source_data[scenario["test_label_key"]][scenario["test_indices"]]
        summarize_labels(f"{scenario['name']} support", support_labels)
        summarize_labels(f"{scenario['name']} test", test_labels)

    return scenarios


def train_joint_embedding_model(EmbeddingCNN, triplet_loss, train_data, train_indices, alpha, device):
    set_seed()
    model = EmbeddingCNN(embedding_dim=EMBED_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    n = int(train_indices.numel())
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE

    for epoch in range(TASK3_EPOCHS):
        model.train()
        permutation = train_indices[torch.randperm(n, device=device)]
        running_total = torch.zeros((), device=device)
        running_item = torch.zeros((), device=device)
        running_family = torch.zeros((), device=device)

        for start in range(0, n, BATCH_SIZE):
            batch_indices = permutation[start:start + BATCH_SIZE]
            images = train_data["images"][batch_indices]
            family_labels = train_data["family_labels"][batch_indices]
            item_labels = train_data["item_labels"][batch_indices]

            embeddings = model(images)
            item_loss = triplet_loss(embeddings, item_labels, margin=0.3)
            family_loss = triplet_loss(embeddings, family_labels, margin=0.3)
            loss = item_loss + alpha * family_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_total += loss.detach()
            running_item += item_loss.detach()
            running_family += family_loss.detach()

        epoch_result = {
            "epoch": epoch + 1,
            "total_loss": (running_total / n_batches).item(),
            "item_loss": (running_item / n_batches).item(),
            "family_loss": (running_family / n_batches).item(),
        }
        history.append(epoch_result)
        log(
            f"alpha={alpha:.2f} | epoch [{epoch + 1}/{TASK3_EPOCHS}] | "
            f"loss={epoch_result['total_loss']:.4f} | "
            f"item={epoch_result['item_loss']:.4f} | "
            f"family={epoch_result['family_loss']:.4f}"
        )

    return model, history


@torch.inference_mode()
def embed_images(model, images):
    model.eval()
    parts = []
    for start in range(0, len(images), EVAL_BATCH_SIZE):
        parts.append(model(images[start:start + EVAL_BATCH_SIZE]).detach())
    return torch.cat(parts, dim=0)


def majority_vote_rows(neighbor_labels_cpu, k):
    if k == 1:
        return neighbor_labels_cpu[:, 0]

    predictions = []
    for row in neighbor_labels_cpu[:, :k]:
        values, counts = torch.unique(row, return_counts=True)
        predictions.append(values[counts.argmax()])
    return torch.stack(predictions)


@torch.inference_mode()
def knn_accuracies(support_emb, support_labels, test_emb, test_labels, k_values):
    max_k = max(k_values)
    neighbor_label_parts = []
    support_labels = support_labels.to(support_emb.device)

    for start in range(0, len(test_emb), KNN_QUERY_BATCH_SIZE):
        query = test_emb[start:start + KNN_QUERY_BATCH_SIZE]
        similarities = query @ support_emb.T
        neighbor_indices = similarities.topk(max_k, largest=True).indices
        neighbor_label_parts.append(support_labels[neighbor_indices].detach().cpu())

    neighbor_labels_cpu = torch.cat(neighbor_label_parts, dim=0)
    test_labels_cpu = test_labels.detach().cpu()

    accuracies = {}
    for k in k_values:
        predictions = majority_vote_rows(neighbor_labels_cpu, k)
        accuracies[k] = (predictions == test_labels_cpu).float().mean().item()
    return accuracies


def evaluate_model(model, train_data, test_data, scenarios, k_values, model_name, alpha, epochs, run_mode):
    train_emb = embed_images(model, train_data["images"])
    test_emb = embed_images(model, test_data["images"])
    results = []

    for scenario in scenarios:
        support_data = train_data if scenario["support_source"] == "train" else test_data
        test_source_data = train_data if scenario["test_source"] == "train" else test_data
        source_emb = train_emb if scenario["support_source"] == "train" else test_emb
        query_emb = train_emb if scenario["test_source"] == "train" else test_emb

        support_indices = scenario["support_indices"]
        test_indices = scenario["test_indices"]
        support_labels = support_data[scenario["support_label_key"]][support_indices]
        test_labels = test_source_data[scenario["test_label_key"]][test_indices]

        accuracies = knn_accuracies(
            source_emb[support_indices],
            support_labels,
            query_emb[test_indices],
            test_labels,
            k_values,
        )

        for k, accuracy in accuracies.items():
            results.append({
                "model_name": model_name,
                "run_mode": run_mode,
                "alpha": alpha,
                "epochs": epochs,
                "scenario": scenario["name"],
                "classification_level": scenario["classification_level"],
                "task2_key": scenario["task2_key"],
                "n_support": int(support_indices.numel()),
                "n_test": int(test_indices.numel()),
                "accuracy": accuracy,
                "k": k,
            })

    del train_emb, test_emb
    torch.cuda.empty_cache()
    return results


def selection_score(result_bundle):
    by_scenario = {result["scenario"]: result["accuracy"] for result in result_bundle}
    return (
        by_scenario["scenario_2_families_main"] + by_scenario["scenario_4_new_families"],
        by_scenario["scenario_1_items_main"] + by_scenario["scenario_3_new_items"],
        min(by_scenario.values()),
    )


def print_results(results, title):
    log(title)
    header = f"{'model':<18} {'mode':<16} {'alpha':>7} {'k':>3} {'scenario':<28} {'level':<9} {'accuracy':>10} {'n_test':>8}"
    log(header)
    log("-" * len(header))
    for result in results:
        alpha = "-" if result.get("alpha") is None else f"{result['alpha']:.2f}"
        log(
            f"{result['model_name']:<18} "
            f"{result['run_mode']:<16} "
            f"{alpha:>7} "
            f"{result['k']:>3} "
            f"{result['scenario']:<28} "
            f"{result['classification_level']:<9} "
            f"{100 * result['accuracy']:>9.2f}% "
            f"{result['n_test']:>8}"
        )


def cpu_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def main():
    total_started = time.perf_counter()
    timings = {}
    run_mode = "full_sweep_fast"

    gpu_name = timed("cuda_check", timings, require_cuda)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_seed()

    EmbeddingCNN, triplet_loss, GardenDataset = timed("load_notebook_definitions", timings, load_notebook_definitions)

    device = torch.device("cuda")
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    def make_datasets():
        train_all = GardenDataset(class_level="both", transform=transform, subset="train", family_subset="all", item_subset="all")
        test_all = GardenDataset(class_level="both", transform=transform, subset="test", family_subset="all", item_subset="all")
        return train_all, test_all

    train_all_dataset, test_all_dataset = timed("instantiate_datasets", timings, make_datasets)
    train_data = timed("preload_train_all_to_gpu", timings, lambda: preload_dataset(train_all_dataset, device, "train_all"))
    test_data = timed("preload_test_all_to_gpu", timings, lambda: preload_dataset(test_all_dataset, device, "test_all"))
    scenarios = timed("build_scenarios", timings, lambda: build_scenarios(train_data, test_data, device))
    train_indices = mask_to_indices(make_mask(train_data, family_subset="main", item_subset="main"), device)

    all_results = []

    def evaluate_baseline():
        model = EmbeddingCNN(embedding_dim=EMBED_DIM).to(device)
        model.load_state_dict(torch.load(TASK2_MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
        return evaluate_model(model, train_data, test_data, scenarios, TASK3_K_VALUES, "task2_baseline", None, 0, run_mode)

    baseline_results = timed("evaluate_task2_baseline_fast", timings, evaluate_baseline)
    all_results.extend(baseline_results)
    print_results(baseline_results, "Task 2 baseline evaluated with optimized full sweep")

    best_result_bundle = None
    best_score = None
    best_state_dict = None
    best_history = None
    best_metadata = None
    histories = {}

    for alpha in TASK3_ALPHA_VALUES:
        def train_and_eval_alpha():
            candidate_model, candidate_history = train_joint_embedding_model(
                EmbeddingCNN,
                triplet_loss,
                train_data,
                train_indices,
                alpha,
                device,
            )
            candidate_results = evaluate_model(
                candidate_model,
                train_data,
                test_data,
                scenarios,
                TASK3_K_VALUES,
                "joint_triplet",
                alpha,
                TASK3_EPOCHS,
                run_mode,
            )
            candidate_state = cpu_state_dict(candidate_model)
            del candidate_model
            torch.cuda.empty_cache()
            return candidate_results, candidate_history, candidate_state

        candidate_results, candidate_history, candidate_state = timed(
            f"train_eval_alpha_{alpha:.2f}",
            timings,
            train_and_eval_alpha,
        )
        histories[str(alpha)] = candidate_history
        all_results.extend(candidate_results)
        print_results(candidate_results, f"Candidate results alpha={alpha:.2f}")

        for k in TASK3_K_VALUES:
            candidate_bundle = [result for result in candidate_results if result["k"] == k]
            candidate_score = selection_score(candidate_bundle)
            if best_score is None or candidate_score > best_score:
                best_score = candidate_score
                best_result_bundle = copy.deepcopy(candidate_bundle)
                best_state_dict = copy.deepcopy(candidate_state)
                best_history = copy.deepcopy(candidate_history)
                best_metadata = {
                    "run_mode": run_mode,
                    "alpha": alpha,
                    "k": k,
                    "epochs": TASK3_EPOCHS,
                    "gpu": gpu_name,
                    "selection_score": best_score,
                    "optimization": "preloaded GPU tensors, one train/test embedding pass per model, one topk(max_k) per scenario",
                }

    print_results(all_results, "All optimized full sweep results")
    print_results(best_result_bundle, "Best optimized full sweep result bundle")

    checkpoint = {
        "model_state_dict": best_state_dict,
        "best_results": best_result_bundle,
        "best_metadata": best_metadata,
        "training_history": best_history,
        "all_training_histories": histories,
        "task2_reference_results": {
            "scenario_1_items_main": None,
            "scenario_2_families_main": None,
            "scenario_3_new_items": None,
            "scenario_4_new_families": None,
        },
        "selection_rule": "maximize scenarios 2+4, then scenarios 1+3, then worst scenario",
        "run_configs": {
            "full_sweep_fast": {
                "alpha_values": TASK3_ALPHA_VALUES,
                "k_values": TASK3_K_VALUES,
                "epochs": TASK3_EPOCHS,
            }
        },
        "seed": TASK3_SEED,
    }
    TASK3_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, TASK3_SAVE_PATH)
    torch.save(checkpoint, TASK3_FULL_SWEEP_SAVE_PATH)
    log(f"Saved best Task 3 model to {TASK3_SAVE_PATH}")
    log(f"Saved full-sweep copy to {TASK3_FULL_SWEEP_SAVE_PATH}")

    total_minutes = (time.perf_counter() - total_started) / 60
    summary = {
        "mode": run_mode,
        "total_minutes": total_minutes,
        "cell_minutes": timings,
        "best_metadata": best_metadata,
        "best_result_bundle": best_result_bundle,
        "best_training_history": best_history,
        "all_training_histories": histories,
        "all_results": all_results,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Wrote {SUMMARY_PATH}")
    log(f"Total runtime: {total_minutes:.2f} min")


if __name__ == "__main__":
    main()
