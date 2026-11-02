"""Extract and document every T-Social attempt from the run archive.

T-Social is the one dataset in the suite that produced no test evaluation. It was
not skipped: both models were attempted repeatedly, and each attempt recorded its
own artefact bundle. This script pulls those bundles out of the results archive
into ``evidence/tsocial/`` and writes the attempt log that the deliverable cites.

Checkpoints are excluded: the three T-Social CARE-GNN checkpoints are 231 MB each
and add nothing a reader can use.

Run:  python tsocial_attempts.py
"""

from __future__ import annotations

import csv
import json
import os
import zipfile
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
DELIVERABLE = os.path.dirname(HERE)
PROJECT = os.path.dirname(DELIVERABLE)
ARCHIVE = os.path.join(PROJECT, "tfinance_tsocial_results.zip")
EVIDENCE = os.path.join(DELIVERABLE, "evidence", "tsocial")
TABLES = os.path.join(DELIVERABLE, "tables")


RUNTIME_BUNDLE = os.path.join(PROJECT, "comp8851.zip")


def extract_dataset_records():
    """Pull T-Social's dataset manifest and split registry out of the runtime bundle.

    No model completed on T-Social, so its graph statistics never reached a run
    summary the way the other five datasets' did. They were computed when the
    canonical view was frozen, and they live in the runtime bundle that was shipped
    to the benchmark host. Copying them in lets the dataset profile cover all six
    datasets even though only five have results.
    """
    if not os.path.isfile(RUNTIME_BUNDLE):
        return []
    wanted = ("data/dataset_manifest.json", "data/checksums.sha256",
              "data/tsocial/tsocial_dataset_report.json",
              "shared/splits/tsocial/tsocial_split_manifest_seed2.json")
    target_dir = os.path.join(EVIDENCE, "dataset")
    os.makedirs(target_dir, exist_ok=True)
    written = []
    with zipfile.ZipFile(RUNTIME_BUNDLE) as archive:
        names = set(archive.namelist())
        for name in wanted:
            if name not in names:
                continue
            target = os.path.join(target_dir, os.path.basename(name))
            with archive.open(name) as src, open(target, "wb") as dst:
                dst.write(src.read())
            written.append(os.path.relpath(target, DELIVERABLE))
    return written


def extract():
    """Copy every T-Social artefact except checkpoints into evidence/tsocial/."""
    if not os.path.isfile(ARCHIVE):
        print(f"archive not found: {ARCHIVE}")
        return []
    os.makedirs(EVIDENCE, exist_ok=True)
    written = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for info in archive.infolist():
            name = info.filename
            if "tsocial" not in name or not info.file_size:
                continue
            if "/checkpoints/" in name:
                continue
            # results_care-gnn/raw/care-gnn/tsocial/unified/tr40/seed_2/x
            #   -> final_seed2/x
            # results_care-gnn/tuning/care-gnn_tsocial/trial_00/.../seed_2/x
            #   -> tuning_trial00/x
            if "/tuning/" in name:
                trial = name.split("/trial_")[1].split("/")[0]
                folder = f"tuning_trial{trial}"
            elif "/raw/" in name:
                seed = name.split("/seed_")[1].split("/")[0]
                folder = f"final_seed{seed}"
            else:
                folder = "summary"
            target_dir = os.path.join(EVIDENCE, folder)
            os.makedirs(target_dir, exist_ok=True)
            target = os.path.join(target_dir, os.path.basename(name))
            with archive.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            written.append(os.path.relpath(target, DELIVERABLE))
    return written


def read_summary(folder):
    path = os.path.join(EVIDENCE, folder, "summary.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# GHRN's T-Social attempts ran on the benchmark host as a five-rung escalation
# ladder, each rung reducing the resource demand of the one before. Every rung
# failed. Transcribed from the execution notebook (notebookdb72ab9871.ipynb,
# preserved in notebooks/); the artefact bundles stayed on that host under
# probe_ghrn_1..5/results_ghrn/raw/ghrn/tsocial/unified/tr40/seed_2.
#
# Two distinct failure modes, which matters: rung 1 ran out of GPU memory, and
# once the configuration was small enough to fit, rungs 2-5 hit an index-dtype
# defect instead. The second is a code bug, not a capacity limit.
DGL_DTYPE = ('DGLError: Expect argument "u" to have data type torch.int32. '
             'But got torch.int64.')
CUDA_OOM = ("OutOfMemoryError: CUDA out of memory — tried to allocate 1.38 GiB "
            "with 638.81 MiB free of 14.56 GiB")

GHRN_LADDER = [
    OrderedDict(
        attempt="probe_ghrn_1", model="GHRN", phase="escalation rung 1 of 5",
        seed=2, status="FAILED", device="cuda:0", epochs_completed=None,
        wall_seconds=120.0, mean_train_epoch_s=None, mean_validation_epoch_s=None,
        peak_gpu_mb=None, best_validation_auprc=None, test_evaluated=False,
        failure_type=CUDA_OOM,
        config="hid_dim=64, order=2 (published config)",
        run_id="ghrn_tsocial_unified_tr40_seed2_t4_20260917T100407Z",
        provenance="execution notebook"),
    OrderedDict(
        attempt="probe_ghrn_2", model="GHRN", phase="escalation rung 2 of 5",
        seed=2, status="FAILED", device="cuda:0", epochs_completed=None,
        wall_seconds=120.0, mean_train_epoch_s=None, mean_validation_epoch_s=None,
        peak_gpu_mb=None, best_validation_auprc=None, test_evaluated=False,
        failure_type=DGL_DTYPE, config="hid_dim=32, order=2 (half width)",
        run_id="ghrn_tsocial_unified_tr40_seed2_t4_20260917T101137Z",
        provenance="execution notebook"),
    OrderedDict(
        attempt="probe_ghrn_3", model="GHRN", phase="escalation rung 3 of 5",
        seed=2, status="FAILED", device="cuda:0", epochs_completed=None,
        wall_seconds=120.0, mean_train_epoch_s=None, mean_validation_epoch_s=None,
        peak_gpu_mb=None, best_validation_auprc=None, test_evaluated=False,
        failure_type=DGL_DTYPE, config="hid_dim=32, order=1 (lower order)",
        run_id="ghrn_tsocial_unified_tr40_seed2_t4_20260917T101930Z",
        provenance="execution notebook"),
    OrderedDict(
        attempt="probe_ghrn_4", model="GHRN", phase="escalation rung 4 of 5",
        seed=2, status="FAILED", device="cpu", epochs_completed=None,
        wall_seconds=420.0, mean_train_epoch_s=None, mean_validation_epoch_s=None,
        peak_gpu_mb=None, best_validation_auprc=None, test_evaluated=False,
        failure_type=DGL_DTYPE, config="hid_dim=64, order=2 (published config)",
        run_id="ghrn_tsocial_unified_tr40_seed2_t4_20260917T103149Z",
        provenance="execution notebook"),
    OrderedDict(
        attempt="probe_ghrn_5", model="GHRN", phase="escalation rung 5 of 5",
        seed=2, status="FAILED", device="cpu", epochs_completed=None,
        wall_seconds=300.0, mean_train_epoch_s=None, mean_validation_epoch_s=None,
        peak_gpu_mb=None, best_validation_auprc=None, test_evaluated=False,
        failure_type=DGL_DTYPE, config="hid_dim=32, order=2 (half width)",
        run_id="ghrn_tsocial_unified_tr40_seed2_t4_20260917T104635Z",
        provenance="execution notebook"),
]


def build_attempt_log():
    """One row per recorded T-Social attempt, from the artefact bundles."""
    rows = []
    for folder in sorted(os.listdir(EVIDENCE)):
        data = read_summary(folder)
        if not data:
            continue
        timing = data.get("timing") or {}
        cfg = data.get("configuration") or {}
        error = data.get("error") or {}
        rows.append(OrderedDict(
            attempt=folder,
            model=data.get("model"),
            phase="tuning" if folder.startswith("tuning") else "final run",
            seed=data.get("train_seed"),
            status=data.get("status"),
            device=cfg.get("resolved_device"),
            epochs_completed=timing.get("training_epochs_completed"),
            wall_seconds=round(timing.get("total_train_plus_validation_seconds") or 0, 1),
            mean_train_epoch_s=round(timing.get("mean_train_epoch_seconds") or 0, 1),
            mean_validation_epoch_s=round(
                (timing.get("total_validation_seconds") or 0)
                / max(timing.get("training_epochs_completed") or 1, 1), 1),
            peak_gpu_mb=round(timing.get("peak_gpu_memory_mb") or 0, 1),
            best_validation_auprc=data.get("best_selection_value"),
            test_evaluated=bool(data.get("test_metrics")),
            failure_type=error.get("type") or "",
            config=f"emb_size={cfg.get('emb_size')}, lr={cfg.get('lr')}",
            run_id=data.get("run_id") or "",
            provenance="artefact bundle",
        ))
    rows.extend(GHRN_LADDER)
    rows.sort(key=lambda r: (r["model"] or "", r["phase"], str(r["attempt"])))
    return rows


def best_validation_curve():
    """The per-epoch validation curve of the final CARE-GNN attempt."""
    path = os.path.join(EVIDENCE, "final_seed2", "validation_metrics.csv")
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main():
    written = extract()
    records = extract_dataset_records()
    print(f"extracted {len(written)} T-Social artefact files and "
          f"{len(records)} dataset records into evidence/tsocial/")

    rows = build_attempt_log()
    if not rows:
        print("no T-Social bundles found")
        return

    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(TABLES, "tsocial_attempts.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    curve = best_validation_curve()
    best = None
    if curve:
        best = max(curve, key=lambda r: float(r["auprc"]))

    lines = [
        "# T-Social — every recorded attempt", "",
        "T-Social produced no test evaluation. It was **not skipped**: both models "
        "were attempted repeatedly, and each attempt below is taken either from its "
        "own artefact bundle in `../evidence/tsocial/` or from the run console, as "
        "the final column states.", "",
        "| Attempt | Model | Phase | Status | Device | Config | Epochs | Wall | "
        "Best val AUPRC | Test | Failure |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        auprc = (f"{float(r['best_validation_auprc']):.4f}"
                 if r["best_validation_auprc"] is not None else "—")
        wall = (f"{r['wall_seconds'] / 60:,.0f} m"
                if r["wall_seconds"] else "—")
        lines.append(
            f"| {r['attempt']} | {r['model']} | {r['phase']} | **{r['status']}** | "
            f"{r['device'] or '—'} | {r.get('config') or '—'} | "
            f"{r['epochs_completed'] or '—'} | {wall} | {auprc} | "
            f"{'yes' if r['test_evaluated'] else 'no'} | {r['failure_type'] or '—'} |")

    modes = OrderedDict()
    for r in rows:
        if r["status"] == "FAILED" and r["failure_type"]:
            key = r["failure_type"].split(":")[0]
            modes[key] = modes.get(key, 0) + 1

    lines += [
        "",
        f"**{len(rows)} attempts, not one test evaluation.** The CARE-GNN rows are read "
        "from artefact bundles in `../evidence/tsocial/`. The GHRN rows are transcribed "
        "from the execution notebook preserved in `../notebooks/`; that ladder ran on "
        "the benchmark host and its bundles stayed there under `probe_ghrn_1..5/`.", "",
        "## Failure modes", "",
        "| Failure | Attempts |", "|---|---|",
    ] + [f"| {k} | {v} |" for k, v in modes.items()] + [
        "",
        "The two GHRN modes are worth separating. **Rung 1 ran out of GPU memory** at "
        "the published width: it needed a further 1.38 GiB with 638.81 MiB free of the "
        "T4's 14.56 GiB. That is a capacity limit and a genuine RQ3 finding — T-Social "
        "does not fit GHRN at published width on a 16 GB card. **Rungs 2 to 5 then hit "
        "an index-dtype defect**, on GPU and CPU alike, which is a code bug rather than "
        "a resource limit and is fixed in `../source_code/models/ghrn/ghrnlib/"
        "backend.py` but not yet re-run. Shrinking the model got past the memory wall "
        "and straight into the bug, so GHRN's true cost on T-Social is still unmeasured.",
    ]

    if curve and best:
        lines += [
            "", "## Validation curve of the final CARE-GNN attempt", "",
            f"The run trained for {len(curve)} epochs before it was stopped. It was "
            f"learning: validation AUPRC rose from "
            f"{float(curve[0]['auprc']):.4f} at epoch 0 to a best of "
            f"**{float(best['auprc']):.4f}** at epoch {best['epoch']} "
            f"(AUROC {float(best['auroc']):.4f}).", "",
            "| Epoch | Val AUPRC | Val AUROC | Val Macro-F1 | Threshold |",
            "|---|---|---|---|---|",
        ]
        for r in curve:
            mark = " **(best)**" if r["epoch"] == best["epoch"] else ""
            lines.append(
                f"| {r['epoch']}{mark} | {float(r['auprc']):.4f} | "
                f"{float(r['auroc']):.4f} | {float(r['macro_f1']):.4f} | "
                f"{float(r['threshold']):.2f} |")
        lines += [
            "", "**These are validation figures from an incomplete run.** They are "
            "reported here to show what happened, and they are deliberately kept out "
            "of every results table in this package. No T-Social number appears "
            "anywhere a test metric is expected.",
        ]

    with open(os.path.join(TABLES, "tsocial_attempts.md"), "w", encoding="utf-8") as h:
        h.write("\n".join(lines) + "\n")

    print(json.dumps({
        "attempts": len(rows),
        "models": sorted({r["model"] for r in rows if r["model"]}),
        "statuses": sorted({r["status"] for r in rows if r["status"]}),
        "any_test_evaluated": any(r["test_evaluated"] for r in rows),
        "validation_epochs_recorded": len(curve),
        "best_validation_auprc": float(best["auprc"]) if best else None,
    }, indent=2))


if __name__ == "__main__":
    main()
