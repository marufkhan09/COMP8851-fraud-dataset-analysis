# COMP8851 Assessment 3 — GitHub ↔ Mac ↔ Vast.ai upload map and run order

**Checked 2026-10-15** against `marufkhan09/COMP8851-fraud-dataset-analysis` GitHub `main` and `Pathik` branches, local Finder screenshots, and the verified v3 pre-rental package. No rental has been made or training started by this guide.

## Decisions

- The project repository is `https://github.com/marufkhan09/COMP8851-fraud-dataset-analysis`.
- Team `main` commit checked: `d36f5003d934862ecbe4acf828d910e6be40bf96`. Pathik commit checked: `d49f7d4c19c9876efec172c7e71f42f16b06abdb`.
- The branches have diverged. Do not force-push, reset, or blindly merge either.
- Create a new branch from `origin/Pathik`: `pathik/ass3-benchmark`; keep `main` as a read-only reference for new team changes.
- **GitHub:** source changes, experiment launchers, small JSON/CSV/MD results, provenance and docs. **Vast.ai:** GPU compute, author repos, virtualenvs, canonical raw datasets/splits, checkpoints and detailed logs. **Mac:** original backup folders and a copy of every final evidence archive.

## Step A — Keep the original Mac checkout safe

In Finder, navigate to `COMP8851-fraud-dataset-analysis`; right-click that folder and choose **New Terminal at Folder** (or open Terminal and `cd` into the actual folder). Check:

```bash
git rev-parse --show-toplevel
git remote -v
git status --short
git branch -a
```

Do not remove the `bwgnn vast ai`, `pmp vast ai`, `gaap evidence`, `pmp notebooks and docx`, `calibration vast ai` or other prior folders. Any local uncommitted work must remain intact.

Create a separate Git worktree without changing the original checkout:

```bash
BASE="$(git rev-parse --show-toplevel)"
git fetch origin --prune
git worktree add -b pathik/ass3-benchmark "$BASE/../COMP8851-ass3" origin/Pathik
cd "$BASE/../COMP8851-ass3"
mkdir -p docs/pathik/ass3 tools/vast/configs tools/vast/scripts tools/vast/guides
mkdir -p models/rgtan models/sawgad models/eogfd results/ass3/extensions
```

If the branch name is already in use, inspect its contents with `git branch -a` instead of reusing it blindly. If a worktree directory already exists, stop rather than deleting it.

## Step B — Copy the *selected* local Ass3 preparation files to Git

The screenshots show the following files inside your original checkout's `ASS 3 - Final report` folder. Copy only these named files and do not recursively copy the whole folder:

```bash
cp "$BASE/ASS 3 - Final report/COMP8851_Ass2_Evidence_Freeze_Audit_Explained.docx" \
   docs/pathik/ass3/Evidence_Freeze_Audit.docx
cp "$BASE/ASS 3 - Final report/COMP8851_Benchmark_Closure_Plan_Explained.docx" \
   docs/pathik/ass3/Benchmark_Closure_Plan.docx
cp "$BASE/ASS 3 - Final report/COMP8851_Vast_Project_Config_v3.yaml" \
   tools/vast/configs/vast_project_config.yaml
cp "$BASE/ASS 3 - Final report/COMP8851_Bootstrap_Pinned_Repos_v3.sh" \
   tools/vast/scripts/bootstrap_pinned_repos.sh
cp "$BASE/ASS 3 - Final report/COMP8851_Offer_Preview_DRY_RUN_v3.sh" \
   tools/vast/scripts/offer_preview_DRY_RUN.sh
cp "$BASE/ASS 3 - Final report/COMP8851_First_30_Minutes_v3.md" \
   tools/vast/guides/first_30_minutes.md
```

Before publishing these copied files, check `git status --short` and `git diff --stat` for unexpected data. These documents are **planning**, not verified final evidence. Add a short `docs/pathik/ass3/README.md` explaining this status. Prefer to leave `models/rgtan`, `models/sawgad` and `models/eogfd` empty until canonical adapters are written and tested. Empty directories are not tracked by Git; that is normal.

Then stage **only** the selected paths:

```bash
git add docs/pathik/ass3 tools/vast
# Inspect which files are staged; confirm no raw datasets/checkpoints:
git diff --cached --stat
git diff --cached --name-only
git commit -m "ass3: add benchmark planning and Vast setup documentation"
git push -u origin pathik/ass3-benchmark
```

Do not push to `main` or `Pathik` directly. Do not commit environment folders, model checkpoint weights, raw datasets, large ZIPs or authentication keys. Ask a teammate to review if merging is later required.

## Step C — What to upload to Vast.ai

**Upload one file:** `COMP8851_PreRent_Ready_v3.zip` from Downloads to `/workspace/` on the rented Vast.ai instance. The full v3 ZIP is preferred over six manually selected copies; it contains the checks and manifest templates.

The repo contents are cloned by the package bootstrap; don't reupload the Mac repo. However, **the canonical source datasets, frozen split files, and any evidence that exists only on your Mac/previous instance must be transferred separately and checksum-verified**.

Copy the ZIP using Vast's file manager or SCP, using the *exact SSH host/port from that instance*:

```bash
scp -P PORT "$HOME/Downloads/COMP8851_PreRent_Ready_v3.zip" root@SSH_HOST:/workspace/
```

Replace `PORT` and `SSH_HOST` with the real instance SSH values; do not use the public marketplace `host:45423` as an SSH address or the machine number `m:14804` as an instance ID.

On Vast.ai:

```bash
cd /workspace
unzip COMP8851_PreRent_Ready_v3.zip
cd COMP8851_PreRent_Ready_v3
bash scripts/09_local_selfcheck.sh
bash scripts/01_host_preflight.sh | tee host_preflight.log
bash scripts/02_bootstrap_repositories.sh | tee bootstrap.log
```

The bootstrap creates:

```text
/workspace/comp8851/
  repos/team_repo/             pinned Pathik source (READ-ONLY reference)
  repos/antifraud/             pinned RGTAN implementation
  repos/SAWGAD/               pinned SAWGAD implementation
  repos/EOGFD/                pinned EOGFD implementation
  assets/                     canonical datasets and frozen split files (not included)
  envs/                       separate runtimes, when installed
  manifests/                  protocol, source pins and asset checks
  evidence/                   historical and new training evidence
  logs/                       training and preflight logs
  cache/                      reusable compute assets
```

**Additional team-main reference:** after bootstrap, if you need the latest team code/splits, clone the checked `main` separately; do not overwrite `team_repo`:

```bash
git clone https://github.com/marufkhan09/COMP8851-fraud-dataset-analysis.git \
  /workspace/comp8851/repos/team_main_reference
git -C /workspace/comp8851/repos/team_main_reference checkout --detach \
  d36f5003d934862ecbe4acf828d910e6be40bf96
```

If you have pushed the new branch, clone it into another separate directory for development, not over pinned references:

```bash
git clone --branch pathik/ass3-benchmark \
  https://github.com/marufkhan09/COMP8851-fraud-dataset-analysis.git \
  /workspace/comp8851/repos/team_ass3_work
```

## Step D — Canonical assets (not GitHub uploads)

Official source graphs and training splits must be identified, transferred and verified. The v3 manifest requires them and is intentionally incomplete.

**T-Social previously verified target paths:**

```text
/workspace/comp8851/assets/tsocial/tsocial
/workspace/comp8851/assets/tsocial/tsocial_seed2_nested_splits.npz
```

Historical expected SHA256:
- graph: `8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0`
- split: `b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519`

For **Amazon**, the live `main` branch currently contains `shared/splits/amazon/amazon_TR10_seed2.npz`, `amazon_TR20_seed2.npz`, `amazon_TR30_seed2.npz`, `amazon_TR40_seed2.npz` and a nested-split manifest. These are candidate canonical assets **subject to hash/ID verification** against your earlier models; `Pathik/shared/splits` alone contains only `.gitkeep`. Do not assume the GitHub files are accepted across the benchmark without checking.

For the remaining source/split files, recover from old Vast assets, local backup or model evidence; don't invent paths or silently regenerate a new split. Especially audit **T-Finance's corrected 1,804 fraud labels**, **Elliptic chronology**, **FDCompCN relation view**.

Inspect candidate files on your Mac without altering them:

```bash
find "$BASE" -type f \( -name '*.npz' -o -name '*.dgl' -o -name '*.mat' -o -name '*tsocial*' \) -print
```

After copying a dataset, edit a **working copy** of `configs/assets_manifest.json` to point to its real path and SHA256. Run:

```bash
python3 scripts/03_verify_assets.py --manifest configs/assets_manifest.json \
    --workspace /workspace/comp8851 --dataset Amazon
```

It should say `BLOCKED` until the source/split files and hashes are genuinely present. Never remove that guard just to begin official experiments.

## Step E — Which model to try first?

1. **Fastest *safe environment smoke*:** **SAWGAD on Amazon**, using one run, **one warm-up epoch and one full epoch**. Its released Amazon downloader and modern setup are more self-contained. Example **only after its environment passes CUDA/DGL tests**:

```bash
cd /workspace/comp8851/repos/SAWGAD
uv run sawgad --num-runs 1 --warmup-epochs 1 \
  --warmup-batches-per-epoch 1 --full-epochs 1
```

This is **NOT an Assessment 3 benchmark result**. It simply demonstrates that the CUDA/DGL environment, synthetic data generation and model can execute. The first run may still take extra time to download dependencies and Amazon data.

2. **First model to aim for a *canonical full benchmark*:** **RGTAN on Amazon**, after adapting it to exactly the Ass2 frozen split and validation-only protocol and obtaining approval for how its native five-fold method is handled. It is expected to be a more tractable supervised baseline than the other two, but no measured runtime comparison exists yet.
3. **Then SAWGAD on Amazon under the formally approved training/label regime.** Its native 100+350 epochs and 30-label weak supervision cannot be silently equated to the usual 100-epoch TR40 benchmark.
4. **Last EOGFD**, first on Amazon or another small dataset, because its released runner has a class-name import discrepancy and evaluates test metrics inside training. Repair that in a documented external adapter/runner before official training. Its population-based computation may make T-Social infeasible under 48 GB GPU memory.

Never launch all 432 planned jobs from the ZIP directly: they are intentionally blocked planning rows, not runnable training scripts. Complete provenance and correctness gates before testing T-Social.

## Step F — GitHub publication after real results

Only after evidence passes audit, commit limited reproducibility artifacts to your new branch:

```text
models/rgtan/     adapter, launcher, provenance/README
models/sawgad/    adapter, launcher, provenance/README
models/eogfd/     adapter, launcher, provenance/README
results/ass3/extensions/<model>/<dataset>/
                  small per-seed CSV, aggregate CSV, run config JSON, checksums, README
```

Keep checkpoints and raw graph assets off GitHub; archive safely on Mac/external storage. Do not put sensitive Vast SSH keys or billing details in Git. Track every `git rev-parse HEAD` and all source/split SHA256 hashes.

**First reply to send back once rented:** outputs of `09_local_selfcheck.sh`, `01_host_preflight.sh`, `02_bootstrap_repositories.sh`; then the location of the canonical Amazon source/splits. We will not assume an old run is final until the underlying evidence is inspected.
