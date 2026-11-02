# Kaggle integration test (stage 1)

The Kaggle kernel `kaggle_smoke.py` runs the same production code as Vast.ai. It installs torch 2.4.0 and DGL 2.4.0, verifies the environment, fetches the pinned official code and prepares and verifies all six datasets. It then runs the test suite and the 18-pair GPU smoke matrix, plus a full-budget pilot on FDCompCN. Smoke and pilot outputs are feasibility evidence only.

## 1. Secure authentication (do this yourself; never paste a token into chat)

The token pasted into the chat on 2026-10-11 must be treated as compromised. Revoke it at <https://www.kaggle.com/settings> → API → *Expire API Token*. Then create a new one and store it locally **without echoing it into shell history**:

PowerShell (Windows):

```powershell
New-Item -ItemType Directory -Force "$HOME\.kaggle" | Out-Null
$t = Read-Host -AsSecureString "Kaggle token"
[System.Net.NetworkCredential]::new('', $t).Password | Set-Content -NoNewline "$HOME\.kaggle\access_token"
```

Linux/macOS:

```bash
mkdir -p ~/.kaggle && read -rs KT && printf '%s' "$KT" > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token && unset KT
```

Install or upgrade the CLI with `pip install --upgrade kaggle`. Check access with `kaggle datasets list --mine`. Inside a Kaggle notebook no token is needed; the kernel itself uses none.

## 2. Push, run, download

```bash
python scripts/kaggle_push.py --user ahmadyasin1
```

This does four things:

1. Stages a private dataset `ahmadyasin1/comp8851-phase2-bundle`, containing the code plus the frozen canonical caches and split registry.
2. Pushes the private GPU kernel `ahmadyasin1/comp8851-phase2-smoke`, with Internet enabled.
3. Polls its status.
4. Downloads `/kaggle/working/outputs` into `results/kaggle/`.

Kaggle limits respected:

- a session lasts at most 12 hours and gets about 30 GB of RAM;
- T-Social is fully schema- and hash-verified by `prepare_data.py`, but executed on a documented 300,000-node induced subgraph;
- the GPU (T4 or P100) is not the protocol's A6000, so timings are not hardware-comparable.
