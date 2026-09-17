COMP8851 PMP × T-Finance controlled benchmark backup.

Purpose:
Preserve the important reproducibility/evidence files before the Vast instance
is stopped or destroyed.

Intended contents:
- unified final results and logs
- per-run JSON/CSV evidence
- epoch timing data
- final summaries/status
- frozen split/data assets available on this host
- execution scripts
- exact PMP repository commit
- relevant PMP source/config snapshot
- Python/package environment
- GPU evidence
- SHA256 inventory

Large model checkpoint directories are intentionally excluded.
