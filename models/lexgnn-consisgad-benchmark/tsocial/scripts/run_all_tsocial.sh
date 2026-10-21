source /workspace/activate_comp8851_elliptic.sh
cd /workspace/tsocial_work
while ! grep -q lexgnn logs/gate_status.txt 2>/dev/null; do sleep 20; done
python run_tsocial_benchmark.py consisgad > logs/driver_consisgad.log 2>&1
