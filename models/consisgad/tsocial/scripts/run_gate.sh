source /workspace/activate_comp8851_elliptic.sh
cd /workspace/tsocial_work
for m in consisgad lexgnn; do
  timeout 7200 python adapters/$m/train_tsocial.py --config configs/${m}_gate.json > logs/${m}_gate.log 2>&1
  echo "$m exit=$?" >> logs/gate_status.txt
done
