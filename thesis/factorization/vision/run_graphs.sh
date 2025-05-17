# --- plot ------------------------------------------------
python -m factorization.vision.graphs \
  --global_json global_flops_results.json \
  --manual_flops_json manual_flops_results.json \
  --manual_energy_json manual_energy_results.json \
  --output flops_vs_acc.pdf