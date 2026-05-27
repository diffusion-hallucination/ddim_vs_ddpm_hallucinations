# Experiment Catalogue


- `scripts/protocols/train_main_paper_model.sh`
  - Trains the 2D Gaussian-mixture model used by the main paper.
  - Default run name: `gaussian25_h64_t64_sigma5`.
  - Default checkpoint: `results/gaussian25_h64_t64_sigma5/checkpoints/final_model.pth`.

- `scripts/protocols/run_main_paper_experiments.sh`
  - Runs the main-paper Gaussian experiments: Exp. A, Exp. B, Exp. E, DDPM/DDIM sample counting, and the DDIM-step interpolation-rate sweep figure.
  - Default output root: `eval_results/gaussian25_h64_t64_sigma5/`.

- `scripts/protocols/run_appendix_experiments.sh`
  - Runs the non-image appendix experiments on the trained 2D Gaussian model.
  - Includes Exp. I, Exp. C, Exp. J, Exp. K, the exact-score analogue of Exp. E, and Exp. L.

- `scripts/protocols/run_exp_d_image_experiment.sh`
  - Runs the appendix image experiment for the triangle image model.

- `scripts/protocols/train_high_dim_models.sh`
  - Trains the high-dimensional Gaussian models for the configured dimensions.
  - Defaults to dimensions `2 4 8 32 64`.

- `scripts/protocols/run_high_dim_assumption_verification.sh`
  - Runs the high-dimensional Exp. A assumption checks and Exp. B exponential-convergence checks after the corresponding checkpoints exist.
  - Produces cumulative high-dimensional plots under `eval_results/dimensional_tau2_varpi/`, including `high_dim_assumption_cumulative.pdf` and `high_dim_convergence_cumulative.pdf`.

- `scripts/protocols/run_gaussian25_ablations.sh`
  - Runs the Gaussian25 ablations over `T`, `sigma`, and `ell`.

All protocol scripts support `DRY_RUN=true` to print the commands without executing them.

## Experiment Files

- `experiments/exp_a_two_mode.py`
  - Implements Exp. A: verification of the two key assumptions as a function of `kappa`.
  - Also provides the sample-count utility used by `visualize_samples`.
  - Main outputs:
    - `two_mode_assumption/exp_a_tau_vs_kappa_<solver>.pdf`
    - `two_mode_assumption/exp_a_tau_vs_kappa_<solver>.csv`
    - `two_mode_assumption/exp_a_tau_vs_kappa_<solver>.json`
    - `two_mode_assumption/exp_a_u_vs_kappa_<solver>.pdf`
    - `samples/ddim_hallucination_<num_samples>_<ddim_steps>.txt`
    - `samples/ddpm_hallucination_<num_samples>.txt`

- `experiments/exp_b_convergence_to_line.py`
  - Implements Exp. B: exponential convergence back toward the closest-pair line segment.
  - Main outputs:
    - `exp_convergence/exp_b_convergence_curve_<solver>.pdf`
    - `exp_convergence/exp_b_convergence_curve_<solver>.csv`
    - `exp_convergence/exp_b_convergence_curve_<solver>.json`

- `experiments/exp_c_compute_jacobian.py`
  - Implements Exp. C: DDIM midpoint-Jacobian eigenvalue curves.
  - This is part of the appendix pipeline.
  - Main outputs:
    - `eigvalues/ddim_quad_50.pdf`
    - `eigvalues/exp_c_eig_curves_ddim_quad_50.csv`
    - `eigvalues/exp_c_eig_curves_ddim_quad_50.json`

- `experiments/exp_d_image_diffusion.py`
  - Implements Exp. D: image-diffusion sampling and shape-count based hallucination measurement.
  - Run through `scripts/protocols/run_exp_d_image_experiment.sh`.

- `experiments/exp_e_hall_with_radius.py`
  - Implements Exp. E: interpolation or hallucination rate after starting trajectories on `L_t`.
  - Runs DDIM, DDIM plus a small number of DDPM steps, and DDPM reference curves.
  - Main outputs:
    - `hallucinations_radius/ddim_quad_50_hall_with_radius.pdf`
    - `hallucinations_radius/ddim_quad_50_ddpm_mix_z2_hall_with_radius.pdf`
    - `hallucinations_radius/ddim_quad_50_ddpm_mix_z5_hall_with_radius.pdf`
    - `hallucinations_radius/ddim_quad_50_ddpm_mix_z8_hall_with_radius.pdf`
    - `hallucinations_radius/ddpm_hall_with_radius.pdf`
    - `hallucinations_radius/overlay_tau<tau>_gaussian25_ddim_quad_50.pdf`
    - `hallucinations_radius/exp_e_hall_radius_<solver>.csv`
    - `hallucinations_radius/exp_e_hall_radius_<solver>.json`

- `experiments/exp_i_midpoint_entry_verification.py`
  - Implements Exp. I: midpoint-entry probability decomposition at the configured `tau_3`.
  - The appendix script keeps the learned-score setting with `tau_3=11` and `vartheta=0.35 ell_t`.
  - Main outputs:
    - `midpoint_entry_verification/exp_i_midpoint_probability_table_*.md`
    - `midpoint_entry_verification/exp_i_midpoint_probability_table_*.csv`
    - `midpoint_entry_verification/exp_i_midpoint_probability_table_*.json`

- `experiments/exp_j_ddim_eta_hallucination_rate.py`
  - Implements Exp. J: final interpolation rate as DDIM eta varies.
  - Includes the DDPM reference row in the saved table.
  - Main outputs:
    - `hallucinations_eta/exp_j_ddim_eta_hallucination_rate.pdf`
    - `hallucinations_eta/exp_j_ddim_eta_hallucination_rate_long.csv`
    - `hallucinations_eta/exp_j_ddim_eta_hallucination_rate.json`

- `experiments/exp_k_ddim_eta_hall_radius.py`
  - Implements Exp. K: Exp. E-style radius sweep at fixed `tau_3` while varying DDIM eta.
  - Main outputs:
    - `hallucinations_radius_eta/exp_k_ddim_eta_tau<tau>_cumulative.pdf`
    - `hallucinations_radius_eta/exp_k_ddim_eta_hall_radius_long.csv`
    - `hallucinations_radius_eta/exp_k_ddim_eta_hall_radius.json`

- `experiments/exp_l_tau3_bisector_closed_form.py`
  - Implements Exp. L: full-mixture exact-score checks for the `tau_3` bisector geometry and the empirical `a` threshold.
  - Checks `K` on the full interval and `lambda_rep` on the truncated interval.
  - Main outputs:
    - `tau3_bisector_closed_form_verification/empirical_a_threshold/summary.json`
    - `tau3_bisector_closed_form_verification/empirical_a_threshold/success_vs_a.csv`
    - `tau3_bisector_closed_form_verification/empirical_a_threshold/fixed_midpoint_exact_full_mixture/summary.json`
    - `tau3_bisector_closed_form_verification/empirical_a_threshold/fixed_midpoint_exact_full_mixture/per_t_summary.csv`
    - `tau3_bisector_closed_form_verification/empirical_a_threshold/fixed_midpoint_exact_full_mixture/K_worst_time_series.pdf`
    - `tau3_bisector_closed_form_verification/empirical_a_threshold/fixed_midpoint_exact_full_mixture_truncated/lambda_rep_worst_time_series.pdf`

- `experiments/plt_utils.py`
  - Shared plotting helper for Gaussian sample visualizations.

The figure comparing DDIM interpolation rate as a function of the number of DDIM steps against the DDPM interpolation rate is produced by `run_visualization.py`, not by a standalone experiment file.

The full main-paper script runs the required sample counting and then generates the figure:

```bash
bash scripts/protocols/run_main_paper_experiments.sh
```

Internally, the script first writes sample-count files such as:

- `eval_results/gaussian25_h64_t64_sigma5/samples/ddpm_hallucination_100000.txt`
- `eval_results/gaussian25_h64_t64_sigma5/samples/ddim_hallucination_100000_50.txt`
- `eval_results/gaussian25_h64_t64_sigma5/samples/ddim_hallucination_100000_1000.txt`

Then it runs:

```bash
python run_visualization.py \
  run_name=gaussian25_h64_t64_sigma5 \
  results_dir=eval_results \
  mode=ddim_interpolation_rate_sweep \
  expected_ddim_steps_grid=[50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000] \
  plot_min_ddim_steps=200
```