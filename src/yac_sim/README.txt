YAC / SCC simulation code (paper-aligned)

Run (paper 1 experiments):
  python -m yac_sim --outdir result

Key notes:
- Process/measurement noise modeled as Gaussian with stds (sigma_w, sigma_v).
- Intermittent Kalman filter drives the estimation covariance; triggering uses tr(P_k) > delta.
- Outputs include prediction error norm ||tilde x||, innovation norm, and trace(P_k).
- Trade-off curves use delivered packets as the primary communication budget metric.
