FRIS simulation code (paper-aligned)

Run (paper experiments):
  python run_experiments.py

Key notes:
- Process/measurement noise modeled as Gaussian with stds (sigma_w, sigma_v).
- Intermittent Kalman filter drives the estimation covariance; triggering uses tr(P_k) > delta.
- Packet drops are modeled as i.i.d. Bernoulli by default; Gilbert-Elliott is used for bursty loss tests.
- Outputs include error norm ||x_k - x_hat_k||, trace(P_k), and the metrics J_P, J_C, J_X.
