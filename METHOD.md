# Compact mathematical reference

The complete explanation is in README.md.

Dynamics: x=[theta;omega], A_q=[[0,I],[-M_q^-1 L,-M_q^-1 D]], B_q=[[0],[M_q^-1]], u=Kx.

Offline demonstration generation uses exact ZOH plant discretization, dt-scaled quadratic stage costs and backward finite-horizon Riccati recursion, with zero terminal cost.

Construct P = inverse([[I,-I],[-I,2I]]) and search K_seed=-g[I,I] by doubling g and checking feasibility. Fix P during imitation optimization. Minimize ||KX-U||_F^2/N + 1e-6||K||_F^2 subject to (A_q+B_qK)^T P + P(A_q+B_qK) <= -1e-5 I for every mode. Candidate acceptance independently checks P eigenvalues above 1e-7 and worst Lyapunov eigenvalue below -1e-7. A rejected candidate triggers explicit seed fallback.

This fixed-P convex subproblem is solved numerically by SLSQP, not the paper's joint nonconvex BMI algorithm. A common quadratic certificate concerns continuous-time feedback and switching only among the modelled modes. Eigenvalue tests have finite precision. Modal stability alone is weaker; failure with a selected common P does not prove instability.

Rollouts use exp((A_q+B_qK)dt), not Ad+BdK. Trajectory metrics are sampled rectangular sums. No physical unit conversion, persistent disturbance, saturation, state estimator or nonlinear AC validation is included.
