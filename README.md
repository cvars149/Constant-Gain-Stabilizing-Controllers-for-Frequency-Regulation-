# Constant-Gain Stabilizing Controllers for Frequency Regulation

Project owner: C. Varshith Reddy.
  Academic supervisor: Prof. Priyank Srivastava, IIT Delhi.

## 1. What the project does

The objective is to obtain one constant state-feedback matrix K that regulates a simplified power-network model across several inertia configurations. It uses offline optimal-control demonstrations to guide performance, and a common quadratic Lyapunov certificate to check stability. The deployed law is simply u = Kx: it does not retrain online and does not need to read the active inertia-mode index.

There are two controllers in the experiment:

1. A stability-constrained imitation controller with a fixed common certificate matrix P.
2. An unconstrained ridge-regression imitation baseline.

The framework is inspired by Srivastava, Hidalgo-Gonzalez and Cortes, arXiv:2203.04502 (2022), published as Learning Constant-Gain Stabilizing Controllers for Frequency Regulation Under Variable Inertia. This implementation intentionally solves a smaller problem than the paper: it fixes P instead of jointly optimizing K and P. Sources are listed in docs/REFERENCES.md.

## 2. Why inertia and frequency matter in this model

Frequency deviation represents how fast electrical angle deviations are changing. In the adopted swing-equation model, inertia determines how strongly an imbalance accelerates the frequency state. Different inertia values therefore change the plant dynamics. A controller tuned to one plant cannot simply be assumed to behave well under every other mode or under switching.

This repository represents changing inertia using a finite set of matrices. It does not simulate renewable generation physics, weather, converter inner loops or actual generator dispatch. The renewable-energy motivation comes from the cited literature; the implemented system is a deliberately simplified linear model.

## 3. State, input and network equations

For n buses:

- theta is an n-element vector of angle deviations.
- omega is an n-element vector of frequency deviations.
- x = [theta; omega] has 2n elements.
- u has n elements and represents ideal controllable input at the buses.
- L is the network Laplacian.
- M_q is the positive diagonal inertia matrix in mode q.
- D is the damping matrix; this example uses D = I.

The equations are:

```text
theta_dot = omega
M_q omega_dot = -L theta - D omega + u

x_dot = A_q x + B_q u

A_q = [ 0             I        ]
      [ -M_q^-1 L    -M_q^-1 D ]

B_q = [ 0       ]
      [ M_q^-1  ]
```

The controller convention is u = Kx, so stabilising K normally contains negative feedback entries. The closed-loop matrix is F_q = A_q + B_q K. This sign convention matters when comparing with sources that write u = -Kx.

## 4. Default experiment configuration

| Setting | Default |
|---|---|
| Network | Synthetic, undirected, unit-weight three-bus ring |
| Inertia modes | 0.2, 0.5, 1, 2, 5, 9, uniform across buses in each mode |
| Damping | Identity matrix |
| Demonstration interval | 0.02 model time units |
| Demonstration horizon | 60 intervals per scenario |
| Number of scenarios | 12 |
| Training mode dwell | Four samples per sampled block |
| Initial training states | Normal distribution, standard deviation 0.1 |
| State-cost weights | 1 on angles, 100 on frequencies, multiplied by dt |
| Input-cost weight | Identity, multiplied by dt |
| Terminal cost | Zero |
| Training seed | 7 |
| Evaluation seed | Training seed + 1000 |
| Evaluation duration | 500 intervals of 0.02 = 10 model time units |
| Evaluation initial state | Zero angles; frequency deviation 0.05 at every bus |

These are synthetic choices, not claimed paper parameters. Values are in model units; frequency deviation is not calibrated to Hz.

## 5. Complete workflow: start to finish

### Step A: Build the network

`network(n)` creates a ring adjacency matrix W, then L = diag(sum(W)) - W. Tests check symmetry and zero row sums. The command line requires at least three buses.

### Step B: Create inertia modes

`modes(L, inertias)` creates A_q and B_q for each positive inertia specification. The default CLI uses uniform inertia. The Python API also accepts a vector of different positive inertias at different buses. Non-positive inertia is rejected.

### Step C: Generate offline demonstrations

`demonstrations()` samples switching schedules and starting states. For each schedule it solves a finite-horizon discrete LQR problem by backward Riccati recursion, then simulates forward to obtain pairs of state and ideal input.

The continuous plant is converted to an exact zero-order-hold discrete model through a block matrix exponential. The stage cost approximates the continuous quadratic cost with dt-scaled weights. A zero terminal cost is used.

Backward recursion, with S_next as the remaining value matrix:

```text
K_t = -(R + B_d^T S_next B_d)^-1 B_d^T S_next A_d
S_t = Q + A_d^T S_next A_d + A_d^T S_next B_d K_t
```

The result is X with shape (2n, N) and U with shape (n, N). Each column is one demonstration sample. The default N is 12 × 60 = 720.

The demonstration oracle knows the future sampled mode sequence. This is an offline training advantage; the final deployed controller has no such input. This is supervised imitation of optimal-control examples, not reinforcement learning or model-free field-data learning.

### Step D: Construct a feasible starting controller

`feasible_seed()` constructs:

```text
H = [ I  -I ]
    [-I  2I ]
P = inverse(H)
K_seed = -g [I I]
```

It tries g = 1, 2, 4, ... and independently checks every mode. It stops at a numerically certified seed or raises an error after 24 attempts. This is an implementation choice inspired by local-feedback feasibility ideas, not the full paper algorithm.

### Step E: Fit one constant controller

`fit()` holds P fixed and minimizes:

```text
||K X - U||_F^2 / N + 1e-6 ||K||_F^2
```

subject to every mode satisfying:

```text
(A_q + B_q K)^T P + P(A_q + B_q K) <= -1e-5 I
```

With P fixed, the matrix inequalities are affine in K and the objective is convex. The implementation uses SciPy SLSQP with eigenvalue constraints. Numerical solver success is not assumed: its status and message are saved.

The candidate is accepted if an independent check finds lambda_min(P) > 1e-7 and all Lyapunov residual maxima < -1e-7. Otherwise the certified seed is returned and the fallback flag is recorded. The acceptance threshold differs from the synthesis margin, so inspect actual residuals as well as the Boolean flag.

### Step F: Fit the baseline

The ridge baseline minimizes a regularized imitation error without stability constraints. It may fit the demonstrations more closely because it has fewer restrictions. A better training error is not automatically a stronger stability result.

### Step G: Verify stability

`certificate()` reports:

- Minimum eigenvalue of P.
- Largest Lyapunov residual eigenvalue across the modes.
- Largest real closed-loop eigenvalue across the modes.
- Whether the numerical common-P conditions pass.

A negative maximum real eigenvalue for each F_q establishes modal stability numerically. A single positive-definite P decreasing in every mode gives the stronger common quadratic certificate for arbitrary switching among these modelled continuous-time modes, with well-defined trajectories.

Failure of the baseline with this chosen P does NOT prove the baseline is unstable. It might have another common certificate or require a different analysis. Numerical checks are finite-precision checks, not interval-arithmetic proofs.

### Step H: Run a held-out switching simulation

`rollout()` propagates each interval using exp((A_q+B_q K)dt). This corresponds to continuous state feedback during the interval, not input held constant between samples. This distinction matters because the certificate is continuous-time.

The same held-out switching sequence and initial state are used for both controllers. The sequence changes potentially every evaluation interval, unlike the four-sample training blocks. Training schedules guide performance; the common certificate is checked over all specified modes, not only the sampled transitions.

### Step I: Save and interpret results

The command writes plots, numeric arrays, trajectories and metrics. These are new synthetic results generated by this code. The plot alone is not a stability proof, and a single held-out run is not a statistical performance study.

## 6. Installation and first run

Use Python 3.10 or newer. Extract this repository and open a terminal in the directory containing this README and pyproject.toml.

```bash
python -m venv .venv
```

Activate on Linux/macOS:

```bash
source .venv/bin/activate
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install and test:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m frequency_control.run --output results
```

`requirements.txt` is an alternative dependency list; it is not necessary after installing the editable package above. No external data download is required to run the example. Internet access is normally needed to install dependencies initially.

## 7. Command-line options and changing experiments

```bash
python -m frequency_control.run --help
python -m frequency_control.run --seed 42 --output results_seed42
python -m frequency_control.run --buses 12 --output results_12
```

A 12-bus run still uses a synthetic ring. It does NOT become the paper's 12-bus three-region network. Larger systems increase the number of gain variables and make constrained fitting slower.

To change inertia values, edit the list passed to `modes()` in run.py. To change LQR weights or training size, edit `demonstrations()` in core.py or call it directly with supported arguments. Keep positive inertias and rerun tests and certificates after changes.

## 8. Understanding every generated output

| File | Meaning |
|---|---|
| frequency_response.png | Bus 0 frequency comparison; not all buses |
| metrics.json | Configuration, optimiser status, fit and trajectory metrics, certificates |
| certified_fixed_P.csv | Time and all state components for constrained control |
| unconstrained_ridge.csv | Equivalent baseline trajectory |
| certified_fixed_P.npz | K, P, states, inputs and switching sequence |
| unconstrained_ridge.npz | Equivalent baseline arrays |

For n buses, CSV rows contain time, theta_0 to theta_(n-1), then omega_0 to omega_(n-1). There are 501 state rows for 500 intervals. Inputs have 500 rows and are evaluated at interval starts.

Metric details:

- `optimizer_success`: whether the optimiser declared convergence.
- `candidate_accepted`: whether the candidate passed the independent certificate.
- `used_seed_fallback`: whether fitting failed certification and the seed was used.
- `objective`: summed squared input mismatch divided by N, plus regularization.
- `imitation_mse`: mean squared mismatch over ALL input entries; this differs from the objective's scaling and excludes regularization.
- `worst_lyapunov_eigenvalue`: must be strictly negative to pass the chosen common-P test.
- `worst_spectral_abscissa`: maximum real modal eigenvalue.
- `sampled_frequency_l1`: dt times sum of absolute frequency samples over buses and interval starts.
- `sampled_control_l1`: analogous input magnitude sum.
- `final_state_norm`: Euclidean norm of the last state.

The L1 metrics use rectangular quadrature and are not exact continuous integrals.

Read saved arrays:

```python
import numpy as np
import json

data = np.load('results/certified_fixed_P.npz')
K = data['K']
P = data['P']
states = data['states']
with open('results/metrics.json') as f:
    metrics = json.load(f)
print(K.shape, P.shape, states.shape)
print(metrics['optimization'])
```

## 9. Files and why they are separated

```text
README.md                         Complete project walkthrough
pyproject.toml                    Package configuration and dependencies
requirements.txt                  Alternative dependency list
.gitignore                        Exclude temporary/generated local files
frequency_control/
    __init__.py                   Package version
    core.py                       Mathematical model and algorithms
    run.py                        End-to-end experiment runner
tests/
    test_core.py                  Six automated checks
docs/
    METHOD.md                     Concise mathematical reference
    REFERENCES.md                 Paper attribution and links
    VALIDATION.md                 Actual executed validation status
examples/generated/
    frequency_response.png        Executed demonstration plot
    metrics.json                  Executed numerical metrics
    test_log.txt                  Test execution output
    certified_fixed_P.csv         Constrained-controller trajectory
    unconstrained_ridge.csv        Baseline trajectory
    certified_fixed_P.npz         Constrained matrices and arrays
    unconstrained_ridge.npz        Baseline matrices and arrays
.github/workflows/
    tests.yml                     GitHub Actions installation/test/demo checks
```

Only core.py and run.py contain the main implementation. Remaining files support installation, explanation, checking or reproducibility. Saved examples are intentionally tracked; fresh results in `results/` are ignored.

## 10. Tests and automated checks

The six tests check network symmetry/row sums, seed certification including heterogeneous inertia, exact discretization of a simple plant, deterministic data generation, fitted-certificate and rollout behavior, and rejection of zero inertia. Run them with `python -m pytest -q`.

Passing tests does not establish correctness for arbitrary power networks. The GitHub workflow runs tests and a demo automatically. It does not require the optimiser to declare convergence; consult metrics for that status.

## 11. Actual validation and interpretation

See docs/VALIDATION.md and examples/generated/metrics.json for the run bundled in this ZIP. The distinction between optimizer convergence and independent feasibility is explicit there. A candidate can be numerically certified even if the solver stops at its iteration limit. Do not label that outcome a globally optimal controller.

## 12. Differences from the original research

The original research jointly searches a feedback gain and Lyapunov certificate using a BMI-based procedure. This repository fixes the certificate and optimizes only the gain. This is more restrictive and may lose performance. It does not reproduce the original sparsity method, paper topology, sample counts or experimental claims.

The implementation must be described as a paper-inspired demonstrator or restricted implementation, not a new research contribution or exact replication. The cited theoretical results belong to their authors. Using the academic project dates above does not establish when this code was written.

## 13. Limitations and appropriate use

- Linear angle/frequency model, not nonlinear AC power flow.
- Full state availability and ideal input access at every bus.
- No actuator saturation, generation limits, storage energy constraint or communication delay.
- No measurement noise, state estimator or cybersecurity model.
- No persistent disturbance or explicit load-step term; the example regulates an initial deviation.
- Only specified positive inertia modes are certified; unmodelled modes are not covered.
- Continuous feedback certificate does not automatically certify sampled-data implementation.
- Angle feedback assumes an available reference; absolute-angle regulation is not a physical measurement architecture demonstrated here.
- No hardware or field-data validation and no calibrated physical units.

Use for education and numerical exploration, not deployment on an electrical grid.

## 14. Troubleshooting

- `ModuleNotFoundError`: activate the environment and run editable installation from the repository root.
- No plot window: expected; the program saves a PNG using a noninteractive backend.
- Iteration limit: inspect optimiser status and certificate; a feasible candidate may still be returned without convergence.
- Fallback flag true: the returned controller is the seed, not an accepted fitted gain.
- Baseline certificate false: do not infer instability from failure with one P.
- Long 12-bus execution: start with the default three-bus example.
- Different numbers across environments: numerical solvers and library versions can differ; compare status and residuals, not just rounded values.

## 15. Attribution and licensing

Project owner: Varshith. Research ideas and paper results remain attributed to the cited authors. Papers are referenced, not redistributed. No software licence is assigned on the owner's behalf; select an appropriate licence before offering this code for reuse. No claims of professor endorsement, original research authorship or historical project results are made.
