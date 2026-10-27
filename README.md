# AION 2D MOT — AI surrogate

A factorised generative surrogate of the AION 2D+ magneto-optical trap, replacing an
AtomECS run costing ≈22 minutes per parameter set with ≈28 ms of model time on an NVIDIA RTX 3080
(well under a second on a CPU).

Two models are applied in sequence for a given set of nine control parameters θ:

| | | |
|---|---|---|
| **DNN** | `P(θ)` | how many atoms are transmitted past the threshold plane, out of 5×10⁶ launched |
| **CNF-DS** | `p(x \| θ)` | the position and velocity vectors `x = (X, Y, Vx, Vy, Vz)` of those atoms |

The generative half is a **conditional** normalising flow trained with an initial
**denoising schedule**, adapted from data mollification (Tran et al., NeurIPS 2023): the
target is smoothed towards noise for the first ten epochs and annealed to clean data, after
which training proceeds as a standard flow with exponential learning-rate decay and early
stopping. In the paper it recovers part of the deficit a conventional flow shows in the
sparsely populated regions of the parameter space, at a small fraction of the sampling cost
of a diffusion model; the diffusion model remains the more accurate.

Reference: A. Evans, E. Bentine, M. Mieskolainen and C. Baynham, *Generative Surrogates for
Cold-Atom Interferometry* (2026). The paper evaluates three random seeds; the checkpoints
here are seed 2026. The training and evaluation code is available from the authors on
request.

## Install

```bash
conda env create -f environment.yml
conda activate aion_2dmot
```

The packaged checkpoints and their exact paper sources are recorded in
[`models/PROVENANCE.md`](models/PROVENANCE.md).

## Run one parameter set

Edit `params.json`, then:

```bash
python simulate.py                    # summary to stdout
python simulate.py --out vectors.csv  # also write the sampled vectors
```

## Optimise the source (example, not validated)

The paper presents the surrogate as a proof of concept and leaves the source optimisation to
future work: a first optimisation within the sampled range returned configurations at its
boundary, so any optimum found here should be extended and confirmed with AtomECS before it
is relied on. The optimiser and objective are provided as the intended application.

```bash
python optimise.py --iterations 1000
```

Bayesian optimisation over the nine parameters with HyperOpt's tree-structured Parzen
estimator. The search space is fixed to the ranges the surrogate was trained over, so the
optimiser cannot wander outside its region of validity. Evaluations, the running best and
the trials object are written to `optimisation_output/` as the run proceeds.

## The objective

The default is **capturable flux** — the loading rate of the following 3D MOT, which is the
figure of merit the 2D MOT source literature uses. An atom counts if it is slow enough to be
captured, clears the remainder of the differential pumping tube, and is still inside the
capture volume when it arrives:

```
J(θ) = N P(θ) · ⟨ 1[v_z ≤ v_c] · 1[atom clears the tube] · 1[atom arrives within R_c] ⟩
```

The last two are evaluated ballistically including gravity. That matters: the divergence
angle is `v⊥/v_z`, so for a fixed transverse velocity a *slower* beam diverges more, and it
sags further, as `½g(L/v_z)²`. The objective therefore has an interior optimum in beam
velocity rather than a monotonic preference for slow atoms.

Every constant is a measurable property of the apparatus rather than a free weight, and
`objective.py` documents where each comes from. The drift distance is 0.3675 m from the
30 mm threshold plane to the 3D MOT centre, derived from the displacement encoded in the
AtomECS 3D-MOT source.

`objective.py` also provides `weighted_penalty`, an earlier soft-penalty form, for
comparison. It optimises two moments rather than the distribution, applies no capture
threshold, and rewards low speed without limit, so it favours beams too slow to survive the
drift. Select it with `--objective weighted_penalty`.

## Validity

The surrogate was trained on 2¹⁹ Sobol-distributed parameter sets covering:

| Symbol | Parameter | Min | Max | Units |
|---|---|---|---|---|
| $\delta_c$ | Cooling beam detuning | −250 | 0 | MHz |
| $w_c$ | Cooling beam radius | 7 | 15 | mm |
| $P_c$ | Cooling beam power | 50 | 350 | mW |
| $\delta_p$ | Push beam detuning | −350 | 0 | MHz |
| $w_p$ | Push beam radius | 0 | 3 | mm |
| $P_p$ | Push beam power | 0 | 20 | mW |
| $d_p$ | Push beam offset | 0 | 5 | mm |
| $\nabla B$ | Quadrupole gradient | 0 | 100 | G/cm |
| $B_v$ | Vertical bias field | −20 | 20 | G |

Outside these the models extrapolate; `simulate.py` warns when a parameter is out of range
and `optimise.py` cannot leave them. Vectors are produced at the threshold plane
`τ_z = 30 mm`, for AtomECS runs with the pipe radius extended to 7.85 mm so that the
training data is not truncated by the aperture. Applying a narrower physical aperture is
then a cut on the generated samples, which is what the objective does.

## Files

```
aion_model.py     model definitions, loading, sampling
objective.py      capturable flux, and the soft-penalty form for comparison
simulate.py       evaluate one parameter set
optimise.py       Bayesian optimisation over the nine parameters
params.json       parameter set and objective constants
models/           weights, normalisation, and their provenance
```
