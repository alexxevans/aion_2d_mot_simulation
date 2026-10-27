"""
Objective functions for optimising the 2D MOT source.

The default, `capturable_flux`, is the loading rate of the following 3D MOT: the
transmitted flux weighted by the acceptance of the next trap. This is the figure of
merit the 2D MOT source literature uses --- Tiecke et al. (Phys. Rev. A 80, 013409)
define an overall efficiency as the ratio of the 3D MOT loading rate to the capturable
flux from the oven; Yang et al. (arXiv:1505.04507) optimise every stage of a strontium
source "in terms of the atom loading rate"; Schoser et al. (Phys. Rev. A 66, 023410)
quote integrated brightness, flux per unit divergence. In all of them an atom is either
inside the capture range or it is not: there is no soft penalty on speed.

Formally this is Eq. (18) of Tiecke et al., the velocity-averaged transfer efficiency
    chibar_t(alpha_z, v_c) = int_0^{v_c} chi_t(v_z/v_t) phi_0(v_z, alpha_z) dv_z
evaluated by Monte Carlo over a learned joint p(x | theta) instead of over a fitted
axial distribution and a closed-form transverse efficiency. Their chi_t assumes a point
source, a single isotropic transverse velocity scale and separability in v_z; none of
those are needed here.

An atom counts if five conditions hold.

  1. it is slow enough to be captured,                   v_z <= v_c
  2. it is inside the differential pumping tube at the threshold plane
  3. it is still inside the tube at the end of that tube
  4. it arrives inside the capture volume of the 3D MOT
  5. it is not diverging too fast to be captured on arrival

(2) and (3) together enforce containment along the whole remaining tube: over
`pipe_remaining_m` the gravitational sag is below 0.1 mm even at the slowest velocities
in the training set, so the trajectory is straight to well within the tube radius, and
the norm of an affine function attains its maximum on an interval at an endpoint.
Checking only (3), as earlier versions did, admits atoms that were never inside the tube
-- which matters because the surrogate was trained with the pipe opened to 7.85 mm, so
the physical 1.5 mm constraint has to be reimposed in full here.

(5) is the criterion of Barbiero et al., Phys. Rev. Appl. 13, 014013 (2020), who for a
strontium source count an atom as captured "if the divergence of the atomic trajectory
computed along the push direction is lower that the geometrical constraint imposed by
the MOT capture angle theta_MOT = 16 mrad and if the final longitudinal velocity is
below the MOT capture velocity". Without it an atom that lands on the trap centre
carrying several m/s of transverse velocity is counted, and physically it flies through.
The divergence is evaluated on arrival, not at the threshold plane, because gravity
changes v_y appreciably over the drift.

Conditions 3 and 4 are evaluated ballistically, including gravity, which introduces the
coupling a moment-based objective misses: the divergence angle is v_perp / v_z, so for a
fixed transverse velocity a *slower* beam diverges more, and it also sags further, as
(1/2) g (L/v_z)^2. Gravity is not optional -- switching it off raises the accepted
fraction on the seed ground truth from 0.80% to 1.07%, a 25% effect. There is therefore
an interior optimum in the *beam's* mean axial velocity rather than a monotonic
preference for slow atoms; per atom, faster is better right up to the cut. Tiecke et al.
observe the same thing from the other side, their recapture fraction being "limited by
beam divergence".

`weighted_penalty` reproduces the earlier soft-penalty objective for comparison. It is
retained because the two rank configurations very differently, which is worth reporting,
not because it is recommended.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

G = 9.80665                                  # m/s^2


def _flight_time(v: pd.DataFrame, distance_m: float) -> np.ndarray:
    """Time of flight over `distance_m` along the beam axis, seconds."""
    return distance_m / np.clip(v["Vz"].to_numpy(), 1e-6, None)


def _propagate(v: pd.DataFrame, distance_m: float, *,
               g_perp: float = G) -> np.ndarray:
    """Radial displacement (mm) after ballistic flight over `distance_m`, from the
    threshold plane, including gravitational sag. X, Y in mm; velocities in m/s.

    `g_perp` is the component of gravity transverse to the beam axis. It is less than
    `G` when the beamline is not horizontal; see `beam_tilt_rad` in `capturable_flux`.
    """
    t = _flight_time(v, distance_m)
    x = v["X"].to_numpy() + v["Vx"].to_numpy() * t * 1e3       # mm
    y = v["Y"].to_numpy() + v["Vy"].to_numpy() * t * 1e3 - 0.5 * g_perp * t**2 * 1e3
    return np.hypot(x, y)


def _divergence_on_arrival(v: pd.DataFrame, distance_m: float, *,
                           g_perp: float = G) -> np.ndarray:
    """|v_perp| / v_z at the end of the drift, radians. Gravity has added -g_perp t to
    the transverse velocity by the time the atom arrives."""
    t = _flight_time(v, distance_m)
    vx = v["Vx"].to_numpy()
    vy = v["Vy"].to_numpy() - g_perp * t
    return np.hypot(vx, vy) / np.clip(v["Vz"].to_numpy(), 1e-6, None)


def acceptance(vectors: pd.DataFrame, *, capture_velocity: float,
               capture_radius_mm: float, drift_distance_m: float,
               pipe_radius_mm: float, pipe_remaining_m: float,
               capture_angle_rad: float | None = None,
               require_in_pipe_at_plane: bool = True,
               g_perp: float = G) -> np.ndarray:
    """Boolean acceptance per atom. See the module docstring for the five conditions.

    `capture_angle_rad=None` and `require_in_pipe_at_plane=False` recover the earlier
    three-condition form, for reproducing superseded results.
    """
    if vectors.empty:
        return np.zeros(0, dtype=bool)
    ok = vectors["Vz"].to_numpy() <= capture_velocity
    if require_in_pipe_at_plane:
        ok &= np.hypot(vectors["X"].to_numpy(),
                       vectors["Y"].to_numpy()) <= pipe_radius_mm
    ok &= _propagate(vectors, pipe_remaining_m, g_perp=g_perp) <= pipe_radius_mm
    ok &= _propagate(vectors, drift_distance_m, g_perp=g_perp) <= capture_radius_mm
    if capture_angle_rad is not None:
        ok &= _divergence_on_arrival(vectors, drift_distance_m,
                                     g_perp=g_perp) <= capture_angle_rad
    return ok


def capturable_flux(transmission_probability: float, vectors: pd.DataFrame, *,
                    capture_velocity: float = 30.0,
                    capture_radius_mm: float = 7.5,
                    drift_distance_m: float = 0.3675,
                    pipe_radius_mm: float = 1.5,
                    pipe_remaining_m: float = 0.015,
                    capture_angle_rad: float | None = 0.016,
                    require_in_pipe_at_plane: bool = True,
                    beam_tilt_rad: float = 0.3787) -> dict[str, float]:
    """Fraction of the atoms leaving the oven that are delivered into the capture
    volume of the 3D MOT.

    `transmission_probability` is P(theta), the fraction of emitted atoms reaching the
    threshold plane, so the objective is the dimensionless efficiency P(theta)<A>.
    Multiplying by the number emitted per run recovers a count; multiplying by the
    oven's emission rate recovers a loading rate in atoms per second. Neither is done
    here: the oven rate is not a simulation parameter, and estimating it from vapour
    pressure and aperture geometry is uncertain by more than an order of magnitude.

    Defaults and their provenance:
      capture_velocity   30 m/s   strontium blue (461 nm) 3D MOT, Hussain et al. 2026,
                                  quoted there as an estimate with no derivation. The
                                  published range for Sr is 30-60 m/s (Barbiero 2020:
                                  60; arXiv:2509.13764: ~50; Tiecke fitted 45.5 for Li).
                                  AION's 15 mm beams at 18-36 mW/cm^2 are at the low end
                                  of beam size and intensity, so a low v_c is expected.
                                  SWEEP THIS: J is most sensitive to it, because the
                                  optimum pins against the cut rather than sitting
                                  inside it.
      capture_radius_mm  7.5 mm   half of the 15 mm 3D MOT beams, AION vacuum-system
                                  paper (arXiv:2305.20060). Tiecke measured R_a = 9 mm
                                  and arXiv:2509.13764 uses r_trap = 9 mm, so this is
                                  the conservative end. Assumes "15 mm" is a diameter,
                                  and that capture extends to the geometric edge of the
                                  beam overlap; it does not, quite.
      capture_angle_rad  16 mrad  MOT capture angle, Barbiero et al. 2020. Set to None
                                  to drop condition 5.
      pipe_radius_mm     1.5 mm   3 mm differential pumping tube, AION vacuum paper, and
                                  the r_pipe of the AtomECS runs the surrogate was
                                  trained on.
      pipe_remaining_m   0.015    the threshold plane IS the tube entrance, so the
                                  whole tube lies ahead and this is its full length, not
                                  a remainder. 15 mm is the AION differential-pumping
                                  aperture length from the vacuum-system paper
                                  (arXiv:2305.20060), "3 mm diameter and 15 mm length",
                                  the same sentence that gives pipe_radius_mm. The tube
                                  is imposed retrospectively on the recorded vectors --
                                  the training runs used a wide aperture -- so the
                                  apparatus value is the right one to use here rather
                                  than the simulation's own bound.
      drift_distance_m   0.3675 m distance from the 30 mm threshold plane to the 3D MOT
                                  centre, derived from the 397.5 mm displacement in
                                  atomecs/sampling/raw rust/mot3d.rs. That vector is
                                  normalised by the code, so its magnitude is inferred,
                                  not used; corroborated by the CAD render only.
      beam_tilt_rad      0.0      angle of the beam axis below horizontal. The CAD
                                  render (Fig. 3 of the paper) shows the beamline
                                  running downhill, and the mot3d.rs input direction
                                  (355, -147, 102) implies about 21 degrees. With a
                                  tilt, only g*cos(tilt) acts transverse to the beam;
                                  the longitudinal component g*sin(tilt) changes v_z by
                                  under 1% over the drift and is neglected.
                                  0.3787 rad = 21.7 deg, from the mot3d.rs input
                                  direction; confirm against the AION design.

    On the seed ground truth the accepted fraction is 0.80% with the earlier
    three-condition form, 0.71% adding the threshold-plane cut, and 0.47% adding the
    divergence criterion.
    """
    if transmission_probability <= 0 or vectors.empty:
        return {"objective": 0.0, "accepted_fraction": 0.0, "delivered_fraction": 0.0,
                "mean_vz": 0.0, "mean_speed": 0.0, "radial_std_mm": 0.0,
                "mode_vz": 0.0}
    g_perp = G * np.cos(beam_tilt_rad)
    ok = acceptance(vectors, capture_velocity=capture_velocity,
                    capture_radius_mm=capture_radius_mm,
                    drift_distance_m=drift_distance_m,
                    pipe_radius_mm=pipe_radius_mm,
                    pipe_remaining_m=pipe_remaining_m,
                    capture_angle_rad=capture_angle_rad,
                    require_in_pipe_at_plane=require_in_pipe_at_plane,
                    g_perp=g_perp)
    frac = float(ok.mean())
    kept = vectors[ok]
    speed = np.sqrt(kept["Vx"]**2 + kept["Vy"]**2 + kept["Vz"]**2) if len(kept) else pd.Series([0.0])
    # mode of the UNCONDITIONED axial distribution: the alpha_z of the source
    # literature, and the statistic comparable with published beam velocities. The
    # conditional mean below is not, since acceptance favours faster atoms.
    vz_all = vectors["Vz"].to_numpy()
    hist, edges = np.histogram(vz_all, bins=np.linspace(0, max(vz_all.max(), 1e-6), 121))
    mode_vz = float(0.5 * (edges[:-1] + edges[1:])[hist.argmax()]) if hist.sum() else 0.0
    return {
        "objective": transmission_probability * frac,
        "accepted_fraction": frac,
        "delivered_fraction": transmission_probability * frac,
        "mean_vz": float(kept["Vz"].mean()) if len(kept) else 0.0,
        "mode_vz": mode_vz,
        "mean_speed": float(speed.mean()),
        "radial_std_mm": float(np.hypot(kept["X"], kept["Y"]).std(ddof=0)) if len(kept) > 1 else 0.0,
    }


def weighted_penalty(transmission_count: float, vectors: pd.DataFrame, *,
                     pipe_radius_mm: float = 1.5,
                     velocity_weight: float = 0.25,
                     displacement_weight: float = 1.1) -> dict[str, float]:
    """The earlier soft-penalty objective, kept for comparison only.

    J = N P_pipe / (w_v <v> + w_r sigma_r), with the mean speed and the radial spread
    taken at the threshold plane over atoms inside the pipe radius. It uses two moments
    rather than the distribution, applies no capture threshold, and rewards low speed
    without limit, so it prefers beams too slow to survive the drift.
    """
    if transmission_count <= 0 or vectors.empty:
        return {"objective": 0.0, "in_pipe_fraction": 0.0, "mean_speed": 0.0,
                "radial_std_mm": 0.0}
    r = np.hypot(vectors["X"], vectors["Y"])
    inside = r <= pipe_radius_mm
    frac = float(inside.mean())
    kept = vectors[inside]
    if kept.empty:
        return {"objective": 0.0, "in_pipe_fraction": frac, "mean_speed": 0.0,
                "radial_std_mm": 0.0}
    speed = float(np.sqrt(kept["Vx"]**2 + kept["Vy"]**2 + kept["Vz"]**2).mean())
    sigma = float(np.hypot(kept["X"], kept["Y"]).std(ddof=0)) if len(kept) > 1 else 0.0
    cost = velocity_weight * speed + displacement_weight * sigma
    return {"objective": (transmission_count * frac) / cost if cost > 0 else 0.0,
            "in_pipe_fraction": frac, "mean_speed": speed, "radial_std_mm": sigma}


OBJECTIVES = {"capturable_flux": capturable_flux, "weighted_penalty": weighted_penalty}
