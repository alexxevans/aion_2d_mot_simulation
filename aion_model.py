"""
Factorised AI surrogate of the AION 2D+ magneto-optical trap.

Two models, applied in sequence for a given set of nine control parameters theta:

  P(theta)         a deep neural network predicting the number of atoms transmitted
                   past the threshold plane, out of N_atoms launched from the oven
  p(x | theta)     a conditional normalising flow with denoising schedule (CNF-DS)
                   generating transmitted-atom vectors x = (X, Y, Vx, Vy, Vz)

Together they replace an AtomECS run costing ~22 minutes per parameter set.

Reference: Evans, Bentine, Mieskolainen and Baynham, "Generative Surrogates for Cold-Atom
Interferometry" (2026).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import normflows as nf
except ModuleNotFoundError as exc:                      # pragma: no cover
    raise ModuleNotFoundError(
        "normflows is required. Create the environment with "
        "`conda env create -f environment.yml`."
    ) from exc

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"

THETA_COLUMNS = [
    "cooling_beam_detuning", "cooling_beam_radius", "cooling_beam_power_mw",
    "push_beam_detuning", "push_beam_radius", "push_beam_power",
    "push_beam_offset", "quadrupole_gradient", "vertical_bias_field",
]
VECTOR_COLUMNS = ["X", "Y", "Vx", "Vy", "Vz"]

# Sobol ranges the surrogate was trained over. Outside these the models extrapolate
# and the output should not be trusted.
THETA_RANGES = {
    "cooling_beam_detuning": (-250.0, 0.0),      # MHz
    "cooling_beam_radius":   (7.0, 15.0),        # mm
    "cooling_beam_power_mw": (50.0, 350.0),      # mW
    "push_beam_detuning":    (-350.0, 0.0),      # MHz
    "push_beam_radius":      (0.0, 3.0),         # mm
    "push_beam_power":       (0.0, 20.0),        # mW
    "push_beam_offset":      (0.0, 5.0),         # mm
    "quadrupole_gradient":   (0.0, 100.0),       # G/cm
    "vertical_bias_field":   (-20.0, 20.0),      # G
}

# Atoms launched per AtomECS run; the DNN predicts a count out of this number.
N_ATOMS_TRAINING = 5.0e6


class DNN(nn.Module):
    """Transmission-count network, 9-128-64-32-1 with ReLU (11,649 parameters)."""

    def __init__(self, input_dim: int = 9):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)


def build_cnf(latent_size: int = 5, context_size: int = 9, k: int = 8,
              hidden_layers: int = 4, hidden_units: int = 128):
    """Conditional normalising flow: k blocks of a coupled rational-quadratic spline
    followed by an LU linear permutation, over a fixed diagonal Gaussian base."""
    flows = []
    for _ in range(k):
        flows += [nf.flows.CoupledRationalQuadraticSpline(
            latent_size, hidden_layers, hidden_units, num_context_channels=context_size)]
        flows += [nf.flows.LULinearPermute(latent_size)]
    q0 = nf.distributions.DiagGaussian(latent_size, trainable=False)
    return nf.ConditionalNormalizingFlow(q0, flows, None)


def load_normalisation(path: Optional[Path] = None) -> dict[str, tuple[float, float]]:
    """(mean, std) per column, fitted on the training split only."""
    path = path or (MODELS / "normalisation_params.csv")
    frame = pd.read_csv(path)
    return {c: tuple(float(v) for v in ast.literal_eval(frame[c].values[0]))
            for c in frame.columns}


class Surrogate:
    """The complete surrogate. Load once, then call `sample` per parameter set."""

    def __init__(self, models_dir: Optional[Path] = None, device: Optional[str] = None,
                 require_dnn: bool = True):
        self.dir = Path(models_dir) if models_dir else MODELS
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.norm = load_normalisation(self.dir / "normalisation_params.csv")

        self.cnf = build_cnf().to(self.device)
        self.cnf.load_state_dict(torch.load(
            self.dir / "cnf_ds.pth", map_location=self.device, weights_only=False))
        self.cnf.eval()

        self.dnn = None
        self.dnn_scaler = None
        dnn_path, scaler_path = self.dir / "dnn.pth", self.dir / "dnn_scaler.pkl"
        if dnn_path.exists() and scaler_path.exists():
            import joblib
            self.dnn = DNN().to(self.device)
            self.dnn.load_state_dict(torch.load(
                dnn_path, map_location=self.device, weights_only=False))
            self.dnn.eval()
            self.dnn_scaler = joblib.load(scaler_path)
        elif require_dnn:
            raise FileNotFoundError(
                f"Deterministic model not found. Expected {dnn_path} and {scaler_path}.\n"
                "See models/PROVENANCE.md for what these files are and where they come from."
            )

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def check_ranges(theta: dict[str, float]) -> list[str]:
        """Names of any parameters outside the trained Sobol ranges."""
        return [k for k, v in theta.items()
                if k in THETA_RANGES and not (THETA_RANGES[k][0] <= v <= THETA_RANGES[k][1])]

    def _context(self, theta: dict[str, float], n: int) -> torch.Tensor:
        row = [(theta[c] - self.norm[c][0]) / self.norm[c][1] for c in THETA_COLUMNS]
        return torch.tensor([row], dtype=torch.float32, device=self.device).expand(n, -1)

    def _unnormalise(self, arr: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(arr, columns=VECTOR_COLUMNS)
        for c in VECTOR_COLUMNS:
            mean, std = self.norm[c]
            df[c] = df[c] * std + mean
        df["Vz"] = np.expm1(df["Vz"])            # the flow is trained on ln(1 + Vz)
        return df

    # ------------------------------------------------------------ public API
    def transmission(self, theta: dict[str, float], n_atoms: float = N_ATOMS_TRAINING) -> float:
        """Expected number of atoms transmitted past the threshold plane."""
        if self.dnn is None:
            raise RuntimeError("Deterministic model not loaded; see models/PROVENANCE.md.")
        frame = pd.DataFrame([[theta[c] for c in THETA_COLUMNS]], columns=THETA_COLUMNS)
        x = torch.tensor(self.dnn_scaler.transform(frame), dtype=torch.float32,
                         device=self.device)
        with torch.no_grad():
            count = float(self.dnn(x).item())
        return max(0.0, count) / N_ATOMS_TRAINING * n_atoms

    def sample(self, theta: dict[str, float], n: int, batch: int = 200_000) -> pd.DataFrame:
        """Draw n transmitted-atom vectors from p(x | theta), in physical units
        (X, Y in mm; Vx, Vy, Vz in m/s)."""
        if n <= 0:
            return pd.DataFrame(columns=VECTOR_COLUMNS)
        out, remaining = [], int(n)
        with torch.no_grad():
            while remaining > 0:
                m = min(batch, remaining)
                s, _ = self.cnf.sample(m, self._context(theta, m))
                out.append(s.detach().cpu().numpy())
                remaining -= m
        return self._unnormalise(np.concatenate(out, axis=0))

    def evaluate(self, theta: dict[str, float], n_atoms: float = N_ATOMS_TRAINING,
                 n_estimate: int = 20_000) -> tuple[float, pd.DataFrame]:
        """Transmission count, and vectors for estimating distribution properties.

        `n_estimate` vectors are drawn regardless of the predicted count. The count sets
        how many atoms a run delivers; the sampled vectors estimate what *fraction* of
        them are useful, and that fraction is a property of p(x | theta) which can be
        estimated to whatever precision is affordable. Sampling only `count` vectors
        would tie the Monte Carlo error of the objective to the yield, so that exactly
        the sparse configurations most in need of a reliable estimate would get the
        noisiest one. On a GPU the flow is launch-dominated, so a larger `n_estimate`
        costs almost nothing.
        """
        count = self.transmission(theta, n_atoms)
        return count, self.sample(theta, int(n_estimate))
