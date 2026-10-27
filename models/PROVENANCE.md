# Model artefacts

Checkpoints for the surrogate described in the paper (random seed 2026 of the three seeds
evaluated there). Other seeds, the BDT baseline, the CNF and diffusion models, and the
training code are available from the authors on request.

## `cnf_ds.pth` + `normalisation_params.csv`

CNF-DS, seed 2026.

- Architecture: 8 coupled rational-quadratic spline blocks, 4 hidden layers, 128 hidden units, with LU linear permutations.
- `normalisation_params.csv` holds the output normalisation fitted on the seed-2026 training split.
- SHA-256:
  - `cnf_ds.pth`: `72ab14407169d8c6c43461a1a2258f68dc724a746019b5f782b874445c642c0c`
  - `normalisation_params.csv`: `5d97b0475b807e9dc765e5882b5dcb4cd95658b8426d1a0b10b8233ddb224259`

## `dnn.pth` + `dnn_scaler.pkl`

DNN, seed 2026.

- Architecture: 9-128-64-32-1 ReLU network, 11,649 parameters; trained on an NVIDIA GeForce RTX 3080 with batch size 256.
- Test metrics on the held-out split: RMSE 9.94e-6, R² 0.9424, MAE 4.91e-6.
- SHA-256:
  - `dnn.pth`: `8ea4c9b7cd62aed4e4551e181035cb68bcfb42e1923c0e846e7c458131d347d9`
  - `dnn_scaler.pkl`: `6c133fdc66109a7a26afe90df71f2120b43cadef8b4e71de92e46213e64ad08f`
