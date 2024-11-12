# aion_2d_mot_simulation

AI Model of the 2D+ Magneto-Optical Trap used in AION

The model consists of:

- A deep neural network to predict the probability of a generated atom being transmitted
- A CNF-PD (Conditional Normalising Flow with Pre-Diffusion) model, which uses an initial period of denoising where the data is gradually introduced from static noise, followed by standard CNF training with learning rate decay and early stopping rounds; this model samples the vectors

## How to Run

1. Create conda environment:
- Create the environment using the provided YAML file:
  ```
  conda env create -f environment.yml
  ```
- Activate the environment:
  ```
  conda activate 2d_mot_sim
  ```
2. Set the parameters in the `params.json` file.
3. Execute the `sim.py` script.

## Capabilities

- The model can only capture a pipe with up to 11mm on the x-axis and 7.75mm on the y-axis, the data is clipped to set pipe dimensions in params.json
- Currently the model is trained to predict data at the entrance to the pipe.
- The model is trained on data corresponding to the following parameter ranges:

| Symbol | Parameter                     | Minimum Value | Maximum Value | Units |
| ------ | ----------------------------- | ------------- | ------------- | ------ |
| $\delta_c$ | Cooling Beam Detuning         | -250          | 0             | MHz   |
| $P_{c}$ | Cooling Beam Power            | 50            | 350           | mW    |
| $w_c$ | Cooling Beam Waist           | 7             | 15            | mm    |
| $\delta_p$ | Push Beam Detuning            | -350          | 0             | MHz   |
| $P_{p}$ | Push Beam Power               | 0             | 20            | mW    |
| $w_p$ | Push Beam Waist              | 0             | 3             | mm    |
| $d_{p}$ | Push Beam Offset              | 0             | 5             | mm    |
| $\nabla B$ | Quadrupole Gradient           | 0             | 100           | G/cm  |
| $B_{v}$ | Vertical Bias Field           | -20           | 20            | G     |

Ensure that the values of the parameters are within these ranges to ensure stability. Going beyond these ranges may give unpredictable results.
