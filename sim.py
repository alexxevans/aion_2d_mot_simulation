import torch
import numpy as np
import joblib
import normflows as nf
import pandas as pd
import pickle
import time
import os
import json

# Define the DNN Model
class DNN(torch.nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = torch.nn.Linear(9, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# CNF Model Definition
def create_cnf_model(latent_size, conditioning_size):
    K = 8
    hidden_units = 128
    hidden_layers = 4

    flows = []
    for i in range(K):
        flows += [nf.flows.CoupledRationalQuadraticSpline(
            latent_size, hidden_layers, hidden_units,
            num_context_channels=conditioning_size)]
        flows += [nf.flows.LULinearPermute(latent_size)]

    q0 = nf.distributions.DiagGaussian(latent_size, trainable=False)
    model = nf.ConditionalNormalizingFlow(q0, flows, None)
    return model

# Function to unnormalise data
def unnormalise_data(samples, normalisation_params):
    for col in ['X', 'Y', 'Vx', 'Vy', 'Vz']:
        mean, std = normalisation_params[col]
        samples[col] = samples[col] * std + mean
    return samples

# Load DNN model and scaler
dnn_model = DNN()
dnn_model.load_state_dict(torch.load('models/DNN_LBFGS_best_model.pth'))
dnn_model.eval()
dnn_scaler = joblib.load('models/dnn_scaler.pkl')

# Load CNF model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnf_model = create_cnf_model(latent_size=5, conditioning_size=9)
cnf_model.load_state_dict(torch.load('models/CNF_random.pth', map_location=device))
cnf_model.eval()

# Load normalization parameters for CNF
with open('models/CNF_mol_normalisation_params.pkl', 'rb') as file:
    normalisation_params = pickle.load(file)

# Define the conditioning columns
conditioning_columns = [
    'cooling_beam_detuning', 'cooling_beam_radius', 'cooling_beam_power_mw',
    'push_beam_detuning', 'push_beam_radius', 'push_beam_power',
    'push_beam_offset', 'quadrupole_gradient', 'vertical_bias_field'
]

def normalise_cnf_inputs(params, normalisation_params):
    normalised_params = []
    for param, col in zip(params, conditioning_columns):
        mean, std = normalisation_params[col]
        normalised_param = (param - mean) / std
        normalised_params.append(normalised_param)
    return normalised_params

def predict_and_generate(params, total_atoms, filename, radius=1.5, offset=0):
    # DNN Prediction
    params_2d = np.array(params).reshape(1, -1)
    params_df = pd.DataFrame(params_2d, columns=conditioning_columns)
    dnn_scaled_params = dnn_scaler.transform(params_df)
    input_tensor = torch.tensor(dnn_scaled_params, dtype=torch.float32)
    with torch.no_grad():
        output = dnn_model(input_tensor)

    start_time = time.time()
    initial_prediction = max(0, output.item()) / 5e6  # normalise the prediction
    num_vectors = round(initial_prediction * total_atoms)

    print(f"Initial predicted number of atoms (vectors): {num_vectors}")

    # CNF Generation
    if num_vectors > 0:
        # normalise inputs for CNF
        cnf_normalised_params = normalise_cnf_inputs(params, normalisation_params)
        context_data = torch.tensor(cnf_normalised_params, dtype=torch.float32).to(device).unsqueeze(0)
        # Repeat context_data to match num_vectors
        context_data = context_data.repeat(num_vectors, 1)

        # Generate all samples at once for efficiency
        with torch.no_grad():
            samples, _ = cnf_model.sample(num_vectors, context_data)
            all_samples = samples.cpu().numpy()

        df_samples = pd.DataFrame(all_samples, columns=['X', 'Y', 'Vx', 'Vy', 'Vz'])
        df_samples = unnormalise_data(df_samples, normalisation_params)
        df_samples['Vz'] = np.expm1(df_samples['Vz'])  # Revert log transformation for Vz

        # Calculate distance from offset position
        df_samples['distance'] = np.sqrt((df_samples['X'] - offset) ** 2 + df_samples['Y'] ** 2)

        # Filter samples within the specified radius
        within_radius = df_samples['distance'] <= radius
        df_within_radius = df_samples[within_radius]

        percentage_within_radius = within_radius.mean()

        # Adjust prediction
        final_prediction = num_vectors * percentage_within_radius
        final_probability = final_prediction / total_atoms

        end_time = time.time()
        total_time = end_time - start_time

        print(f"Time taken to generate vectors: {total_time:.2f} seconds")
        print(f"Percentage of atoms within radius {radius} mm centered at X = {offset} mm: {percentage_within_radius:.2%}")
        print(f"Final prediction: {final_prediction}")
        print(f"Final probability: {final_probability:.6f}")

        # Save only the samples within the specified radius
        df_within_radius = df_within_radius.drop(columns=['distance'])
        df_within_radius.to_csv(filename, index=False)
        print(f"Generated samples within radius {radius} mm centered at X = {offset} mm saved to '{filename}'")

        return final_prediction, df_within_radius
    else:
        print("No atoms predicted. No samples generated.")
        return 0, pd.DataFrame(columns=['X', 'Y', 'Vx', 'Vy', 'Vz'])

# Load parameters from JSON file
with open('params.json', 'r') as f:
    data = json.load(f)

params_dict = data['params']
total_atoms = data['total_atoms']
radius = data['pipe_radius']
offset = data['pipe_offset']

# Ensure the parameters are in the correct order
params = [params_dict[key] for key in conditioning_columns]

# Specify the filename to save the generated samples
filename = 'generated_samples.csv'

# Call the function to predict and generate samples
final_prediction, df_samples = predict_and_generate(params, total_atoms, filename, radius=radius, offset=offset)