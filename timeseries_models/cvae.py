import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append("../")
from device_data import get_meter_data

# Conditional variational autoencoder

def run():
    # === Load your data ===
    fname = "../data/Alburgh/2024-01-01_2024-12-31_South_Alburgh_Load_corrected.parquet"
    timeseries_df = pd.read_parquet(fname)   # contains 'asset_id', 'timestamp', 'value'
    timeseries_df = timeseries_df.astype(float)
    timeseries_df.columns = timeseries_df.columns.astype(int)
    #timeseries_df = pd.read_parquet("test_sample100.parquet")
    pivoted = timeseries_df.T
    device_flags_df, keys = get_meter_data()  # contains 'Meter Number', 'cchp', 'home charger', 'solar'

    # === Pivot time series to wide format ===
    # Assumption: time series are fixed-length per asset (e.g., 1 day or 1 week)
    #pivoted = timeseries_df.pivot(index='asset_id', columns='timestamp', values='value')
    #pivoted = pivoted.dropna()  # Drop incomplete records

    # === Match and align device flags ===
    device_flags_df = device_flags_df.rename(columns={'Meter Number': 'asset_id'})
    valid_ids = device_flags_df["asset_id"].str.isnumeric()
    device_flags_df = device_flags_df[valid_ids]
    device_flags_df["asset_id"] = device_flags_df["asset_id"].astype(int)
    device_flags_df.set_index("asset_id", inplace=True)

    xl_num = device_flags_df.index.unique()
    valid_ami = pivoted.index.isin(xl_num)
    pivoted.drop(index=pivoted.index[~valid_ami], inplace=True)
    device_flags_df = device_flags_df.loc[pivoted.index]
    features = device_flags_df[['cchp', 'home charger', 'solar']].astype(float)

    # === Normalize time series ===
    scaler = StandardScaler()
    load_data = scaler.fit_transform(pivoted.values)

    model = train_model(load_data, features, n_epochs=5)
    test_result(model, scaler)
    

# === Torch Dataset ===
class LoadDataset(Dataset):
    def __init__(self, loads, features):
        self.loads = torch.tensor(loads, dtype=torch.float32)
        self.features = torch.tensor(features.values, dtype=torch.float32)

    def __len__(self):
        return len(self.loads)

    def __getitem__(self, idx):
        return self.loads[idx], self.features[idx]

# === CVAE Model ===
class CVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def encode(self, x, c):
        h = self.encoder(torch.cat([x, c], dim=1))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, c)
        return x_hat, mu, logvar

# === Loss Function ===
def loss_function(x_hat, x, mu, logvar):
    recon = nn.functional.mse_loss(x_hat, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kld

# === Train the Model ===
def train_model(load_data, features, n_epochs=50):
    dataset = LoadDataset(load_data, features)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CVAE(input_dim=load_data.shape[1], cond_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_c in dataloader:
            batch_x, batch_c = batch_x.to(device), batch_c.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = model(batch_x, batch_c)
            loss = loss_function(x_hat, batch_x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss:.2f}")
    return model

# === Generate a New Sample ===
def generate_sample(model, device_flags_tensor, scaler):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        z = torch.randn(1, 16).to(device)
        cond = torch.tensor(device_flags_tensor, dtype=torch.float32).unsqueeze(0).to(device)
        generated = model.decode(z, cond)
        return scaler.inverse_transform(generated.cpu().numpy())[0]

def test_result(model, scaler):
    import matplotlib.pyplot as plt
    # Example: generate a household with all three devices
    sample_flags = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]]  # [cchp, home charger, solar]
    for s in sample_flags:
        sample_curve = generate_sample(model, s, scaler)
        plt.plot(sample_curve, label="{0}{1}{2}".format(*s))
    plt.title("Synthetic Load Curve")
    plt.xlabel("Time Step")
    plt.ylabel("Load (kW)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run()
