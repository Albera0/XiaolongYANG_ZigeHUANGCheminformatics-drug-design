import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from tqdm import tqdm
import hdbscan


# reload environmental path(revised by ai)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



from Preprocess import load_hiv_dataset
from Read_Data import DatasetFilter
def get_dataset_filter(dataset):
    return DatasetFilter(dataset)
from Clustering.DBSCAN import MolecularDescriptorClustering


# banalance data

print("\n" + "=" * 70)
print("Preparing balanced molecular descriptor data for Autoencoder training")
print("=" * 70 + "\n")


dataset = load_hiv_dataset("Dataset/HIV.csv")

# use DatasetFilter generate data
filter_tool = get_dataset_filter(dataset)
balanced_df = filter_tool.get_balanced_dataset(ratio=3)  # adjustable

# extract MolecularDescriptor
instance = MolecularDescriptorClustering(dataset)
instance.X = balanced_df.copy()
instance.prepare_features()

X_scaled = instance.X_scaled
labels = instance.X["HIV_active"].values

print(f"Shape of X_scaled: {X_scaled.shape}")
print(f"Number of samples: {X_scaled.shape[0]}, features: {X_scaled.shape[1]}")

# convert into PyTorch Tensor
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
tensor_dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)

# divide dataset
train_size = int(0.8 * len(tensor_dataset))
test_size = len(tensor_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(tensor_dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

#network configuration
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size)
        )
    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.net(x)


class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, input_size)
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


#Training
input_size = X_scaled.shape[1]
latent_size = 2
hidden_size = 32
epochs = 200
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Autoencoder(input_size, hidden_size, latent_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

os.makedirs("Models", exist_ok=True)
model_path = os.path.join("Models", "autoencoder_balanced.pth")

print("\n[INFO] Training Autoencoder...\n")
loss_history = {"train": [], "eval": []}

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for batch_x, _ in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{epochs}"):
        batch_x = batch_x.to(device)
        recon = model(batch_x)
        loss = loss_fn(recon, batch_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * batch_x.size(0)

    avg_train_loss = total_train_loss / len(train_dataset)
    loss_history["train"].append(avg_train_loss)

    # Eval
    model.eval()
    total_eval_loss = 0
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            recon = model(batch_x)
            total_eval_loss += loss_fn(recon, batch_x).item() * batch_x.size(0)
    avg_eval_loss = total_eval_loss / len(test_dataset)
    loss_history["eval"].append(avg_eval_loss)

    print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.5f} | Eval Loss: {avg_eval_loss:.5f}")

torch.save(model.state_dict(), model_path)
print(f"\nTraining complete. Model saved to {model_path}\n")

# feature extraction and visualization
model.eval()
with torch.no_grad():
    latent_features = model.encoder(torch.tensor(X_scaled, dtype=torch.float32).to(device)).cpu().numpy()

print(f"Latent feature shape: {latent_features.shape}")

# data analysis (same as DBSCAN)
print("\n" + "=" * 70)
print("Running HDBSCAN clustering on Autoencoder latent features")
print("=" * 70)

clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10, metric='euclidean')
cluster_labels = clusterer.fit_predict(latent_features)
instance.X["Cluster_HDBSCAN"] = cluster_labels

unique, counts = np.unique(cluster_labels, return_counts=True)
print("\nCluster distribution:")
for u, c in zip(unique, counts):
    print(f"  Cluster {u}: {c} molecules")

# visualization
plt.figure(figsize=(9, 7))
scatter = plt.scatter(latent_features[:, 0], latent_features[:, 1],
                      c=cluster_labels, cmap='Spectral', alpha=0.7, s=10)
plt.colorbar(scatter, label="Cluster ID")
plt.title("HDBSCAN Clustering on Autoencoder Latent Space (Balanced)")
plt.xlabel("Latent Dim 1")
plt.ylabel("Latent Dim 2")
plt.tight_layout()
plt.savefig("Figure/hdbscan_balanced_clusters.png", dpi=300)
plt.show()
print("[Saved] Figure/hdbscan_balanced_clusters.png")


activity_summary = instance.X.groupby("Cluster_HDBSCAN")["HIV_active"].mean()
print("\nAverage HIV activity per cluster:")
print(activity_summary)

plt.figure(figsize=(8, 5))
activity_summary.plot(kind='bar', color='teal')
plt.title("Average HIV Activity Ratio per Cluster (Balanced Data)")
plt.xlabel("Cluster ID")
plt.ylabel("Active Ratio")
plt.tight_layout()
plt.savefig("Figure/hdbscan_balanced_activity.png", dpi=300)
plt.show()
print("[Saved] Figure/hdbscan_balanced_activity.png")

print("\nBalanced Autoencoder + HDBSCAN completed successfully.")


print("\n" + "=" * 70)
print("Performing detailed high-activity cluster analysis (Scaffold / FG / Descriptors)")
print("=" * 70)


top_clusters = activity_summary.sort_values(ascending=False).head(3).index.tolist()
print(f"Top clusters by HIV activity: {top_clusters}")


analysis_instance = MolecularDescriptorClustering(dataset)
analysis_instance.X = instance.X.copy()
analysis_instance.cluster_labels = cluster_labels


try:
    analysis_instance.analyze_high_activity_clusters(clusters_to_analyze=top_clusters)
    print("\nScaffold / Functional Group / Descriptor analysis completed successfully.")
except Exception as e:
    print(f"\nAnalysis failed: {e}")

# loss visualization
plt.figure(figsize=(8, 5))
plt.plot(loss_history["train"], label="Train Loss", linewidth=2, color="steelblue")
plt.plot(loss_history["eval"], label="Validation Loss", linewidth=2, linestyle="--", color="darkorange")
plt.title("Autoencoder Training and Validation Loss Curve", fontsize=13)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("MSE Loss", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
