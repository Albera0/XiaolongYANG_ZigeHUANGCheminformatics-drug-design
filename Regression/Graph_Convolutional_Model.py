import sys 
sys.path.append(".") 
import Read_Data as rd
from Preprocess import GraphFeature
import Draw_Figure as drf
import torch
import torch.nn.functional as F
from torch.nn import GRU
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, MLP, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import numpy as np
from pytorch_lightning.loggers import CSVLogger
import pandas as pd

#Data Loading
lipo_df, smiles_list, y = rd.DataRead()
print(y[:10])
#create a new list  to list of non-canonical SMILES
canonical_smiles = [rd.CanonicalizeSmiles(smiles) for smiles in smiles_list]

graph_feature = GraphFeature(canonical_smiles, y).shuffle()

# Normalize target to mean = 0 and std = 1.
mean = graph_feature.data.y.mean()
std = graph_feature.data.y.std()
graph_feature.data.y = (graph_feature.data.y - mean) / std
mean, std = mean.item(), std.item()

#GNN model
class MPNN(pl.LightningModule):
    def __init__(self, hidden_dim, out_dim,
                train_data, valid_data, test_data,
                std, batch_size=32, lr=1e-3, 
                dropout=0.1):
        
        super().__init__()
        self.std = std  # std of data's target
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout

        # Initial layers
        self.atom_emb = AtomEncoder(emb_dim=hidden_dim)
        self.bond_emb = BondEncoder(emb_dim=hidden_dim)

        # Message passing layers
        nn = MLP([hidden_dim, hidden_dim*2, hidden_dim*hidden_dim])
        self.conv = NNConv(hidden_dim, hidden_dim, nn, aggr='mean')
        self.gru = GRU(hidden_dim, hidden_dim)

        # Readout layers
        self.mlp = MLP([hidden_dim, int(hidden_dim/2), out_dim])

    def forward(self, data, mode="train"):

        # Initialization
        x = self.atom_emb(data.x)
        h = x.unsqueeze(0)
        edge_attr = self.bond_emb(data.edge_attr)

        # Message passing
        for i in range(3):
            m = F.relu(self.conv(x, data.edge_index, edge_attr))  # send message and aggregation
            x, h = self.gru(m.unsqueeze(0), h)  # node update
            x = x.squeeze(0)

            # x_new, h = self.gru(m.unsqueeze(0), h)  # node update
            # x_new = x_new.squeeze(0)

            # # Adding dropout and skip connection
            # x = x + x_new
            # if mode == "train":
            #     x = F.dropout(x, p=self.dropout, training=self.training)

        # Readout
        x = global_add_pool(x, data.batch)
        x = self.mlp(x)

        return x.view(-1)

    def training_step(self, batch, batch_idx):
        # Here we define the train loop.
        out = self.forward(batch, mode="train")
        loss = F.mse_loss(out, batch.y)
        self.log("Train loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Define validation step. At the end of every epoch, this will be executed
        out = self.forward(batch, mode="valid")
        loss = F.mse_loss(out * self.std, batch.y * self.std)  # report MSE
        self.log("Valid MSE", loss, on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # What to do in test
        out = self.forward(batch, mode="test")
        loss = F.mse_loss(out * self.std, batch.y * self.std)  # report MSE
        self.log("Test MSE", loss)

    def configure_optimizers(self):
        # Here we configure the optimization algorithm.
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr
        )
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
    
#Split data for training randomly
def RandomSplitter(dataset, frac_train, frac_valid, frac_test):

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    #Set random seed as 42
    np.random.seed(42)
    num_datapoints = len(dataset)
    train_cutoff = int(frac_train * num_datapoints)
    valid_cutoff = int((frac_train + frac_valid) * num_datapoints)
    shuffled = np.random.permutation(range(num_datapoints))
    return (
        shuffled[:train_cutoff],
        shuffled[train_cutoff:valid_cutoff],
        shuffled[valid_cutoff:],
        )

#Data preprocess finish, start prepare for training.
#Split the trianing testing and validation set.
train_idx, valid_idx, test_idx = RandomSplitter(graph_feature, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
train_dataset = graph_feature[train_idx]
valid_dataset = graph_feature[valid_idx]
test_dataset = graph_feature[test_idx]

#model information
gnn_model = MPNN(
    hidden_dim=64,
    out_dim=1,
    std=std,
    train_data=train_dataset,
    valid_data=valid_dataset,
    test_data=test_dataset,
    lr=1e-3,
    batch_size=64,
    dropout=0.1
)

#Check the modelstructure
print(gnn_model)

#Ask GPT to record losses
logger = CSVLogger("logs", name="mpnn_experiment")

#Training the model
trainer = pl.Trainer(max_epochs = 100, logger=logger)
trainer.fit(model=gnn_model)

# Load metrics from CSV
metrics_path = f"{logger.log_dir}/metrics.csv"
metrics = pd.read_csv(metrics_path)
metrics = metrics.groupby('epoch').last().reset_index()
epochs = metrics['epoch']
train_losses =  metrics['Train loss']
valid_losses =  metrics['Valid MSE']

#Check training & validation loss
drf.LossVis(epochs, train_losses,  valid_losses,"MPNN")

#Predicted and True values comperasion
gnn_model.eval()
y_true, y_pred = [], []

for batch in gnn_model.test_dataloader():
    with torch.no_grad():
        #Predicted values
        out = gnn_model(batch)
        y_true.append(batch.y * gnn_model.std)
        y_pred.append(out * gnn_model.std)

y_true = torch.cat(y_true).cpu().numpy()
y_pred = torch.cat(y_pred).cpu().numpy()

#Draw figure
drf.PredictedTrue(y_true, y_pred, "MPNN")

#Test the model on the test set
results = trainer.test(ckpt_path="best")

#Compuate teh test RMSE
test_mse = results[0]["Test MSE"]
test_rmse = test_mse ** 0.5
print(f"\nMPNN model performance: RMSE on test set = {test_rmse:.4f}.\n")

