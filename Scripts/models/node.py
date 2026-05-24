import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from models.node_lib.arch import DenseBlock


class NODEModel:
    def __init__(
        self,
        num_layers=4,
        layer_dim=128,
        depth=6,
        tree_dim=2,
        epochs=50,
        batch_size=512,
        lr=0.001
    ):
        self.num_layers = num_layers
        self.layer_dim = layer_dim
        self.depth = depth
        self.tree_dim = tree_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.model = None

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        y_encoded = np.searchsorted(self.classes_, y)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y_encoded, dtype=torch.long)

        n_features = X.shape[1]
        n_classes = len(np.unique(y))

        dense = DenseBlock(
            input_dim=n_features,
            layer_dim=self.layer_dim,
            num_layers=self.num_layers,
            tree_dim=self.tree_dim,
            depth=self.depth,
            flatten_output=True
        )

        output_dim = self.num_layers * self.layer_dim * self.tree_dim

        classifier = nn.Linear(output_dim, n_classes)

        self.model = nn.Sequential(
            dense,
            classifier
        ).to(self.device)

        # NODE data-aware initialization
        init_size = min(1000, len(X))
        dense.to(self.device)
        dense.eval()
        with torch.no_grad():
            dense(X[:init_size].to(self.device))
        dense.train()

        loader = DataLoader(
            TensorDataset(X, y),
            batch_size=self.batch_size,
            shuffle=True
        )

        optimizer = Adam(
            self.model.parameters(),
            lr=self.lr
        )

        class_counts = np.bincount(y_encoded)
        class_weights = len(y_encoded) / (len(class_counts) * class_counts)
        class_weights = torch.tensor(class_weights,dtype=torch.float32).to(self.device)

        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        self.model.train()

        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                y_pred = self.model(xb)
                loss = loss_fn(y_pred, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        self.model.eval()

        with torch.no_grad():
            y_pred = self.model(X)

        pred_idx = torch.argmax(y_pred, dim=1).cpu().numpy()
        return self.classes_[pred_idx]