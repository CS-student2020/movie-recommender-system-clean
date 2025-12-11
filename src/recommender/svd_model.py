from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class RatingsDataset(Dataset):
    """
    PyTorch Dataset wrapping user, item, rating tensors.
    """

    def __init__(self, users, items, ratings):
        if not (len(users) == len(items) == len(ratings)):
            raise ValueError("User, item, rating tensors must have equal length.")

        self.users = users
        self.items = items
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            self.users[idx],
            self.items[idx],
            self.ratings[idx],
        )


class MatrixFactorization(nn.Module):
    """
    Lightweight MF/SVD-style model used for collaborative filtering.
    Automatically infers number of users/items from ratings DataFrame.

    Expected API (based on tests):
        model = MatrixFactorization(ratings_df, epochs=2)
        model.train(verbose=False)
        pred = model.predict(user_id, movie_id)
    """

    def __init__(
        self,
        ratings_df,
        embedding_dim=32,
        epochs=5,
        lr=0.01,
        batch_size=1024,
        device=None,
    ):
        super().__init__()

        self.ratings_df = ratings_df.copy()
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        # Auto-detect device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Map user/movie IDs to contiguous integer indices
        self.user_map = {uid: i for i, uid in enumerate(ratings_df["userId"].unique())}
        self.item_map = {iid: i for i, iid in enumerate(ratings_df["movieId"].unique())}

        self.num_users = len(self.user_map)
        self.num_items = len(self.item_map)

        # Embeddings
        self.user_emb = nn.Embedding(self.num_users, embedding_dim)
        self.item_emb = nn.Embedding(self.num_items, embedding_dim)

        # Move to device
        self.to(self.device)

        # Prepare training dataset
        user_tensor = torch.tensor(
            ratings_df["userId"].map(self.user_map).values, dtype=torch.long
        )
        item_tensor = torch.tensor(
            ratings_df["movieId"].map(self.item_map).values, dtype=torch.long
        )
        rating_tensor = torch.tensor(
            ratings_df["rating"].values, dtype=torch.float32
        )

        self.dataset = RatingsDataset(user_tensor, item_tensor, rating_tensor)
        self.dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True
        )

        # Loss + optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------
    def forward(self, user_ids, item_ids):
        u_vec = self.user_emb(user_ids)
        i_vec = self.item_emb(item_ids)
        return (u_vec * i_vec).sum(dim=1)

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    def train(self, verbose=True):
        """
        Train MF model for N epochs. Test only checks that training runs,
        not that loss converges.
        """
        super().train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for users, items, ratings in self.dataloader:
                users = users.to(self.device)
                items = items.to(self.device)
                ratings = ratings.to(self.device)

                preds = self.forward(users, items)
                loss = self.loss_fn(preds, ratings)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            if verbose:
                print(f"Epoch {epoch+1}/{self.epochs} – Loss: {epoch_loss:.4f}")

    # --------------------------------------------------------
    # Prediction API required by tests
    # --------------------------------------------------------
    def predict(self, user_id, item_id):
        """
        Return a predicted rating for a (userId, movieId) pair.
        """
        if user_id not in self.user_map or item_id not in self.item_map:
            raise ValueError("Unknown userId or movieId.")

        u = torch.tensor([self.user_map[user_id]], dtype=torch.long).to(self.device)
        i = torch.tensor([self.item_map[item_id]], dtype=torch.long).to(self.device)

        self.eval()
        with torch.no_grad():
            return self.forward(u, i).item()
"""
Compatibility facade for SVD-related models.

Historically, code imported RatingsDataset from:
    src.recommender.svd_model

To keep those imports working, this module now simply re-exports
the symbols from the new, more organized location:

    src.recommender.models.svd_model

This pattern (a thin facade that forwards imports) is common in
large codebases when you refactor package structure but want to
avoid breaking existing code or tests.
"""

from src.recommender.models.svd_model import RatingsDataset

__all__ = ["RatingsDataset"]
