import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import os
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import for TPU environment (Colab)
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False

class WordFishDataset(Dataset):
    """Custom PyTorch Dataset for sparse user-item interactions."""
    def __init__(self, user_indices, item_indices, counts):
        self.user_indices = user_indices
        self.item_indices = item_indices
        self.counts = counts

    def __len__(self):
        return len(self.user_indices)

    def __getitem__(self, idx):
        return self.user_indices[idx], self.item_indices[idx], self.counts[idx]

class SparseWordFish(nn.Module):
    """
    A memory-efficient and scalable implementation of WordFish for large, sparse datasets.
    This version uses mini-batch gradient descent and negative sampling.
    """
    def __init__(self, data, user_col='user_id', item_col='item_id'):
        """
        Initializes the model from a DataFrame of user-item interactions.

        Args:
            data (pd.DataFrame): DataFrame with at least two columns for users and items.
            user_col (str): The name of the user ID column.
            item_col (str): The name of the item ID column.
        """
        super(SparseWordFish, self).__init__()
        
        # --- 1. Preprocess and Map IDs ---
        # Aggregate duplicate user-item pairs into counts
        counts_df = data.groupby([user_col, item_col]).size().reset_index(name='counts')

        # Create contiguous integer mappings for users and items
        self.user_map = {uid: i for i, uid in enumerate(counts_df[user_col].unique())}
        self.item_map = {iid: i for i, iid in enumerate(counts_df[item_col].unique())}
        self.item_inv_map = {i: iid for iid, i in self.item_map.items()} # For plotting
        
        self.N = len(self.user_map) # Number of unique users
        self.V = len(self.item_map) # Number of unique items

        # --- 2. Store Data as Tensors ---
        user_indices = torch.LongTensor([self.user_map[uid] for uid in counts_df[user_col]])
        item_indices = torch.LongTensor([self.item_map[iid] for iid in counts_df[item_col]])
        counts = torch.FloatTensor(counts_df['counts'].values)
        
        self.dataset = WordFishDataset(user_indices, item_indices, counts)
        
        # For negative sampling, we need item frequencies
        item_counts = counts_df.groupby(item_col).size()
        item_freqs = torch.zeros(self.V)
        for item, count in item_counts.items():
            item_freqs[self.item_map[item]] = count
        self.item_freqs_dist = torch.pow(item_freqs, 0.75) # Standard practice to smooth distribution

        print(f"Initialized model with {self.N} users and {self.V} items.")
        
        # --- 3. Initialize Model Parameters ---
        # These are now 1D tensors for efficient indexing
        self.alpha = nn.Parameter(torch.zeros(self.N))   # User fixed effects
        self.psi = nn.Parameter(torch.zeros(self.V))     # Item fixed effects
        self.beta = nn.Parameter(torch.randn(self.V))    # Item weights (latent trait)
        self.theta = nn.Parameter(torch.randn(self.N))   # User positions (latent trait)

    def train_model(self, num_epochs=10, batch_size=1024, lr=0.05, num_neg_samples=5):
        """
        Trains the model using mini-batching and negative sampling.

        Args:
            num_epochs (int): Number of training epochs.
            batch_size (int): Number of positive samples per batch.
            lr (float): Learning rate for the Adam optimizer.
            num_neg_samples (int): Number of negative samples per positive sample.
        """
        # --- 1. Setup Device (TPU, GPU, or CPU) ---
        if XLA_AVAILABLE:
            device = xm.xla_device()
            print("Using TPU device.")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using {device} device.")
        
        self.to(device) # Move model parameters to the device
        
        # --- 2. Setup DataLoader and Optimizer ---
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        if XLA_AVAILABLE:
            # Wrap loader for TPU
            para_loader = pl.ParallelLoader(dataloader, [device])
            train_loader = para_loader.per_device_loader(device)
        else:
            train_loader = dataloader

        loss_record = []
        for epoch in range(num_epochs):
            self.train()
            total_loss = 0.0
            pb = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for user_idx, item_idx, counts in pb:
                optimizer.zero_grad()

                # --- 3. Loss for Positive Samples ---
                # Gather parameters for the batch
                alpha_b = self.alpha[user_idx]
                psi_b = self.psi[item_idx]
                theta_b = self.theta[user_idx]
                beta_b = self.beta[item_idx]

                # Compute expected count (lambda)
                eta_pos = alpha_b + psi_b + theta_b * beta_b
                lambda_pos = torch.exp(eta_pos)
                
                # Poisson NLL Loss: lambda - y * log(lambda)
                # We sum this part of the loss
                loss_pos = (lambda_pos - counts * torch.log(lambda_pos + 1e-8)).sum()

                # --- 4. Loss for Negative Samples ---
                # Sample negative items based on frequency distribution
                neg_item_idx = torch.multinomial(
                    self.item_freqs_dist,
                    num_samples=user_idx.size(0) * num_neg_samples,
                    replacement=True
                ).to(device)
                
                # Reshape to match batch structure
                neg_item_idx = neg_item_idx.view(user_idx.size(0), num_neg_samples)
                
                # Gather parameters for negative samples
                # User parameters need to be expanded to match the number of negative samples
                alpha_neg = self.alpha[user_idx].unsqueeze(1).expand(-1, num_neg_samples)
                theta_neg = self.theta[user_idx].unsqueeze(1).expand(-1, num_neg_samples)
                # Item parameters are gathered directly
                psi_neg = self.psi[neg_item_idx]
                beta_neg = self.beta[neg_item_idx]

                # Compute expected count (lambda) for negative samples
                # Here, the count 'y' is 0, so loss is just lambda
                eta_neg = alpha_neg + psi_neg + theta_neg * beta_neg
                lambda_neg = torch.exp(eta_neg)
                loss_neg = lambda_neg.sum()

                # --- 5. Total Loss and Optimization Step ---
                # Average the loss over the batch
                loss = (loss_pos + loss_neg) / user_idx.size(0)
                loss.backward()

                if XLA_AVAILABLE:
                    # Special optimizer step for TPU
                    xm.optimizer_step(optimizer)
                else:
                    optimizer.step()
                
                total_loss += loss.item()
                pb.set_description(f"loss: {loss.item():.4f}")

            avg_loss = total_loss / len(train_loader)
            loss_record.append(avg_loss)
            if XLA_AVAILABLE:
                # Use master_print on TPU to avoid printing from all cores
                xm.master_print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
            else:
                print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # --- 6. Store Final Parameters ---
        # Move parameters back to CPU and convert to numpy for analysis
        self.beta_hat = self.beta.detach().cpu().numpy()
        self.psi_hat = self.psi.detach().cpu().numpy()
        self.alpha_hat = self.alpha.detach().cpu().numpy()
        self.theta_hat = self.theta.detach().cpu().numpy()

        return loss_record
        
    def get_top_n_arg(self, arr, n):
        return np.argpartition(arr, -n)[-n:]

    def stdz(self, x):
        return (x - x.mean()) / x.std()
    
    def get_item_latent_traits(self, standarized=True):
        """Returns a DataFrame of items and their estimated beta and psi values."""
        beta_vals = self.stdz(self.beta_hat) if standarized else self.beta_hat
        psi_vals = self.stdz(self.psi_hat) if standarized else self.psi_hat
        
        df = pd.DataFrame({
            'item_id': [self.item_inv_map[i] for i in range(self.V)],
            'beta_hat': beta_vals,
            'psi_hat': psi_vals
        })
        return df.sort_values(by='beta_hat', ascending=False)

    # The plotting function remains largely the same but uses the stored item map
    def wordplot(self, standarized=True, highlighted=10, alpha=0.05):
        """
        Generate plots to visualize word importances and frequencies.
        'word' here corresponds to 'item'.
        """
        beta = self.beta_hat
        psi = self.psi_hat

        # Calculate item frequencies from the stored dataset
        full_item_indices = self.dataset.item_indices
        freq = torch.bincount(full_item_indices, minlength=self.V).cpu().numpy()

        if isinstance(highlighted, int):
            top_word_id = self.get_top_n_arg(beta, highlighted)
            bottom_word_id = self.get_top_n_arg(-beta, highlighted)
            top_word = [self.item_inv_map[id] for id in top_word_id]
            bottom_word = [self.item_inv_map[id] for id in bottom_word_id]
        elif isinstance(highlighted, list):
            # Convert original IDs to internal indices
            top_word_id = [self.item_map[w] for w in highlighted if w in self.item_map]
            top_word = highlighted
            bottom_word_id, bottom_word = [], []
        else:
            raise ValueError("`highlighted` must be an int or a list of item IDs.")

        if standarized:
            beta = self.stdz(beta)
            psi = self.stdz(psi)

        plt.figure(figsize=(12, 6))
        
        # Plot 1: Beta vs. Log Frequency
        plt.subplot(1, 2, 1)
        plt.scatter(beta, np.log1p(freq), marker="+", color="brown", alpha=alpha)
        plt.xlabel('Estimated Item Latent Trait ($\hat{\\beta}$)')
        plt.ylabel("Item Frequency (log scale)")
        plt.title("Item Trait vs. Frequency")
        for i, word_id in enumerate(top_word_id):
            plt.annotate(top_word[i], (beta[word_id], np.log1p(freq[word_id])))
        for i, word_id in enumerate(bottom_word_id):
            plt.annotate(bottom_word[i], (beta[word_id], np.log1p(freq[word_id])))

        # Plot 2: Beta vs. Psi
        plt.subplot(1, 2, 2)
        plt.scatter(beta, psi, marker="+", color="royalblue", alpha=alpha)
        plt.xlabel('Estimated Item Latent Trait ($\hat{\\beta}$)')
        plt.ylabel('Estimated Item Fixed Effect ($\hat{\\psi}$)')
        plt.title("Item Trait vs. Fixed Effect")
        for i, word_id in enumerate(top_word_id):
            plt.annotate(top_word[i], (beta[word_id], psi[word_id]))
        for i, word_id in enumerate(bottom_word_id):
            plt.annotate(bottom_word[i], (beta[word_id], psi[word_id]))
        
        plt.tight_layout()
        plt.show()
