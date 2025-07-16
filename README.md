# Scalable WordFish Implementation in PyTorch

This repository contains a memory-efficient and scalable Python implementation of the WordFish scaling model, designed for large, sparse datasets. It is built with PyTorch and is optimized for training on GPUs or Google's TPUs.

WordFish is a statistical model used to estimate latent one-dimensional traits (e.g., political positions) from word frequency data. This implementation adapts the model for general-purpose use cases, such as estimating latent attributes of items from user-item interaction data.

---

## 🚀 Key Features

* **Memory Efficient**: Processes data as a sparse list of interactions, avoiding the creation of a massive user-item matrix.
* **Scalable**: Uses mini-batch training with negative sampling to handle datasets with millions or billions of interactions.
* **Hardware Accelerated**: Natively supports training on NVIDIA GPUs (`cuda`) and Google TPUs (`torch_xla`) for significant speed-ups.
* **Easy to Use**: A simple, Scikit-learn-like API for initializing, training, and analyzing the model.
* **Visualization**: Includes built-in plotting functions to easily visualize the estimated item traits and fixed effects.

---

## 🛠️ Installation

1.  Clone the repository to your local machine:
    ```bash
    git clone https://github.com/CSS-Laboratory/Wordfish.git
    cd your-repo-name
    ```

2.  Install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage Example

Here is a quick example of how to use the `SparseWordFish` model.

```python
import pandas as pd
from sparse_wordfish import SparseWordFish # Assume the class is in this file

# 1. Create a sample DataFrame of user-item interactions
# In your case, this would be your dataset of 90M+ rows
data = {
    'user_id': ['user1', 'user1', 'user2', 'user2', 'user2', 'user3', 'user3', 'user4', 'user4'],
    'item_id': ['itemA', 'itemB', 'itemB', 'itemC', 'itemC', 'itemA', 'itemC', 'itemA', 'itemB']
}
df_interactions = pd.DataFrame(data)

# 2. Initialize the model with your data
# This step preprocesses the data and sets up the model structure.
wf_model = SparseWordFish(df_interactions, user_col='user_id', item_col='item_id')

# 3. Train the model
# The training will automatically use a GPU or TPU if available.
# Training on a CPU will be significantly slower.
loss_history = wf_model.train_model(
    num_epochs=10, 
    batch_size=4, # Adjust based on your hardware
    lr=0.02, 
    num_neg_samples=5
)

# 4. Get the estimated latent traits for items
# The results are returned as a pandas DataFrame.
item_traits_df = wf_model.get_item_latent_traits()

print("--- Estimated Item Latent Traits (beta) ---")
print(item_traits_df)
#    item_id  beta_hat   psi_hat
# 1    itemB  1.173812  0.707798
# 0    itemA  0.180901 -1.413197
# 2    itemC -1.354713  0.705399


# 5. Visualize the results
wf_model.wordplot(highlighted=3, standarized=True)
