import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt


def generate_data():
    N_k = np.random.randint(10, 101)
    data = np.random.choice(range(1000), size=N_k, replace=False)
    return data

class SetNN(nn.Module):
    def __init__(self):
        super(SetNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=1000, embedding_dim=20)  
        self.self_attention1 = nn.MultiheadAttention(embed_dim=20, num_heads=4, batch_first=True)  
        self.self_attention2 = nn.MultiheadAttention(embed_dim=20, num_heads=4, batch_first=True) 
        self.fc1 = nn.Linear(20, 10) 
        self.fc2 = nn.Linear(10, 1) 

    def forward(self, x):
        x = self.embedding(x)
        attn_output, _ = self.self_attention1(x, x, x)
        attn_output, _ = self.self_attention2(attn_output, attn_output, attn_output)
        x = attn_output.mean(dim=1)
        x = F.relu(self.fc1(x))  
        x = self.fc2(x)
        return x

class SetDataset(Dataset):
    def __init__(self, size=10000):
        self.data = [generate_data() for _ in range(size)]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Find the maximum value in the set for the label
        x = self.data[idx]
        y = np.max(x)
        # Ensure x is padded to have a length of 100
        x_padded = F.pad(torch.tensor(x, dtype=torch.long), (0, 100 - len(x)), "constant", 0)
        return x_padded, torch.tensor(y, dtype=torch.float)
    
# Define the training loop
def train(model, dataloader, epochs=10):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=0.005)
    mse_loss = []

    # Move model to cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in tqdm(range(epochs), desc="Epochs"):
        total_loss = 0
        for x, y in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
            # Move data to cuda if available
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x).squeeze() 
            loss = F.mse_loss(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        mse_loss.append(total_loss / len(dataloader))
        tqdm.write(f"Epoch {epoch+1}: MSE Loss = {mse_loss[-1]:.4f}")
    return mse_loss

if __name__ == "__main__":

    # Generate a sample data
    sample_data = generate_data()

    # Instantiate the model
    model = SetNN()

    # Create the dataset and data loader
    dataset = SetDataset(size=1000)  
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train the model
    mse_loss = train(model, dataloader, epochs=1000) 

    # plot MSE loss
    plt.plot(mse_loss)
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('MSE Loss Across Epochs')
    plt.savefig('mse_loss_epochs.png')
