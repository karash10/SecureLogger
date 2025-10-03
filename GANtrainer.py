# Final corrected version for the SecureLogger project
# Fixes the RuntimeError by correcting the Generator architecture

import torch
import torch.nn as nn
import numpy as np
import os
import re

# --- Prerequisites ---
if not os.path.exists('combined_dataset.txt'):
    print("FATAL ERROR: 'combined_dataset.txt' not found. Please run the crawler first.")
    exit()

# --- 1. Constants ---
MAX_LEN = 60
NOISE_DIM = 100
LSTM_UNITS = 256
EMBEDDING_DIM = 64
EPOCHS = 5000
BATCH_SIZE = 64

# --- 2. Prepare the Dataset ---
print("[*] Preparing data for training...")
with open('combined_dataset.txt', 'r', encoding='utf-8') as f:
    paths = [line.strip() for line in f if line.strip()]

text = "".join(paths)
chars = sorted(list(set(text)))
char_to_int = {c: i + 1 for i, c in enumerate(chars)}
np.save('char_map.npy', char_to_int)

VOCAB_SIZE = len(char_to_int) + 1

# --- 3. Process data into padded tensors ---
sequences = []
for path in paths:
    encoded = [char_to_int.get(char, 0) for char in path]
    if len(encoded) < MAX_LEN:
        padded = np.pad(encoded, (0, MAX_LEN - len(encoded)), 'constant')
    else:
        padded = encoded[:MAX_LEN]
    sequences.append(padded)

data_tensor = torch.LongTensor(sequences)
train_dataset = torch.utils.data.DataLoader(data_tensor, batch_size=BATCH_SIZE, shuffle=True)
print(f"[*] Data processed. Vocabulary size: {VOCAB_SIZE - 1}.")

# --- 4. Define the GAN Models ---
# THIS IS THE CORRECTED GENERATOR CLASS
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(NOISE_DIM, LSTM_UNITS),
            nn.LeakyReLU(0.2),
            # The output size of this layer now correctly matches the final desired shape
            nn.Linear(LSTM_UNITS, MAX_LEN * VOCAB_SIZE),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.net(x)
        # This reshape will now work correctly
        return x.view(-1, MAX_LEN, VOCAB_SIZE)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.lstm = nn.LSTM(EMBEDDING_DIM, LSTM_UNITS, batch_first=True)
        self.fc = nn.Linear(LSTM_UNITS, 1)
    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden.squeeze(0))

# --- 5. Setup for Training ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Using device: {device}")

generator = Generator().to(device)
discriminator = Discriminator().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# --- 6. The Training Loop ---
print("\n[*] Starting GAN Training... This will take time.")
for epoch in range(EPOCHS):
    for i, sequences_batch in enumerate(train_dataset):
        real_sequences = sequences_batch.to(device)
        current_batch_size = real_sequences.size(0)
        
        # Train Discriminator
        optimizer_d.zero_grad()
        real_labels = torch.ones(current_batch_size, 1).to(device)
        output_real = discriminator(real_sequences)
        loss_d_real = criterion(output_real, real_labels)
        
        noise = torch.randn(current_batch_size, NOISE_DIM).to(device)
        fake_sequences_probs = generator(noise)
        fake_sequences = torch.argmax(fake_sequences_probs, dim=2)
        fake_labels = torch.zeros(current_batch_size, 1).to(device)
        output_fake = discriminator(fake_sequences.detach())
        loss_d_fake = criterion(output_fake, fake_labels)
        
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        optimizer_d.step()
        
        # Train Generator
        optimizer_g.zero_grad()
        output = discriminator(fake_sequences)
        loss_g = criterion(output, real_labels)
        loss_g.backward()
        optimizer_g.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Discriminator Loss: {loss_d.item():.4f}, Generator Loss: {loss_g.item():.4f}")

# --- 7. Save the Trained Model ---
torch.save(generator.state_dict(), 'GeneratorModel.pth')
print("\n[SUCCESS] Training finished. Generator model saved as 'GeneratorModel.pth'")