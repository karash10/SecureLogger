import torch
import torch.nn as nn
import numpy as np

# --- 1. CRITICAL: SHARED CONSTANTS ---
# These values MUST be identical to the ones used in your train_gan.py script
MAX_LEN = 60
NOISE_DIM = 100
LSTM_UNITS = 256

# --- 2. Load the Character Map ---
# This is needed to decode the generator's output back to text
try:
    char_map = np.load('char_map.npy', allow_pickle=True).item()
except FileNotFoundError:
    print("FATAL ERROR: 'char_map.npy' not found. Please run train_gan.py first.")
    exit()

VOCAB_SIZE = len(char_map) + 1
int_to_char = {i: c for c, i in char_map.items()}

# --- 3. Redefine the Generator Class (this must match the training script) ---
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(NOISE_DIM, LSTM_UNITS),
            nn.LeakyReLU(0.2),
            # This is the corrected line to match your trained model
            nn.Linear(LSTM_UNITS, MAX_LEN * VOCAB_SIZE),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.net(x)
        # This reshape will now work
        return x.view(-1, MAX_LEN, VOCAB_SIZE)

# --- 4. Load the Trained Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)

try:
    generator.load_state_dict(torch.load('url_generator_pytorch.pth', map_location=device))
except FileNotFoundError:
    print("FATAL ERROR: 'url_generator_pytorch.pth' not found. Please run train_gan.py first.")
    exit()

generator.eval() # Set the model to evaluation mode

# --- 5. Generate and Print Samples ---
NUM_SAMPLES = 10
print(f"[*] Generating {NUM_SAMPLES} sample URLs from the trained GAN...\n")

# Create random noise on the correct device
noise = torch.randn(NUM_SAMPLES, NOISE_DIM).to(device)

# Generate the output and get the most likely character indices
generated_probs = generator(noise)
generated_indices = torch.argmax(generated_probs, dim=2).cpu().numpy()

# Decode and print each generated URL
for i, seq in enumerate(generated_indices):
    url = "".join([int_to_char.get(char_index, '') for char_index in seq if char_index != 0])
    print(f"Sample {i+1}: {url.strip()}")