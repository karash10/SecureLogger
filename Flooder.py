import time
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os

# --- Configuration ---
LOG_FILE_TO_WATCH = "dummy_access.log"
MODEL_FILE = 'GeneratorModel.pth'
CHAR_MAP_FILE = 'char_map.npy'
FLOOD_RATIO = 10
COOLDOWN_SECONDS = 5 # Cooldown period to prevent infinite loops

# --- 1. Define the CORRECT GAN Generator Class ---
# This blueprint MUST match the one from your successful train_gan.py
MAX_LEN = 60
NOISE_DIM = 100
LSTM_UNITS = 256

char_map = np.load(CHAR_MAP_FILE, allow_pickle=True).item()
VOCAB_SIZE = len(char_map) + 1

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(NOISE_DIM, LSTM_UNITS),
            nn.LeakyReLU(0.2),
            # This is the corrected line that ensures the output size is correct
            nn.Linear(LSTM_UNITS, MAX_LEN * VOCAB_SIZE),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.net(x)
        # This reshape will now work
        return x.view(-1, MAX_LEN, VOCAB_SIZE)

# --- 2. Load the Trained Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
int_to_char = {i: c for c, i in char_map.items()}
generator = Generator().to(device)
generator.load_state_dict(torch.load(MODEL_FILE, map_location=device))
generator.eval()
print("[*] GAN Generator model loaded and ready.")

# --- 3. The Log Injection Function ---
last_injection_time = 0 # Global variable to track the last injection time

def inject_fake_logs(num_to_generate):
    global last_injection_time
    # Update the timestamp at the start of injection
    last_injection_time = time.time()
    
    print(f"[*] Triggered: Generating {num_to_generate} fake logs...")
    noise = torch.randn(num_to_generate, NOISE_DIM).to(device)
    generated_probs = generator(noise)
    generated_indices = torch.argmax(generated_probs, dim=2).cpu().numpy()

    fake_urls = ["".join([int_to_char.get(i, '') for i in seq if i != 0]).strip() for seq in generated_indices]

    with open(LOG_FILE_TO_WATCH, 'a') as f:
        # f.write(f"\n--- REAL-TIME GAN INJECTION ({datetime.now()}) ---\n")
        for url in fake_urls:
            fake_ip = f"10.10.10.{np.random.randint(1, 255)}"
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            log_line = f"{timestamp} - IP: {fake_ip} - Path: {url}\n"
            f.write(log_line)
    print("[+] Injection complete.")

# --- 4. The Watchdog Event Handler (with cooldown) ---
class LogHandler(FileSystemEventHandler):
    def on_modified(self, event):
        # Check the cooldown period first to prevent loops
        if time.time() - last_injection_time < COOLDOWN_SECONDS:
            return

        if not event.is_directory and event.src_path.endswith(LOG_FILE_TO_WATCH):
            inject_fake_logs(FLOOD_RATIO)

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    path = "." 
    event_handler = LogHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    
    print(f"[*] Starting Log Watcher. Monitoring '{LOG_FILE_TO_WATCH}' for changes...")
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()