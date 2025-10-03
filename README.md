# SecureLogger

SecureLogger is a Python-based cyber deception tool that uses a Generative Adversarial Network (GAN) to flood web server access logs with realistic, plausible, but entirely fake log entries. Its purpose is to camouflage real user activity and make it significantly harder for an attacker to analyze log files for reconnaissance.

Features
Customizable Training Data: Works with any simple text file of URL paths, allowing you to train the GAN to mimic any style of web traffic.
GAN-Powered Noise: Utilizes a PyTorch-based GAN (with LSTM layers) to learn and generate highly convincing, novel URL paths.
Dynamic Log Flooding: Can be configured to inject a fixed number of fake logs or to maintain a dynamic noise-to-signal ratio.
Manual & Real-Time Modes: Includes scripts for both on-demand (manual) log injection and continuous, real-time monitoring and injection.

How It Works
The project follows a simple but powerful pipeline:

Prepare Data: You provide a dataset of sample URL paths that you want the AI to learn from.

Train: The train_gan.py script reads your dataset, converts the text to numerical tensors, and trains a GAN to learn the URL patterns. The trained Generator model is saved.

Inject: The manual_injector.py or log_watcher.py script loads the trained Generator, produces a flood of fake URL paths, and injects them as new entries into a target access log file.

+------------------+     +-----------------------+     +-----------------------+
|   Your Dataset   | --> |      GAN Training     | --> |   Trained Generator   |
| (dataset.txt)    |     |   (train_gan.py)      |     | (.pth file)           |
+------------------+     +-----------------------+     +-----------------------+
                                                               |
                                                               v
                                                     +--------------------+
                                                     |    Log Injector    |
                                                     | (manual_injector...)
                                                     +--------------------+
                                                               |
                                                               v
                                                     +--------------------+
                                                     |  Camouflaged Log   |
                                                     +--------------------+

1. Setup the Environment

git clone https://github.com/karash10/SecureLogger.git
cd SecureLogger
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

2) Install Dependencies

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 (For GPU powered machines)
pip install numpy watchdog

Steps to run:

Start the website: In one terminal, run the dummy web app.(TesterWebsite)
python TesterWebsite.py

Start the watcher: In a separate terminal, run the log watcher script. It will run continuously.
python Flooder.py

Test: Visit a page on your website. The watcher will automatically detect the new log and inject a flood of fake ones.
