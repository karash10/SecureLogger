# SecureLogger

<<<<<<< HEAD
SecureLogger is a Python-based cyber deception tool that uses a Generative Adversarial Network (GAN) to flood web server access logs with realistic, plausible, but entirely fake log entries. Its purpose is to camouflage real user activity and make it significantly harder for an attacker to analyze log files for reconnaissance.
=======
SecureLogger is a Python-based cyber deception tool that uses a Generative Adversarial Network (GAN) to flood web server access logs with realistic, plausible, but entirely fake log entries. Its purpose is to obfuscate real user activity, helping defenders mask true traffic patterns and frustrate attackers or unauthorized auditors.
>>>>>>> b96c880 (updated readme file)

## Features

- **Customizable Training Data:** Works with any simple text file of URL paths, allowing you to train the GAN to mimic any style of web traffic.
- **GAN-Powered Noise:** Utilizes a PyTorch-based GAN (with LSTM layers) to learn and generate highly convincing, novel URL paths.
- **Dynamic Log Flooding:** Configure to inject a fixed number of fake logs or maintain a dynamic noise-to-signal ratio.
- **Manual & Real-Time Modes:** Includes scripts for both on-demand (manual) log injection and continuous, real-time monitoring and injection.
- **Easy integration:** Simple scripts for rapid deployment and testing.
- **Data privacy:** Redacts sensitive information, helping maintain privacy in logs.
- **Thread-safe and high-performance logging:** Designed for reliability in operational environments.

## Architecture Diagram

The SecureLogger pipeline is illustrated below:

```
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
```

**Pipeline Steps:**
1. **Prepare Data:** You provide a dataset of sample URL paths that you want the AI to learn from.
2. **Train:** The `train_gan.py` script reads your dataset, converts the text to numerical tensors, and trains a GAN to learn the URL patterns. The trained Generator model is saved.
3. **Inject:** The `manual_injector.py` or `log_watcher.py` script loads the trained Generator, produces a flood of fake URL paths, and injects them as new entries into a target access log file.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/karash10/SecureLogger.git
cd SecureLogger
```

### 2. Set up the environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # For GPU powered machines
pip install numpy watchdog
```

## Usage

### Start the website

In one terminal, run the dummy web app:
```bash
python TesterWebsite.py
```

### Start the watcher

In a separate terminal, run the log watcher script (runs continuously):
```bash
python Flooder.py
```

<<<<<<< HEAD
Test: Visit a page on your website. The watcher will automatically detect the new log and inject a flood of fake ones.
=======
### Test

Visit a page on your website. The watcher will automatically detect the new log and inject a flood of fake ones.

## Configuration

You can configure SecureLogger using environment variables or configuration files. Common options include:

- `LOG_LEVEL`: Set the logging verbosity
- `LOG_FILE`: Specify the log file path
- `ENCRYPTION_KEY`: Set encryption key for log storage

## Security

SecureLogger is built with security in mind:
- All logs can be optionally encrypted at rest.
- Sensitive data can be automatically redacted or hashed.
- Access to logs can be restricted via permissions.

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, improvements, or new features.

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a pull request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please open an issue or contact [karash10](https://github.com/karash10).
>>>>>>> b96c880 (updated readme file)
