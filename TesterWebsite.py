from flask import Flask, request, abort
import logging
from logging.handlers import RotatingFileHandler

# --- Configuration ---
LOG_FILE = "dummy_access.log"
HOST = '127.0.0.1'  # Runs on your local machine
PORT = 5000

# --- Setup a professional-grade logger ---
# This setup is more robust for a "live" application
access_logger = logging.getLogger('access_log')
access_logger.setLevel(logging.INFO)

# Use a rotating file handler to prevent the log file from getting too big
# maxBytes=1000000 means the log will be rolled over after 1MB
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1000000, backupCount=5)
formatter = logging.Formatter('%(asctime)s - IP: %(remote_addr)s - Path: %(path)s')
file_handler.setFormatter(formatter)
access_logger.addHandler(file_handler)

app = Flask(__name__)

# --- Real Website Pages ---
@app.route('/')
def home():
    log_data = {'remote_addr': request.remote_addr, 'path': '/'}
    access_logger.info("Real visit to Homepage", extra=log_data)
    return "<h1>Welcome to Dummy.com!</h1>"

@app.route('/about')
def about():
    log_data = {'remote_addr': request.remote_addr, 'path': '/about'}
    access_logger.info("Real visit to About Page", extra=log_data)
    return "<h1>About Us</h1>"

@app.route('/contact')
def contact():
    log_data = {'remote_addr': request.remote_addr, 'path': '/contact'}
    access_logger.info("Real visit to Contact Page", extra=log_data)
    return "<h1>Contact</h1>"

# --- This route handles all other URLs (fakes and 404s) ---
@app.route('/<path:path>')
def catch_all(path):
    log_data = {'remote_addr': request.remote_addr, 'path': f'/{path}'}
    access_logger.info("Attempt to non-existent page", extra=log_data)
    abort(404)

@app.errorhandler(404)
def page_not_found(e):
    return "<h2>404 Not Found</h2>", 404

if __name__ == '__main__':
    print(f"[*] Starting Dummy.com on http://{HOST}:{PORT}")
    print(f"[*] Logging all access to {LOG_FILE}")
    app.run(host=HOST, port=PORT)