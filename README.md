# Real-Time Anomaly Detection System

## Problem Statement

This project focuses on detecting anomalies in real-time from a streaming data source. The system simulates a data stream based on a noisy sine wave, with random anomalies injected. Multiple anomaly detection techniques such as Z-score, threshold, moving average, interquartile range (IQR), and rolling Z-score are employed. The system uses an ensemble method to combine these techniques, ensuring robust anomaly detection.

The goal is to identify unexpected events or outliers in data that could indicate potential issues in real-world systems like financial transactions, sensor networks, or industrial monitoring.

## Features
- Simulates real-time data streaming with noise and anomalies.
- Utilizes multiple anomaly detection methods.
- Combines methods using majority voting (ensemble detection).
- Provides real-time plotting of the data stream and detected anomalies.

## Installation

To set up and run the project locally, follow these steps:

### Prerequisites

- Python 3.6+
- Required libraries:
  - `numpy`
  - `matplotlib`
  - `argparse`

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/abhisirka2001/anomaly-detection.git
   cd anomaly-detection
   pip install -r requirements.txt


2. **Run the python file:**
   ```bash
   python main.py

3. **Customize with your arguments**
   ```bash
   python main.py --window_size 100 --step_size 10 --delay 0.2


# The system incorporates robust error handling, using try-except blocks to manage exceptions during data generation and anomaly detection, ensuring graceful failures and clear error messages without crashing.




