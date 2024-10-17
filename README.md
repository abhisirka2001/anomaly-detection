# Real-Time Anomaly Detection System

## Problem Statement

This project focuses on detecting anomalies in real-time from a streaming data source. The system simulates a data stream based on a noisy sine wave, with random anomalies injected. Multiple anomaly detection techniques such as Z-score, threshold, moving average, interquartile range (IQR), and rolling Z-score are employed. The system uses an ensemble method to combine these techniques, ensuring robust anomaly detection.

The goal is to identify unexpected events or outliers in data that could indicate potential issues in real-world systems like financial transactions, sensor networks, or industrial monitoring.

## Features
- Simulates real-time data streaming with noise and anomalies.
- Utilizes multiple anomaly detection methods.
- Combines methods using majority voting (ensemble detection).
- Provides real-time plotting of the data stream and detected anomalies.

## Methods 
### Overview of the Algorithm
#### Data Generation:

- A synthetic real-time data stream is created using a noisy sine wave. Anomalies are injected randomly with specified noise levels and magnitudes.

### Anomaly Detection Techniques:

- Z-Score Detection: Identifies anomalies based on how many standard deviations a data point is from the mean. Points with Z-scores exceeding a threshold are flagged as anomalies.
- Threshold-Based Detection: Flags points that exceed a predefined absolute threshold as anomalies.
Moving Average Detection: Calculates the moving average of the data. Anomalies are detected when residuals (the difference between the data points and the moving average) exceed a certain threshold.
- Interquartile Range (IQR) Detection: Uses the IQR to identify anomalies by flagging points that lie outside 1.5 times the IQR from the first and third quartiles.
- Rolling Z-Score Detection: Calculates Z-scores using a rolling window. Points with Z-scores exceeding a threshold in their respective windows are flagged as anomalies.
- Ensemble Detection: Combines the results from all the above methods using majority voting. A data point is flagged as an anomaly if more than half of the methods identify it as such.
Effectiveness
- Robustness: By employing multiple detection techniques, the algorithm increases its robustness to different types of anomalies that may manifest in the data stream.
- Reduced False Positives: The ensemble method mitigates the likelihood of false positives, as it requires agreement among several methods before flagging a point as anomalous.
- Flexibility: The algorithm allows for adjustments to parameters (like thresholds and window sizes), enabling fine-tuning based on the characteristics of the data being analyzed.
- Real-Time Applicability: Designed for a real-time data stream, the methods can be adapted for continuous data inputs, making it suitable for monitoring applications in various fields (e.g., finance, IoT, healthcare).

## Installation

To set up and run the project locally, follow these steps:

### Prerequisites

- Python 3.6+
- Required libraries:
  - `numpy`
  - `matplotlib`
  - `argparse`
- (Above are inbuilt libraries in python, you may not require to download it additionally)

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


### Error Handling 
- The system incorporates robust error handling, using try-except blocks to manage exceptions during data generation and anomaly detection, ensuring graceful failures and clear error messages without crashing.**




