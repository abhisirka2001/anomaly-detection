import numpy as np
import matplotlib.pyplot as plt
import argparse

# Function to simulate a real-time data stream with noise and injected anomalies
def generate_real_time_data_stream(n_samples=1000, noise=0.003, anomaly_prob=0.05):
    """
    Generates a synthetic real-time data stream consisting of a noisy sine wave
    with randomly injected anomalies.

    Parameters:
    -----------
    n_samples : int, optional, default=1000
        The total number of data points to generate.
    
    noise : float, optional, default=0.05
        The standard deviation of the Gaussian noise added to the sine wave.
    
    anomaly_prob : float, optional, default=0.05
        The probability of injecting an anomaly at any given point.

    Returns:
    --------
    X : np.ndarray
        The generated data stream with anomalies.
    
    anomalies : np.ndarray
        A boolean array indicating which points in X are anomalies.
    """
    try:
        # Generate a sine wave with added Gaussian noise
        X = np.sin(np.linspace(0, 10 * np.pi, n_samples)) + noise * np.random.randn(n_samples)
        
        # Inject anomalies at random locations with a specified probability
        anomalies = np.random.choice([True, False], size=n_samples, p=[anomaly_prob, 1 - anomaly_prob])
        X[anomalies] += np.random.uniform(10, 20, size=sum(anomalies))  # Inject large anomalies
        
        return X, anomalies
    except Exception as e:
        print(f"Error generating data stream: {e}")
        return np.array([]), np.array([])

# Z-score based anomaly detection
def z_score_anomaly_detection(data, threshold=2):
    """
    Detects anomalies using Z-score. Anomalies are defined as points where
    the Z-score exceeds a specified threshold.

    Parameters:
    -----------
    data : np.ndarray
        The input data stream for anomaly detection.
    
    threshold : float, optional, default=2
        The Z-score threshold above which points are considered anomalies.

    Returns:
    --------
    np.ndarray
        A boolean array indicating the detected anomalies.
    """
    try:
        mean = np.mean(data)
        std_dev = np.std(data)
        z_scores = np.abs((data - mean) / std_dev)
        return z_scores > threshold
    except Exception as e:
        print(f"Error in Z-score anomaly detection: {e}")
        return np.array([])

# Threshold-based anomaly detection
def threshold_anomaly_detection(data, threshold):
    """
    Detects anomalies based on an absolute threshold.

    Parameters:
    -----------
    data : np.ndarray
        The input data stream for anomaly detection.
    
    threshold : float
        The threshold above which points are considered anomalies.

    Returns:
    --------
    np.ndarray
        A boolean array indicating the detected anomalies.
    """
    try:
        return np.abs(data) > threshold
    except Exception as e:
        print(f"Error in threshold anomaly detection: {e}")
        return np.array([])

# Moving Average anomaly detection
def moving_average_anomaly_detection(data, window_size=5, threshold=2):
    """
    Detects anomalies based on the moving average of the data. Points where
    the residuals (difference from moving average) exceed a certain threshold
    are flagged as anomalies.

    Parameters:
    -----------
    data : np.ndarray
        The input data stream for anomaly detection.
    
    window_size : int, optional, default=5
        The size of the moving average window.
    
    threshold : float, optional, default=2
        The threshold for detecting anomalies based on residuals from the moving average.

    Returns:
    --------
    np.ndarray
        A boolean array indicating the detected anomalies.
    """
    try:
        moving_avg = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
        residuals = np.abs(data[window_size - 1:] - moving_avg)
        anomalies = residuals > threshold * np.std(data)
        return np.pad(anomalies, (window_size - 1, 0), 'constant', constant_values=(False))
    except Exception as e:
        print(f"Error in Moving Average anomaly detection: {e}")
        return np.array([])

# Interquartile Range (IQR) anomaly detection
def iqr_anomaly_detection(data):
    """
    Detects anomalies based on the Interquartile Range (IQR). Points outside
    1.5 * IQR from the first and third quartiles are considered anomalies.

    Parameters:
    -----------
    data : np.ndarray
        The input data stream for anomaly detection.

    Returns:
    --------
    np.ndarray
        A boolean array indicating the detected anomalies.
    """
    try:
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return (data < lower_bound) | (data > upper_bound)
    except Exception as e:
        print(f"Error in IQR anomaly detection: {e}")
        return np.array([])

# Rolling Z-score anomaly detection
def rolling_z_score_anomaly_detection(data, window_size=5, threshold=2):
    """
    Detects anomalies using a rolling Z-score calculation. Points are considered
    anomalies if their Z-score within a rolling window exceeds the specified threshold.

    Parameters:
    -----------
    data : np.ndarray
        The input data stream for anomaly detection.
    
    window_size : int, optional, default=5
        The size of the rolling window for Z-score calculation.
    
    threshold : float, optional, default=2
        The Z-score threshold for detecting anomalies.

    Returns:
    --------
    np.ndarray
        A boolean array indicating the detected anomalies.
    """
    try:
        z_scores = np.zeros(len(data))
        for i in range(window_size, len(data)):
            window = data[i-window_size:i]
            mean = np.mean(window)
            std_dev = np.std(window)
            z_scores[i] = np.abs((data[i] - mean) / std_dev)
        return z_scores > threshold
    except Exception as e:
        print(f"Error in Rolling Z-score anomaly detection: {e}")
        return np.array([])

# Ensemble anomaly detection using majority voting
def ensemble_anomaly_detection(data):
    """
    Detects anomalies using an ensemble of methods including Z-score, threshold,
    moving average, IQR, and rolling Z-score. A point is flagged as an anomaly if
    more than half of the methods agree.

    Parameters:
    -----------
    data : np.ndarray
        The input data stream for anomaly detection.

    Returns:
    --------
    np.ndarray
        A boolean array indicating the detected anomalies.
    """
    try:
        z_score_anomalies = z_score_anomaly_detection(data)
        threshold_anomalies = threshold_anomaly_detection(data, threshold=5)
        moving_avg_anomalies = moving_average_anomaly_detection(data)
        iqr_anomalies = iqr_anomaly_detection(data)
        rolling_z_anomalies = rolling_z_score_anomaly_detection(data)

        # Majority voting: anomaly if more than half of the methods detect an anomaly
        anomaly_votes = (z_score_anomalies.astype(int) + threshold_anomalies.astype(int) +
                         moving_avg_anomalies.astype(int) + iqr_anomalies.astype(int) +
                         rolling_z_anomalies.astype(int))

        ensemble_anomalies = anomaly_votes >= 3  # Majority vote
        
        return ensemble_anomalies
    except Exception as e:
        print(f"Error in ensemble anomaly detection: {e}")
        return np.array([])

# Real-time detection with sliding window
def real_time_anomaly_detection_system(window_size=50, step_size=5, delay=0.1):
    """
    Runs a real-time anomaly detection system using a sliding window approach.
    The system detects anomalies in a data stream using the ensemble anomaly detection method.

    Parameters:
    -----------
    window_size : int, optional, default=50
        The size of the sliding window for detection.
    
    step_size : int, optional, default=5
        The step size for sliding the window forward after each detection.
    
    delay : float, optional, default=0.1
        The time delay (in seconds) between each detection, simulating real-time processing.

    Returns:
    --------
    None
    """
    try:
        data_stream, true_anomalies = generate_real_time_data_stream()
        
        plt.ion()  # Interactive mode on for real-time plotting
        fig, ax = plt.subplots()
        
        line1, = ax.plot([], [], label="Data", color="blue")
        line2, = ax.plot([], [], 'r', label="Detected Anomalies", marker='x', linestyle="None")

        plt.legend()
        ax.set_ylim(np.min(data_stream), np.max(data_stream))

        # Lists to hold all data for plotting
        all_x_data = []
        all_y_data = []
        all_anomaly_indices = []
        all_anomalies = []

        # Sliding window anomaly detection
        for i in range(0, len(data_stream) - window_size, step_size):
            current_data = data_stream[i:i + window_size]

            # Detect anomalies in the current window
            anomalies = ensemble_anomaly_detection(current_data)

            # Update plot data
            all_x_data.extend(range(i, i + window_size))
            all_y_data.extend(current_data)

            # Append indices of detected anomalies for plotting
            anomaly_indices = np.where(anomalies)[0] + i
            all_anomaly_indices.extend(anomaly_indices)
            all_anomalies.extend(data_stream[anomaly_indices])

            # Update plot in real-time
            line1.set_data(all_x_data, all_y_data)
            line2.set_data(all_anomaly_indices, all_anomalies)
            ax.set_xlim(0, len(all_x_data))
            fig.canvas.draw()
            plt.pause(delay)
        
        plt.show()
    except Exception as e:
        print(f"Error in real-time anomaly detection system: {e}")

# Main function to parse arguments and run the real-time anomaly detection system
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run a real-time anomaly detection system.")
    parser.add_argument("--window_size", type=int, default=50, help="Size of the sliding window (default: 50)")
    parser.add_argument("--step_size", type=int, default=5, help="Step size for the sliding window (default: 5)")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between each detection step (default: 0.1 seconds)")

    args = parser.parse_args()

    # Run the real-time anomaly detection system with parsed arguments
    real_time_anomaly_detection_system(window_size=args.window_size, step_size=args.step_size, delay=args.delay)

