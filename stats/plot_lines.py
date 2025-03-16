import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.patches as mpatches

# Read the CSV file and plot metrics over steps with smoothing
def plot_metrics_over_steps(csv_file, smoothing_window=15, poly_order=3):
    # Load the data
    df = pd.read_csv(csv_file)
    
    # Add step column (row index)
    df['step'] = np.arange(len(df))
    
    # Calculate combined accuracy+confidence metric
    df['combined_accuracy'] = df['cur_model_accuracy'] /100 + df['avg_confidence']
    
    # Map model codes to numerical indices
    model_map = {'n': 0, 's': 1, 'm': 2, 'l': 3, 'x': 4}
    df['model_index'] = df['cur_model_index'].map(lambda x: model_map.get(x, -1))
    
    # Apply smoothing function to the metrics
    # Note: smoothing_window must be odd and less than data length
    smoothing_window = min(smoothing_window, len(df) - 1)
    if smoothing_window % 2 == 0:
        smoothing_window -= 1  # Make it odd
    
    if len(df) > smoothing_window:
        queue_smooth = savgol_filter(df['queue_length'], smoothing_window, poly_order)
        accuracy_smooth = savgol_filter(df['combined_accuracy'], smoothing_window, poly_order)
        latency_smooth = savgol_filter(df['total_latency'], smoothing_window, poly_order)
        fps_smooth = savgol_filter(df['fps'], smoothing_window, poly_order)
    else:
        # If not enough data points for smoothing, use original data
        queue_smooth = df['queue_length']
        accuracy_smooth = df['combined_accuracy']
        latency_smooth = df['total_latency']
        fps_smooth = df['fps']
    
    # Create a figure with 5 subplots sharing x-axis (added model type subplot)
    fig, axs = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    fig.suptitle('Model Performance Metrics Over Time', fontsize=16)
    
    # Plot queue length
    queue_mean = df['queue_length'].mean()
    axs[0].plot(df['step'], df['queue_length'], 'b-', alpha=0.3, linewidth=1, label='Raw')
    axs[0].plot(df['step'], queue_smooth, 'b-', linewidth=2, label='Smoothed')
    axs[0].axhline(y=queue_mean, color='b', linestyle='--', alpha=0.7, label=f'Mean: {queue_mean:.2f}')
    axs[0].set_ylabel('Queue Length')
    axs[0].grid(True)
    axs[0].legend(loc='best')
    
    # Plot combined accuracy
    accuracy_mean = df['combined_accuracy'].mean()
    axs[1].plot(df['step'], df['combined_accuracy'], 'g-', alpha=0.3, linewidth=1, label='Raw')
    axs[1].plot(df['step'], accuracy_smooth, 'g-', linewidth=2, label='Smoothed')
    axs[1].axhline(y=accuracy_mean, color='g', linestyle='--', alpha=0.7, label=f'Mean: {accuracy_mean:.2f}')
    axs[1].set_ylabel('Accuracy + Confidence')
    axs[1].grid(True)
    axs[1].legend(loc='best')
    
    # Plot total latency
    latency_mean = df['total_latency'].mean()
    axs[2].plot(df['step'], df['total_latency'], 'r-', alpha=0.3, linewidth=1, label='Raw')
    axs[2].plot(df['step'], latency_smooth, 'r-', linewidth=2, label='Smoothed')
    axs[2].axhline(y=latency_mean, color='r', linestyle='--', alpha=0.7, label=f'Mean: {latency_mean:.4f}')
    axs[2].set_ylabel('Total Latency (s)')
    axs[2].grid(True)
    axs[2].legend(loc='best')
    
    # Plot FPS
    fps_mean = df['fps'].mean()
    axs[3].plot(df['step'], df['fps'], 'purple', alpha=0.3, linewidth=1, label='Raw')
    axs[3].plot(df['step'], fps_smooth, 'purple', linewidth=2, label='Smoothed')
    axs[3].axhline(y=fps_mean, color='purple', linestyle='--', alpha=0.7, label=f'Mean: {fps_mean:.2f}')
    axs[3].set_ylabel('FPS')
    axs[3].grid(True)
    axs[3].legend(loc='best')
    
    # Plot model switches (new dedicated subplot) - removed the black line
    model_colors = {0: 'blue', 1: 'green', 2: 'orange', 3: 'red', 4: 'purple'}
    axs[4].set_ylabel('Model Type')
    axs[4].set_yticks(range(5))
    axs[4].set_yticklabels(['n', 's', 'm', 'l', 'x'])
    axs[4].set_ylim(-0.5, 4.5)
    axs[4].grid(True)
    axs[4].set_xlabel('Steps')
    
    # Color the background to highlight different model regions
    prev_model = None
    start_idx = 0
    for idx, model in enumerate(df['model_index']):
        if model != prev_model:
            if prev_model is not None:
                color = model_colors.get(prev_model, 'gray')
                axs[4].axvspan(start_idx, idx-1, alpha=0.2, color=color)
            start_idx = idx
            prev_model = model
    
    # Color the last region
    if prev_model is not None:
        color = model_colors.get(prev_model, 'gray')
        axs[4].axvspan(start_idx, len(df)-1, alpha=0.2, color=color)
    
    # Add legend for model types
    model_patches = []
    for model_idx, color in model_colors.items():
        if model_idx in df['model_index'].values:
            model_name = list(model_map.keys())[list(model_map.values()).index(model_idx)]
            patch = mpatches.Patch(color=color, alpha=0.2, label=f'Model {model_name}')
            model_patches.append(patch)
    
    axs[4].legend(handles=model_patches, loc='best')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the figure
    plt.savefig('model_metrics_over_steps.png', dpi=300)
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Average Queue Length: {queue_mean:.2f}")
    print(f"Average Accuracy + Confidence: {accuracy_mean:.2f}")
    print(f"Average Total Latency: {latency_mean:.4f} seconds")
    print(f"Average FPS: {fps_mean:.2f}")
    
    # Print model usage statistics
    model_counts = df['cur_model_index'].value_counts()
    print("\nModel Usage Statistics:")
    for model, count in model_counts.items():
        percentage = (count / len(df)) * 100
        print(f"Model {model}: {count} samples ({percentage:.2f}% of time)")

# Usage example
if __name__ == "__main__":
    plot_metrics_over_steps('1_10_ac_stats.csv', smoothing_window=15, poly_order=3)  # Adjust smoothing parameters as needed