import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read the CSV file (assuming data is in the same format as before)
df = pd.read_csv('trace_data.csv')

# Create model mapping
model_map = {
    'yolov5n': 1,
    'yolov5s': 2,
    'yolov5m': 3,
    'yolov5l': 4,
    'yolov5x': 5
}

# Convert models to numerical values
model_values = [model_map[m] for m in df['CurrentModel']]

# Create the figure and primary axis
fig, ax1 = plt.subplots(figsize=(15, 7))

# Plot queue length on primary axis
color1 = '#1f77b4'  # Blue
ax1.set_xlabel('Step')
ax1.set_ylabel('Queue Length', color=color1)
line1 = ax1.plot(df['Step'], df['QueueLength'], color=color1, label='Queue Length')
ax1.tick_params(axis='y', labelcolor=color1)

# Create secondary axis for model types
ax2 = ax1.twinx()
color2 = '#ff7f0e'  # Orange
ax2.set_ylabel('Model Type', color=color2)

# Use drawstyle='steps-post' to create horizontal lines between steps
line2 = ax2.plot(df['Step'], model_values, color=color2, linestyle='-', 
                 drawstyle='steps-post', label='Model Type')
ax2.tick_params(axis='y', labelcolor=color2)

# Set the yticks for model types
ax2.set_yticks(list(model_map.values()))
ax2.set_yticklabels(list(model_map.keys()))

# Add grid
ax1.grid(True, alpha=0.3)

# Add legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')

# Add title
plt.title('Queue Length and Model Type vs Step')

# Adjust layout to prevent label clipping
plt.tight_layout()

plt.show()