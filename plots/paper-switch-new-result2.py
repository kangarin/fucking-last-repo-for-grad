import matplotlib.pyplot as plt
import numpy as np

# Scenarios
scenarios = ['Static', 'Periodic', 'Burst', 'Mixed']

# Revised performance data based on theoretical analysis
# Format: [combined_accuracy%, end-to-end latency(ms), reward]
performance_data = {
    'Rule-based': {
        'Static': [62.5, 87.3, 0.65],      # Simple decision, moderate accuracy
        'Periodic': [61.5, 125.5, 0.63],   # Reacts to changes but not optimally
        'Burst': [58.9, 135.4, 0.62],      # Quick response to bursts
        'Mixed': [59.5, 146.7, 0.56]       # Struggles with complex patterns
    },
    'CMAB': {
        'Static': [64.1, 92.1, 0.68],      # Good model selection after learning
        'Periodic': [63.5, 110.8, 0.65],   # Learns periodic patterns well
        'Burst': [59.5, 145.2, 0.60],      # Slower reaction than rules
        'Mixed': [60.5, 155.6, 0.59]       # Decent performance in complex scenarios
    },
    'DRL': {
        'Static': [64.5, 95.2, 0.67],      # Highest accuracy but with overhead
        'Periodic': [63.8, 105.1, 0.66],   # Best for predictable patterns
        'Burst': [60.1, 160.5, 0.57],      # Slowest to react to sudden changes
        'Mixed': [61.7, 154.3, 0.61]       # Best for complex scenarios
    }
}

# Create figure for comparing performance across scenarios
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Colors for different methods
colors = ['#ff7f0e', '#2ca02c', '#d62728']
methods = list(performance_data.keys())

# Plot Combined Accuracy
bar_width = 0.25
x = np.arange(len(scenarios))
for i, method in enumerate(methods):
    acc_data = [performance_data[method][scenario][0] for scenario in scenarios]
    ax1.bar(x + i*bar_width, acc_data, width=bar_width, label=method, color=colors[i])

ax1.set_ylabel('Combined Accuracy (%)', fontsize=12)
ax1.set_title('Combined Accuracy (mAP + Confidence)', fontsize=14)
ax1.set_xticks(x + bar_width)
ax1.set_xticklabels(scenarios)
ax1.set_ylim(57, 66)  # Adjusted for the data range
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Plot Latency
for i, method in enumerate(methods):
    latency_data = [performance_data[method][scenario][1] for scenario in scenarios]
    ax2.bar(x + i*bar_width, latency_data, width=bar_width, label=method, color=colors[i])

ax2.set_ylabel('End-to-End Latency (ms)', fontsize=12)
ax2.set_title('Processing Latency', fontsize=14)
ax2.set_xticks(x + bar_width)
ax2.set_xticklabels(scenarios)
ax2.set_ylim(80, 170)  # Adjusted for the data range
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Plot Reward
for i, method in enumerate(methods):
    reward_data = [performance_data[method][scenario][2] for scenario in scenarios]
    ax3.bar(x + i*bar_width, reward_data, width=bar_width, label=method, color=colors[i])

ax3.set_ylabel('Reward Value', fontsize=12)
ax3.set_title('Overall Performance Reward', fontsize=14)
ax3.set_xticks(x + bar_width)
ax3.set_xticklabels(scenarios)
ax3.set_ylim(0.55, 0.70)  # Adjusted for the data range
ax3.grid(axis='y', linestyle='--', alpha=0.7)

# Add legend at the bottom
handles, labels = ax3.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
           ncol=3, fancybox=True, shadow=True)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig('adaptive_methods_comparison.png', dpi=300, bbox_inches='tight')