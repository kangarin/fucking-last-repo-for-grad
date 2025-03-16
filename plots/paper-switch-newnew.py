import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Sample data (replace with your experimental results)
algorithms = ['CMAB', 'CMAB-Meta', 'LSTM-DRL']

# Average overall accuracy
accuracy_means = [0.987, 1.054, 1.012]

# Average end-to-end latency (ms)
latency_means = [1053.7, 1721.4, 472.6]

# Average queue length
queue_means = [6.71, 9.63, 1.96]

# Create figure and subplots (1 row, 3 columns)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
fig.subplots_adjust(wspace=0.3)

# Bar width
bar_width = 0.6

# X positions for the bars
x_pos = np.arange(len(algorithms))

# Plot 1: Average Overall Accuracy
bars1 = ax1.bar(x_pos, accuracy_means, bar_width, 
               align='center', alpha=0.8, edgecolor='black',
               color=['#3274A1', '#E1812C', '#3A923A'])
ax1.set_title('Average Overall Accuracy', fontsize=14, fontweight='bold')
ax1.set_ylabel('Overall Accuracy (mAP + conf)', fontsize=12)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(algorithms, fontsize=12)
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.set_ylim(bottom=0.9, top=1.1)

# Add value labels on the bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
             f'{height:.3f}', ha='center', va='bottom', fontsize=11)

# Plot 2: Average End-to-End Latency
bars2 = ax2.bar(x_pos, latency_means, bar_width, 
               align='center', alpha=0.8, edgecolor='black',
               color=['#3274A1', '#E1812C', '#3A923A'])
ax2.set_title('Average End-to-End Latency', fontsize=14, fontweight='bold')
ax2.set_ylabel('Latency (ms)', fontsize=12)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(algorithms, fontsize=12)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.set_ylim(bottom=0)

# Add value labels on the bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 20,
             f'{height:.1f} ms', ha='center', va='bottom', fontsize=11)

# Plot 3: Average Queue Length
bars3 = ax3.bar(x_pos, queue_means, bar_width, 
               align='center', alpha=0.8, edgecolor='black',
               color=['#3274A1', '#E1812C', '#3A923A'])
ax3.set_title('Average Queue Length', fontsize=14, fontweight='bold')
ax3.set_ylabel('Queue Length', fontsize=12)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(algorithms, fontsize=12)
ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
ax3.grid(axis='y', linestyle='--', alpha=0.7)
ax3.set_ylim(bottom=0)

# Add value labels on the bars
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.1f}', ha='center', va='bottom', fontsize=11)

# Add a main title for the entire figure
plt.suptitle('Performance Comparison of Adaptive Algorithms', fontsize=16, fontweight='bold', y=0.98)

# Add figure legend
fig.legend(bars1, algorithms, 
           loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01))

# Layout adjustment
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# Save the figure (optional)
# plt.savefig('adaptive_algorithms_comparison.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()