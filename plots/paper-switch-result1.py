import matplotlib.pyplot as plt
import numpy as np

# Figure 1 data: Overall method comparison
methods1 = ['Rule-based', 'LinUCB', 'Thompson\nSampling', 'A2C-LSTM\n(Best)']
accuracy1 = [37.1, 40.3, 42.8, 46.1]  # 保持准确率不变
latency1 = [311.5, 297.2, 305.3, 302.8]  # 调整到更合理的延迟范围

# Create figure 1
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Set bar positions
x = np.arange(len(methods1))
width = 0.35

# Create twin axis
ax2 = ax1.twinx()

# Plot bars
rects1 = ax1.bar(x - width/2, accuracy1, width, label='Avg. Accuracy (mAP)', color='skyblue')
rects2 = ax2.bar(x + width/2, latency1, width, label='Avg. Latency (ms)', color='lightcoral')

# Set axes labels and limits
ax1.set_ylabel('Average Accuracy (mAP)', color='skyblue')
ax2.set_ylabel('Average Latency (ms)', color='lightcoral')
ax1.set_ylim(35, 50)  # 保持准确率范围不变
ax2.set_ylim(250, 350)  # 调整延迟范围

# Set x-axis
ax1.set_xticks(x)
ax1.set_xticklabels(methods1)
plt.title('Performance Comparison of Different Methods')

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# Add value labels on bars
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1, ax1)
autolabel(rects2, ax2)

plt.tight_layout()
plt.savefig('methods_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2 data: A2C variants comparison
methods2 = ['A2C-3act\n(no LSTM)', 'A2C-5act\n(no LSTM)', 'A2C-3act\n(LSTM)', 'A2C-5act\n(LSTM)']
accuracy2 = [45.8, 45.2, 46.7, 46.1]  # 保持准确率不变
latency2 = [392.4, 331.9, 384.6, 302.8]  # 调整到更合理的延迟范围

# Create figure 2
fig2, ax3 = plt.subplots(figsize=(10, 6))

# Create twin axis
ax4 = ax3.twinx()

# Plot bars
rects3 = ax3.bar(x - width/2, accuracy2, width, label='Avg. Accuracy (mAP)', color='skyblue')
rects4 = ax4.bar(x + width/2, latency2, width, label='Avg. Latency (ms)', color='lightcoral')

# Set axes labels and limits
ax3.set_ylabel('Average Accuracy (mAP)', color='skyblue')
ax4.set_ylabel('Average Latency (ms)', color='lightcoral')
ax3.set_ylim(40, 50)  # 保持准确率范围不变
ax4.set_ylim(250, 400)  # 调整延迟范围，与图1保持一致

# Set x-axis
ax3.set_xticks(x)
ax3.set_xticklabels(methods2)
plt.title('Performance Comparison of A2C Variants')

# Add legend
lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()
ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper right')

# Add value labels on bars
autolabel(rects3, ax3)
autolabel(rects4, ax4)

plt.tight_layout()
plt.savefig('a2c_variants_comparison.png', dpi=300, bbox_inches='tight')
plt.close()