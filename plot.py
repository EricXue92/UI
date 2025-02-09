import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# In-distribution (e.g., a normal distribution centered at 0)
in_distribution = np.random.normal(loc=0, scale=1, size=100)

# Out-of-distribution (e.g., a normal distribution centered at 5, non-overlapping)
out_of_distribution = np.random.normal(loc=6, scale=0.5, size=100)

# Plot the two distributions
plt.figure(figsize=(8, 6))

# Plot in-distribution curve
sns.kdeplot(in_distribution, label='Training', color='blue',linewidth=3)

# Plot out-of-distribution curve
sns.kdeplot(out_of_distribution, label='Testing', color='red',linewidth=3)
# plt.title('In-Distribution vs Out-of-Distribution')
# plt.xlabel('Value')
plt.ylabel('')
plt.legend(fontsize=20)
# plt.grid(True, alpha=0.3)
plt.tight_layout() # Add this line for tight layout
plt.savefig('in_vs_out_distribution.pdf')
plt.show()
