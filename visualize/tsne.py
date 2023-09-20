import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# Load the dataset
digits = load_digits()
X, y = digits.data, digits.target
print(X)

# Apply TSNE
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X)

# Create a scatter plot
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='jet', alpha=0.5)

# Add colorbar and labels
plt.colorbar()
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

# Show plot
plt.savefig('a.png')