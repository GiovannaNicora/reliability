from sklearn.datasets import make_blobs
from matplotlib import pyplot
from pandas import DataFrame



# Generate a simulated datasets with two classes
# drawn from a Gaussian distribution
centers = [[-1, -1], [1, 1]]
X, y = make_blobs(n_samples=500, centers=centers,
                  n_features=2, random_state=1,
                  )
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()

# Now I simulate the fact th
