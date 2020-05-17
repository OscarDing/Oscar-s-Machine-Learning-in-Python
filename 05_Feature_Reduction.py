import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from tools import plot_decision_regions
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import make_moons, make_circles
from kpca import rbf_kernel_pca

author = "Oscar Ding"

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

print(df_wine.head())

"""
Unsupervised dimensionality reduction via principal component analysis
"""
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# Standardizing the data
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Eigen-decomposition of the covariance matrix
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)
print("Eigenvalues: \n", eigen_vals)

# Total and explained variance
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1, 14), var_exp, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Feature transformation
# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

# projection matrix, take top two eigenvalues
w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
print('Matrix: W: \n', w)

# new features
X_train_pca = X_train_std.dot(w)

# visualize feature spread pattern by y
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0], X_train_pca[y_train == l, 1],
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# Principal component analysis in scikit-learn
pca = PCA()
X_train_pca_sk = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()

# Training logistic regression classifier using the first 2 principal components
lr = LogisticRegression(solver='liblinear', multi_class='auto', random_state=1)
lr = lr.fit(X_train_pca, y_train)

plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

"""
Supervised data compression via linear discriminant analysis
"""
# Computing the scatter matrices
# Calculate the mean vectors for each class
np.set_printoptions(precision=4)

mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print("MV: ", (label, mean_vecs[label - 1]))

# Compute the within-class scatter matrix:
d = 13  # number of features
S_W = np.zeros((d, d))

for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))  # scatter matrix for each class
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)  # make column vectors
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter

print('Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))

print('Class label distribution: %s' % np.bincount(y_train)[1:])
# Better: covariance matrix since classes are not equally distributed:
d = 13  # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter

print('Scaled within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))

# Compute the between-class scatter matrix:
mean_overall = np.mean(X_train_std, axis=0)
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train == i+1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)
    mean_overall = mean_overall.reshape(d, 1)
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

print('Between-class scatter matrix: %sx%s' % (S_B.shape[0], S_B.shape[1]))

# Selecting linear discriminants for the new feature subspace
# Solve the generalized eigenvalue problem for the matrix $S_W^{-1}S_B$:
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])

tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)

plt.bar(range(1, 14), discr, alpha=0.5, align='center', label='individual "discriminability"')
plt.step(range(1, 14), cum_discr, where='mid', label='cumulative "discriminability"')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
plt.show()

w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
              eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)

X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train == l, 0],
                X_train_lda[y_train == l, 1] * (-1),
                c=c, label=l, marker=m)

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# LDA via scikit-learn
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

lr = LogisticRegression(solver='liblinear', multi_class='auto', random_state=1)
lr = lr.fit(X_train_lda, y_train)

plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

X_test_lda = lda.transform(X_test_std)

plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

lda.explained_variance_ratio_

"""
Using kernel principal component analysis for nonlinear mappings
"""
# Example 1: Separating half-moon shapes
# X, y = make_moons(n_samples=100, random_state=123)
#
# plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
# plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)
#
# plt.tight_layout()
# plt.show()
#
# # show why pca doesn't work
# scikit_pca = PCA(n_components=2)
# X_spca = scikit_pca.fit_transform(X)
#
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
#
# ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1],
#               color='red', marker='^', alpha=0.5)
# ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1],
#               color='blue', marker='o', alpha=0.5)
#
# ax[1].scatter(X_spca[y == 0, 0], np.zeros((50, 1)) + 0.02,
#               color='red', marker='^', alpha=0.5)
# ax[1].scatter(X_spca[y == 1, 0], np.zeros((50, 1)) - 0.02,
#               color='blue', marker='o', alpha=0.5)
#
# ax[0].set_xlabel('PC1')
# ax[0].set_ylabel('PC2')
# ax[1].set_ylim([-1, 1])
# ax[1].set_yticks([])
# ax[1].set_xlabel('PC1')
#
# plt.tight_layout()
# plt.show()
#
# # kpca
# X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
# fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
# ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1],
#             color='red', marker='^', alpha=0.5)
# ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],
#             color='blue', marker='o', alpha=0.5)
#
# ax[1].scatter(X_kpca[y==0, 0], np.zeros((50,1))+0.02,
#             color='red', marker='^', alpha=0.5)
# ax[1].scatter(X_kpca[y==1, 0], np.zeros((50,1))-0.02,
#             color='blue', marker='o', alpha=0.5)
#
# ax[0].set_xlabel('PC1')
# ax[0].set_ylabel('PC2')
# ax[1].set_ylim([-1, 1])
# ax[1].set_yticks([])
# ax[1].set_xlabel('PC1')
#
# plt.tight_layout()
# plt.show()
#
# # Example 2: Separating concentric circles
# X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
#
# plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
# plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)
#
# plt.tight_layout()
# plt.show()
#
# scikit_pca = PCA(n_components=2)
# X_spca = scikit_pca.fit_transform(X)
#
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
#
# ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1],
#               color='red', marker='^', alpha=0.5)
# ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1],
#               color='blue', marker='o', alpha=0.5)
#
# ax[1].scatter(X_spca[y == 0, 0], np.zeros((500, 1)) + 0.02,
#               color='red', marker='^', alpha=0.5)
# ax[1].scatter(X_spca[y == 1, 0], np.zeros((500, 1)) - 0.02,
#               color='blue', marker='o', alpha=0.5)
#
# ax[0].set_xlabel('PC1')
# ax[0].set_ylabel('PC2')
# ax[1].set_ylim([-1, 1])
# ax[1].set_yticks([])
# ax[1].set_xlabel('PC1')
#
# plt.tight_layout()
# plt.show()
#
# X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
#
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
# ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1],
#               color='red', marker='^', alpha=0.5)
# ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1],
#               color='blue', marker='o', alpha=0.5)
#
# ax[1].scatter(X_kpca[y == 0, 0], np.zeros((500, 1)) + 0.02,
#               color='red', marker='^', alpha=0.5)
# ax[1].scatter(X_kpca[y == 1, 0], np.zeros((500, 1)) - 0.02,
#               color='blue', marker='o', alpha=0.5)
#
# ax[0].set_xlabel('PC1')
# ax[0].set_ylabel('PC2')
# ax[1].set_ylim([-1, 1])
# ax[1].set_yticks([])
# ax[1].set_xlabel('PC1')
#
# plt.tight_layout()
# plt.show()


X, y = make_moons(n_samples=100, random_state=123)
alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=1)

x_new = X[25]
x_new

x_proj = alphas[25] # original projection
x_proj

def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new - row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)

# projection of the "new" datapoint
x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
x_reproj

plt.scatter(alphas[y == 0, 0], np.zeros((50)),
            color='red', marker='^', alpha=0.5)
plt.scatter(alphas[y == 1, 0], np.zeros((50)),
            color='blue', marker='o', alpha=0.5)
plt.scatter(x_proj, 0, color='black',
            label='original projection of point X[25]', marker='^', s=100)
plt.scatter(x_reproj, 0, color='green',
            label='remapped point X[25]', marker='x', s=500)
plt.legend(scatterpoints=1)

plt.tight_layout()
plt.show()

# kpca in sklearn
X, y = make_moons(n_samples=100, random_state=123)
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)

plt.scatter(X_skernpca[y == 0, 0], X_skernpca[y == 0, 1],
            color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y == 1, 0], X_skernpca[y == 1, 1],
            color='blue', marker='o', alpha=0.5)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.show()





