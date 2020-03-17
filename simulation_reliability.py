from matplotlib import pyplot
import pandas as pd
import numpy as np
from collections import Counter
from pop_implemetation import pop_instance_training, check_is_border
from utils import make_classification_adjusted
from sklearn.metrics import *


def create_plot(df_pl, colors = {0:'blue', 1:'red'}, title='', group_by='label', path=None):
    fig, ax = pyplot.subplots()
    grouped = df_pl.groupby(group_by)
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    pyplot.title(title)
    if path is not None:
        pyplot.savefig(path)
    pyplot.show()


# Generate a simulated datasets with two classes
# drawn from a Gaussian distribution
# centers = [[-1, -1], [1, 1]]
# X, y = make_blobs(n_samples=6000, centers=centers,
#                  n_features=2, random_state=1,
#                  )
from sklearn.datasets import make_classification
# n_samples=100
X, y, clusters = make_classification_adjusted(n_samples=6000, n_classes=2, n_features=2, n_informative=2,
                           n_redundant=0,
                           random_state=1, weights=np.array([0.8, 0.2]),
                           class_sep=0.7,
                           n_clusters_per_class=2, shuffle=False)

class_num = Counter(y)
# scatter plot, dots colored by class value
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y, clusters=clusters))
create_plot(df, title='Total population')
create_plot(df, title='Total population - clusters', group_by='clusters',
            colors={1: 'royalblue', 2: 'red', 3: 'blue', 4: 'orangered'},
            path='/Users/giovannanicora/Documents/total_pop_by_clus.png')

# "Hiding" cluster 1
create_plot(df[df['clusters'] != 2],
            title='Training and Test set', path='/Users/giovannanicora/Documents/known_pop.png')
iCl2 = np.where(df['clusters'] == 2)
iCl134 = np.where(df['clusters'] != 2)


from sklearn.model_selection import train_test_split
# Selecting a balanced dataset
import numpy as np
np.random.seed(1)
rsize = round(Counter(y[iCl134])[1]*0.5)
bsize = rsize
iRedTrain = np.random.choice(np.where(y[iCl134]==1)[0], size=rsize, replace=False)
iBlueTrain = np.random.choice(np.where(y[iCl134]==0)[0], size=bsize, replace=False)


X_bal = X[iCl134][np.concatenate((iBlueTrain, iRedTrain))]
y_bal = y[iCl134][np.concatenate((iBlueTrain, iRedTrain))]
X_val = np.concatenate((X[iCl2], X[iCl134][[i for i in range(X[iCl134].shape[0]) if i not in np.concatenate((iBlueTrain, iRedTrain))]]))
y_val = np.concatenate((y[iCl2], y[iCl134][[i for i in range(X[iCl134].shape[0]) if i not in np.concatenate((iBlueTrain, iRedTrain))]]))


# Stratified selection for training and test
X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(X_bal,
                                                                    y_bal,
                                                                    stratify=y_bal,
                                                                    random_state=1,
                                                                    test_size=0.3)

create_plot(pd.DataFrame(dict(x=X_train_bal[:, 0],
                              y=X_train_bal[:, 1], label=y_train_bal)),
            title='Training set balanced')
create_plot(pd.DataFrame(dict(x=X_test_bal[:, 0],
                              y=X_test_bal[:, 1], label=y_test_bal)),
            title='Test set balanced')
create_plot(pd.DataFrame(dict(x=X_val[:, 0],
                              y=X_val[:, 1], label=y_val)),
            title='Validation set')

# POP
(mind_train, maxd_train, isborder_train, attr2outerb_train, attr2innerb_train,
 attr2outerb_train_val, attr2innerb_train_val) = pop_instance_training(X_train_bal, y_train_bal)
train_border_examples = X_train_bal[np.argwhere(np.sum(np.abs(isborder_train), axis=1)!=0).ravel(),:]
ytrain_border_examples = y_train_bal[np.argwhere(np.sum(np.abs(isborder_train), axis=1)!=0).ravel()]




# Training a Support Vector Machine on the balanced dataset
from sklearn.svm import LinearSVC
clf = LinearSVC(random_state=1, tol=1e-10, max_iter=2000)
clf.fit(X_train_bal, y_train_bal)
ypred_test_bal = clf.predict(X_test_bal)


n_border_test = [0]*X_test_bal.shape[0]
for c in list(range(X_test_bal.shape[0])):
    print(c)
    xtest = X_test_bal[c]
    bor = ''
    for i in range(X_test_bal.shape[1]):
        bor = bor + check_is_border(xtest[i], attr2outerb_train_val[i], inner=False)
        bor = bor + check_is_border(xtest[i], attr2outerb_train_val[i], inner=False)
        if 'Outer' in bor or 'Inner' in bor:
            n_border_test[c] = n_border_test[c]+1

rel_test = 1-np.array(n_border_test)/X_test_bal.shape[1]
unreliable_test = np.argwhere(rel_test==0).ravel()
reliable_test = np.argwhere(rel_test==1).ravel() # or !=0

print(accuracy_score(y_test_bal[unreliable_test], ypred_test_bal[unreliable_test]))
print(accuracy_score(y_test_bal[reliable_test], ypred_test_bal[reliable_test]))
print(accuracy_score(y_test_bal, ypred_test_bal))


print(precision_score(y_test_bal[unreliable_test], ypred_test_bal[unreliable_test]))
print(precision_score(y_test_bal[reliable_test], ypred_test_bal[reliable_test]))
print(precision_score(y_test_bal, ypred_test_bal))


df_train_bal = pd.DataFrame(dict(x=X_train_bal[:, 0],
                              y=X_train_bal[:, 1], label=y_train_bal))

df_test_bal = pd.DataFrame(dict(x=X_test_bal[:, 0],
                              y=X_test_bal[:, 1], label=y_test_bal))

colors = {0:'blue', 1:'red'}
fig, ax = pyplot.subplots()
grouped = df_train_bal.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])

grouped = df_test_bal.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key], marker='v')

pyplot.title('Training and Test')
pyplot.show()


acc_test_bal = accuracy_score(y_test_bal, ypred_test_bal)
prec_test_bal = precision_score(y_test_bal, ypred_test_bal)
rec_test_bal = recall_score(y_test_bal, ypred_test_bal)
tn_test_bal, fp_test_bal, fn_test_bal, tp_test_bal = confusion_matrix(y_test_bal, ypred_test_bal).ravel()


# Results on test set are pretty good! Let see what happens when we consider the (true)
# balanced dataset
ypred_val = clf.predict(X_val)

acc_val = accuracy_score(y_val, ypred_val)
prec_val = precision_score(y_val, ypred_val) # Precision decrease a lot! (form 86% to 56%)
rec_val = recall_score(y_val, ypred_val)

tn_val, fp_val, fn_val, tp_val = confusion_matrix(y_val, ypred_val).ravel()

print(classification_report(y_test_bal, ypred_test_bal))
print(classification_report(y_val, ypred_val))


## Reliability on Validation set
n_border_val = [0]*X_val.shape[0]
for c in list(range(X_val.shape[0])):
    # print(c)
    xval = X_val[c]
    bor = ''
    for i in range(X_val.shape[1]):
        bor = bor + check_is_border(xval[i], attr2outerb_train_val[i], inner=False)
        bor = bor + check_is_border(xval[i], attr2outerb_train_val[i], inner=False)
        if 'Outer' in bor or 'Inner' in bor:
            n_border_val[c] = n_border_val[c]+1

rel_val = 1-np.array(n_border_val)/X_val.shape[1]
unreliable_val = np.argwhere(rel_val==0).ravel()
reliable_val = np.argwhere(rel_val==1).ravel() # or !=0

print(accuracy_score(y_val[unreliable_val], ypred_val[unreliable_val]))
print(accuracy_score(y_val[reliable_val], ypred_val[reliable_val]))
print(accuracy_score(y_val, ypred_val))


print(precision_score(y_val[unreliable_val], ypred_val[unreliable_val]))
print(precision_score(y_val[reliable_val], ypred_val[reliable_val]))
print(precision_score(y_val, ypred_val))





tn_val_c2 = 0
tp_val_c2 = 0
fn_val_c2 = 0
fp_val_c2 = 0


for i in range(len(iCl2[0])):
    print(i)
    if y_val[i] == ypred_val[i]:
        if y_val[i] == 1:
            tp_val_c2 = tp_val_c2 + 1
        else:
            tn_val_c2 = tn_val_c2 + 1

    else:
        if y_val[i] == 1:
            fn_val_c2 = fn_val_c2 + 1
        else:
            fp_val_c2 = fp_val_c2 + 1
# Now we add the cluster #2 that we hided before
tp_val_c2/(tp_val_c2+fn_val_c2)
tp_val/(tp_val+fn_val_c2)


# Compute Kernel Similarity and Pairwise distance
from sklearn.metrics.pairwise import rbf_kernel
import time

t = time.time()
kernel_train = rbf_kernel(X_train_bal, X_train_bal)
elapsed_kernel = time.time() - t

t = time.time()
euc_train = euclidean_distances(X_train_bal, X_train_bal)
elapsed_euc = time.time() - t

kernel_test = rbf_kernel(X_test_bal, X_train_bal)
euc_test = euclidean_distances(X_test_bal, X_train_bal)

kernel_val = rbf_kernel(X_val, X_train_bal)
euc_val =euclidean_distances(X_val, X_train_bal)

iTestMisclassifiedTest = []
iTestCorreclyClassifiedTest = []
for i in range(X_test_bal.shape[0]):
    if ypred_test_bal[i] != y_test_bal[i]:
        iTestMisclassifiedTest.append(i)
    else:
        iTestCorreclyClassifiedTest.append(i)


iTestMisclassifiedVal = []
iTestCorreclyClassifiedVal = []


for i in range(X_val.shape[0]):
    if ypred_val[i] != y_val[i]:
        iTestMisclassifiedVal.append(i)
    else:
        iTestCorreclyClassifiedVal.append(i)


iMisclassifiedCl2 = []
iCorrectlyClassifiedCl2 = []
for i in list(range(len(iCl2[0]))):
    print(i)
    if ypred_val[i] != y_val[i]:
        iMisclassifiedCl2.append(i)
    else:
        iCorrectlyClassifiedCl2.append(i)


# Vediamo come varia la distribuzione della similarità tra gli esempi
# correttamente classificati e quelli misclassificati in base al kernel
kernel_correct_test = kernel_test[iTestCorreclyClassifiedTest, :]
kernel_nocorrect_test = kernel_test[iTestMisclassifiedTest, :]
euc_correct_test = euc_test[iTestCorreclyClassifiedTest,:]
euc_nocorrect_test = euc_test[iTestMisclassifiedTest, :]

kernel_correct_val = kernel_val[iTestCorreclyClassifiedVal, :]
kernel_nocorrect_val = kernel_val[iTestMisclassifiedVal, :]
euc_correct_val = euc_val[iTestCorreclyClassifiedVal,:]
euc_nocorrect_val = euc_val[iTestMisclassifiedVal, :]


fig, ax = pyplot.subplots(4,1)
ax[0].hist(kernel_nocorrect_test.ravel())
ax[0].set_title('Histogram of kernel similarities on uncorrectly classified examples on Test Set')
ax[1].hist(kernel_correct_test.ravel())
ax[1].set_title('Histogram of kernel similarities on correctly classified examples on Test Set')
ax[2].hist(kernel_nocorrect_val.ravel())
ax[2].set_title('Histogram of kernel similarities on uncorreclty classified examples on Validation')
ax[3].hist(kernel_correct_val.ravel())
ax[3].set_title('Histogram of kernel similarities on correctly classified examples on Validation')
pyplot.show()


# For each test and validation example, keep the highest similarity with training set
fig, ax = pyplot.subplots(4,1)
ax[0].hist(np.max(kernel_nocorrect_test, axis=1))
ax[0].set_title('Histogram of max kernel similarities on uncorrectly classified examples on Test Set')
ax[1].hist(np.max(kernel_correct_test, axis=1))
ax[1].set_title('Histogram of max kernel similarities on correctly classified examples on Test Set')
ax[2].hist(np.max(kernel_nocorrect_val, axis=1))
ax[2].set_title('Histogram of max kernel similarities on uncorreclty classified examples on Validation')
ax[3].hist(np.max(kernel_correct_val, axis=1))
ax[3].set_title('Histogram of max kernel similarities on correctly classified examples on Validation')
pyplot.show()



fig, ax = pyplot.subplots(2,1)
ax[0].boxplot([kernel_correct_test.ravel(), kernel_nocorrect_test.ravel()])
ax[0].set_title('Correctly and uncorrectly classified examples on Test Set (Kernel)')
ax[1].boxplot([kernel_correct_val, kernel_nocorrect_val.ravel()])
ax[1].set_title('Boxplot of kernel similarities on uncorreclty classified examples on Validation')
pyplot.show()

fig, ax = pyplot.subplots(2,1)
ax[0].boxplot([np.max(kernel_correct_test, axis=1), np.max(kernel_nocorrect_test, axis=1)])
ax[0].set_title('Correctly and uncorrectly classified examples on Test Set (Max Kernel)')
ax[1].boxplot([np.max(kernel_correct_val, axis=1), np.max(kernel_nocorrect_val, axis=1)])
ax[1].set_title('Correctly and uncorreclty classified examples on Validation')
pyplot.show()

fig, ax = pyplot.subplots(2,1)
ax[0].boxplot([euc_correct_test.ravel(), euc_nocorrect_test.ravel()])
ax[0].set_title('Correctly and uncorrectly classified examples on Test Set')
ax[1].boxplot([euc_correct_val, euc_nocorrect_val.ravel()])
ax[1].set_title('Boxplot of kernel similarities on uncorreclty classified examples on Validation')
pyplot.show()



# Provo con una heatmap
fig, ax = pyplot.subplots()
im = ax.imshow(np.concatenate((kernel_nocorrect_test, kernel_correct_test)))
pyplot.show()
fig, ax = pyplot.subplots()
im = ax.imshow(np.concatenate((kernel_nocorrect_val, kernel_correct_val)).T)
pyplot.show()



# Secondo tentativo: uso il kolmogorov smirnov
