# %% md

# Tumor/Normal classification from Single-Cell gene expression data

# %%
import pandas as pd
from collections import Counter
import numpy as np
import scanpy as sc

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.svm import SVC
from sklearn.metrics import *


# %%

path_anno = '/Users/giovannanicora/Documents/single-cell_hierarchies/GSM3587943_AML329-D20.anno.txt'
anno = pd.read_csv(path_anno, sep='\t')

# %%

anno.columns
print(Counter(anno.PredictionRefined))
print(Counter(anno.CellType))

# %%

# Myeloid cells
mcells = ['HSC', 'Prog', 'GMP', 'Promono', 'Mono', 'cDC', 'pDC']

# Selecting only myeloid cells (normal or tumor)
ind_myeloid_malign = [i for i, x in enumerate(anno.CellType) if 'like' in x]
ind_myeloid_benign = [i for i, x in enumerate(anno.CellType) if x in mcells]

# %%

len(ind_myeloid_benign)

# %%

len(ind_myeloid_malign)

# %%

df = pd.read_csv('/Users/giovannanicora/Documents/single-cell_hierarchies/AML329-D20.dem.txt',
                 sep='\t')

# %%

df.shape

# %%

# df.head()
gene_names = df['Gene']
# Selecting only Myeloid
barcodes_myeloid = list(anno.Cell[ind_myeloid_malign]) + list(anno.Cell[ind_myeloid_benign])
m_col = [i for i, x in enumerate(df.columns) if x in barcodes_myeloid]
df = df.iloc[:, m_col]
df.shape

# %%

barcodes = [x for x in df.columns if 'Gene' not in x]
anndata = sc.AnnData(X=df.T.to_numpy(), obs=barcodes, var=gene_names.values)

# %%

anndata.var_names = gene_names.values

# %%

anndata.obs_names = barcodes

# %%

anndata.raw = anndata
sc.pl.highest_expr_genes(anndata, n_top=20, )

# %%

anndata.shape

# %%

# Normalization and filtering
sc.pp.filter_genes(anndata, min_cells=50)

# %%

anndata.shape

# %%

sc.pp.normalize_total(anndata)
sc.pp.log1p(anndata)

# %%

np.max(anndata.raw.X)
np.max(anndata.X)

# %%

sc.pl.highest_expr_genes(anndata, n_top=50, )

# %% md

## Machine Learning

# %%

barcodes2class = dict(zip(anno.Cell,
                          anno.PredictionRefined))
y_true = [barcodes2class[x] for x in anndata.obs_names]
X = anndata.X
X_train = X
classdict = dict(normal=0, malignant=1)
y_true_num = [classdict[x] for x in y_true]
y_train = np.array(y_true_num)



# %%

# POP on the training set
from pop_implemetation import *

(mind_train, maxd_train, isborder_train, attr2outerb_train, attr2innerb_train,
 attr2outerb_train_val, attr2innerb_train_val) = pop_instance_training(X_train, y_train)
train_border_examples = X_train[np.argwhere(np.sum(np.abs(isborder_train), axis=1) != 0).ravel(), :]
ytrain_border_examples = y_train[np.argwhere(np.sum(np.abs(isborder_train), axis=1) != 0).ravel()]

# %%

mind_train

# %%

maxd_train

# %%

clf_svm = SVC()
scores_svm = cross_validate(clf_svm, X_train,
                            y_train,
                            scoring={'accuracy_score': make_scorer(accuracy_score),
                                     'precision_score': make_scorer(precision_score),
                                     'prc': make_scorer(average_precision_score),
                                     'mcc': make_scorer(matthews_corrcoef)})
from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression(penalty='l1', solver='liblinear')  # Lasso
scores_lr = cross_validate(clf_lr, X_train,
                           y_train,
                           scoring={'accuracy_score': make_scorer(accuracy_score),
                                    'precision_score': make_scorer(precision_score),
                                    'prc': make_scorer(average_precision_score),
                                    'mcc': make_scorer(matthews_corrcoef)})

# %%

# Training on the entire training set
clf_svm.fit(X_train, y_train)
clf_lr.fit(X_train, y_train)

# %%

scores_svm

# %%

scores_lr

# %%

[np.mean(scores_svm[x]) for x in scores_svm.keys()]

# %%

[np.mean(scores_lr[x]) for x in scores_lr.keys()]

# %% md

### Predicting on a Validation dataset


#%%
path_anno = '/Users/giovannanicora/Documents/single-cell_hierarchies/GSM3587932_AML328-D0.anno.txt'
anno329d0 = pd.read_csv(path_anno, sep='\t')
anno329d37 = pd.read_csv('/Users/giovannanicora/Documents/single-cell_hierarchies/GSM3587945_AML329-D37.anno.txt',
                         sep='\t')


# Selecting only myeloid cells (normal or tumor)
ind_myeloid_malign_d0 = [i for i, x in enumerate(anno329d0.CellType) if 'like' in x]
ind_myeloid_benign_d0 = [i for i, x in enumerate(anno329d0.CellType) if x in mcells]
ind_myeloid_malign_d37 = [i for i, x in enumerate(anno329d37.CellType) if 'like' in x]
ind_myeloid_benign_d37 = [i for i, x in enumerate(anno329d37.CellType) if x in mcells]

df_d0 = pd.read_csv('/Users/giovannanicora/Documents/single-cell_hierarchies/AML328-D0.dem.txt',
                 sep='\t')
df_d37 = pd.read_csv('/Users/giovannanicora/Documents/single-cell_hierarchies/AML329-D37.dem.txt',
                 sep='\t')
iGenes_d0 = [i for i, x in enumerate(df_d0['Gene']) if x in anndata.var_names]
iGenes_d37 = [i for i, x in enumerate(df_d37['Gene']) if x in anndata.var_names]

gene_names_d0 = df_d0['Gene']
gene_names_d37 = df_d37['Gene']

# Selecting only Myeloid
barcodes_myeloid_d0 = list(anno329d0.Cell[ind_myeloid_malign_d0]) + list(anno329d0.Cell[ind_myeloid_benign_d0])
barcodes_myeloid_d37 = list(anno329d37.Cell[ind_myeloid_malign_d37]) + list(anno329d37.Cell[ind_myeloid_benign_d37])

m_col_d0 = [i for i, x in enumerate(df_d0.columns) if x in barcodes_myeloid_d0]
m_col_d37 = [i for i, x in enumerate(df_d37.columns) if x in barcodes_myeloid_d37]

df_d37 = df_d37.iloc[iGenes_d37, m_col_d37]
df_d0 = df_d0.iloc[iGenes_d0, m_col_d0]


barcodes_d0 = [x for x in df_d0.columns if 'Gene' not in x]
barcodes_d37 = [x for x in df_d37.columns if 'Gene' not in x]

df = pd.concat((df_d0, df_d37),axis=1, ignore_index=True)

barcodes = barcodes_d0+barcodes_d37
anndata_val = sc.AnnData(X=df.T.to_numpy(), obs=barcodes, var=anndata.var_names)

anndata_val.var_names = anndata.var_names

anndata_val.obs_names = barcodes

# %%

anndata_val.shape

# %%

# Normalization per cell
sc.pp.normalize_total(anndata_val)
sc.pp.log1p(anndata_val)

# %%


# %%


barcodes2class = dict(zip(pd.concat((anno329d0.Cell, anno329d37.Cell)),
                          pd.concat((anno329d0.PredictionRefined, anno329d37.PredictionRefined))))



y_true_val = [barcodes2class[x] for x in anndata_val.obs_names]

X_val = anndata_val.X
y_true_val_num = [classdict[x] for x in y_true_val]
y_val = np.array(y_true_val_num)
# %%
vald = Counter(y_val)
# %%

ypred_val_svm = clf_svm.predict(X_val)
ypred_val_lr = clf_lr.predict(X_val)

acc_val_svm = accuracy_score(y_val, ypred_val_svm)
acc_val_lr = accuracy_score(y_val, ypred_val_lr)

prec_val_svm = precision_score(y_val, ypred_val_svm)  # Precision decrease a lot! (form 86% to 56%)
prec_val_lr = precision_score(y_val, ypred_val_lr)  # Precision decrease a lot! (form 86% to 56%)

rec_val_svm = recall_score(y_val, ypred_val_svm)
rec_val_lr = recall_score(y_val, ypred_val_lr)

tn_val_svm, fp_val_svm, fn_val_svm, tp_val_svm = confusion_matrix(y_val, ypred_val_svm).ravel()
tn_val_lr, fp_val_lr, fn_val_lr, tp_val_lr = confusion_matrix(y_val, ypred_val_lr).ravel()

# %%
acc_val_svm
acc_val_lr

prec_val_svm
prec_val_lr

rec_val_svm
rec_val_lr

spec_svm = tn_val_svm / (tn_val_svm + fp_val_svm)
spec_lr = tn_val_lr / (tn_val_lr + fp_val_lr)

spec_svm
spec_lr
# %%

## Reliability on Validation set
n_border_val = [0] * X_val.shape[0]
for c in list(range(X_val.shape[0])):
    # print(c)
    xval = X_val[c]
    bor = ''
    for i in range(X_val.shape[1]):
        bor = bor + check_is_border(xval[i], attr2outerb_train_val[i], inner=False)
        bor = bor + check_is_border(xval[i], attr2outerb_train_val[i], inner=False)
        if 'Outer' in bor or 'Inner' in bor:
            n_border_val[c] = n_border_val[c] + 1

# %%
rel_val = 1 - np.array(n_border_val) / X_val.shape[1]
unreliable_val = np.argwhere(rel_val == 0).ravel()
reliable_val = np.argwhere(rel_val > 0).ravel()  # or !=0

# %%
reliable_val
unreliable_val
# %%

acc_unrel_val_svm = accuracy_score(y_val[unreliable_val], ypred_val_svm[unreliable_val])
prec_unrel_val_svm = precision_score(y_val[unreliable_val], ypred_val_svm[unreliable_val])
mcc_unrel_val_svm = matthews_corrcoef(y_val[unreliable_val], ypred_val_svm[unreliable_val])

acc_rel_val_svm = accuracy_score(y_val[reliable_val], ypred_val_svm[reliable_val])
prec_rel_val_svm = precision_score(y_val[reliable_val], ypred_val_svm[reliable_val])
mcc_rel_val_svm = matthews_corrcoef(y_val[reliable_val], ypred_val_svm[reliable_val])

acc_val_svm = accuracy_score(y_val, ypred_val_svm)
prec_val_svm = precision_score(y_val, ypred_val_svm)
mcc_val_svm = matthews_corrcoef(y_val, ypred_val_svm)

# %%

print('Number of elements in validation:' + str(X_val.shape[0]))
print('Number of reliable examples in validation:' + str(len(reliable_val)))
print('Number of unreliable examples in validation:' + str(len(unreliable_val)))

# %% md Performance on the validation

# %%

print('Accuracy on validation:' + str(acc_val_svm))
print('Precision on validation:' + str(prec_val_svm))
print('MCC on validation:' + str(mcc_val_svm))


# %%

print('Accuracy on reliable validation:' + str(acc_rel_val_svm))
print('Precision on reliable validation:' + str(prec_rel_val_svm))
print('MCC on reliable validation:' + str(mcc_rel_val_svm))



# %%

print('Accuracy on unreliable validation:' + str(acc_unrel_val_svm))
print('Precision on unreliable validation:' + str(prec_unrel_val_svm))
print('MCC on unreliable validation:' + str(mcc_unrel_val_svm))

# %%

z = 1.96
interval = z * np.sqrt(mcc_val_svm * (1 - mcc_val_svm) / X_val.shape[0])
conf_inf = [mcc_val_svm - interval, mcc_val_svm + interval]
print('Confidence intervals for MCC on Validation:' + str(conf_inf[0]) + '-' + str(conf_inf[1]))

# %%

z = 1.96
interval = z * np.sqrt(prec_val_svm * (1 - prec_val_svm) / X_val.shape[0])
conf_inf = [prec_val_svm - interval, prec_val_svm + interval]
print('Confidence intervals for Precision on Validation:' + str(conf_inf[0]) + '-' + str(conf_inf[1]))
