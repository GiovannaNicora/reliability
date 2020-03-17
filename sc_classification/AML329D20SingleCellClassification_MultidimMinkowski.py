#!/usr/bin/env python
# coding: utf-8

# # Tumor/Normal classification from Single-Cell gene expression data

# In[1]:


import pandas as pd
from collections import Counter
import numpy as np
import scanpy as sc


from sklearn.model_selection import train_test_split, cross_validate
from sklearn.svm import SVC
from sklearn.metrics import *


# In[2]:


dir_anno = '/Users/giovannanicora/Documents/single-cell_hierarchies/per_amia/GSE116256_RAW/anno_files/'
path_anno = dir_anno+'GSM3587943_AML329-D20.anno.txt.gz'
anno = pd.read_csv(path_anno, sep='\t')


# In[3]:


anno.columns
print(Counter(anno.PredictionRefined))
print(Counter(anno.CellType))


# In[4]:


# Myeloid cells
mcells = ['HSC', 'Prog', 'GMP', 'Promono', 'Mono', 'cDC', 'pDC']

# Selecting only myeloid cells (normal or tumor)
ind_myeloid_malign = [i for i,x in enumerate(anno.CellType) if 'like' in x]
ind_myeloid_benign = [i for i,x in enumerate(anno.CellType) if x in mcells]


# In[5]:


len(ind_myeloid_benign)


# In[6]:


len(ind_myeloid_malign)


# In[5]:


ge_files = '/Users/giovannanicora/Documents/single-cell_hierarchies/per_amia/GSE116256_RAW/ge_files/'
df = pd.read_csv(ge_files+'GSM3587942_AML329-D20.dem.txt.gz',
                 sep='\t')


# In[8]:


df.shape


# In[6]:


#df.head()
gene_names = df['Gene']
# Selecting only Myeloid
barcodes_myeloid = list(anno.Cell[ind_myeloid_malign])+list(anno.Cell[ind_myeloid_benign])
m_col = [i for i,x in enumerate(df.columns) if x in barcodes_myeloid]
df = df.iloc[:, m_col]
df.shape


# In[7]:


barcodes = [x for x in df.columns if 'Gene' not in x]
anndata = sc.AnnData(X=df.T.to_numpy(), obs=barcodes, var=gene_names.values)


# In[8]:


anndata.var_names = gene_names.values


# In[9]:


anndata.obs_names = barcodes


# In[10]:


anndata.raw = anndata
sc.pl.highest_expr_genes(anndata, n_top=20, )


# In[11]:


anndata.shape


# In[11]:


# Normalization and filtering
sc.pp.filter_genes(anndata, min_cells=50)


# In[16]:


anndata.shape


# In[12]:


sc.pp.normalize_total(anndata, target_sum=1e4)
sc.pp.log1p(anndata)


# In[13]:


np.max(anndata.raw.X)
np.max(anndata.X)


# In[14]:


sc.pl.highest_expr_genes(anndata, n_top=50, )


# ## Machine Learning

# In[15]:


barcodes2class = dict(zip(anno.Cell,
                          anno.PredictionRefined))
y_true = [barcodes2class[x] for x in anndata.obs_names]
X = anndata.X
X_train = X
classdict = dict(normal=0, malignant=1)
y_true_num = [classdict[x] for x in y_true]
y_train = np.array(y_true_num)


# POP on training set

# In[16]:


# POP on the training set
from pop_implemetation import *
m='minkowski'
(isborder_train, mind_train, maxd_train) = find_multidim_borders(X_train, y_train,
                                                                 metric=m, r=1)
train_border_examples = X_train[np.argwhere(np.abs(isborder_train) != 0).ravel(), :]
ytrain_border_examples = y_train[np.argwhere(np.abs(isborder_train) != 0).ravel()]

# In[16]:


clf_svm = SVC()
scores_svm = cross_validate(clf_svm, X_train, 
                        y_train, cv=5,
                        scoring={'accuracy_score':make_scorer(accuracy_score),
                                 'precision_score':make_scorer(precision_score),
                                 'prc':make_scorer(average_precision_score),
                                 'mcc':make_scorer(matthews_corrcoef)})

                                


# In[20]:


# Training on the entire training set
clf_svm.fit(X_train, y_train)


# In[21]:


scores_svm


# In[22]:


mean_cv = [x+':'+str(np.mean(scores_svm[x])) for x in scores_svm.keys() ]
std_cv = [x+':'+str(np.std(scores_svm[x])) for x in scores_svm.keys() ]


# In[19]:


print("Accuracy: %0.2f (+/- %0.2f)" % (scores_svm['test_accuracy_score'].mean(), scores_svm['test_accuracy_score'].std() * 2))
print("Precision: %0.2f (+/- %0.2f)" % (scores_svm['test_precision_score'].mean(), scores_svm['test_precision_score'].std() * 2))

print("MCC: %0.2f (+/- %0.2f)" % (scores_svm['test_mcc'].mean(), scores_svm['test_mcc'].std() * 2))


# In[17]:


scores_svm.keys()


# ### Predicting on a Validation dataset
# We try to predict the class of new unseen examples from different patients

# In[26]:


path_anno = dir_anno+'GSM3587932_AML328-D0.anno.txt.gz'
anno329d0 = pd.read_csv(path_anno, sep='\t')
anno329d37 = pd.read_csv(dir_anno+'GSM3587945_AML329-D37.anno.txt.gz',
                         sep='\t')


# Selecting only myeloid cells (normal or tumor)
ind_myeloid_malign_d0 = [i for i, x in enumerate(anno329d0.CellType) if 'like' in x]
ind_myeloid_benign_d0 = [i for i, x in enumerate(anno329d0.CellType) if x in mcells]
ind_myeloid_malign_d37 = [i for i, x in enumerate(anno329d37.CellType) if 'like' in x]
ind_myeloid_benign_d37 = [i for i, x in enumerate(anno329d37.CellType) if x in mcells]


# In[27]:


df_d0 = pd.read_csv(ge_files+'GSM3587931_AML328-D0.dem.txt.gz',
                 sep='\t')
df_d37 = pd.read_csv(ge_files+'GSM3587944_AML329-D37.dem.txt.gz',
                 sep='\t')
iGenes_d0 = [i for i, x in enumerate(df_d0['Gene']) if x in anndata.var_names]
iGenes_d37 = [i for i, x in enumerate(df_d37['Gene']) if x in anndata.var_names]

gene_names_d0 = df_d0['Gene']
gene_names_d37 = df_d37['Gene']


# In[28]:


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


# In[29]:


anndata_val.shape


# In[30]:


# Normalization per cell
sc.pp.normalize_total(anndata_val)
sc.pp.log1p(anndata_val)


# In[31]:



barcodes2class = dict(zip(pd.concat((anno329d0.Cell, anno329d37.Cell)),
                          pd.concat((anno329d0.PredictionRefined, anno329d37.PredictionRefined))))



y_true_val = [barcodes2class[x] for x in anndata_val.obs_names]

X_val = anndata_val.X
y_true_val_num = [classdict[x] for x in y_true_val]
y_val = np.array(y_true_val_num)


# In[32]:


vald = Counter(y_val)
vald


# In[33]:


ypred_val_svm = clf_svm.predict(X_val)

acc_val_svm = accuracy_score(y_val, ypred_val_svm)

prec_val_svm = precision_score(y_val, ypred_val_svm)  # Precision decrease a lot! (form 86% to 56%)

rec_val_svm = recall_score(y_val, ypred_val_svm)

tn_val_svm, fp_val_svm, fn_val_svm, tp_val_svm = confusion_matrix(y_val, ypred_val_svm).ravel()


# In[34]:


print('Accuracy on validation:'+str(acc_val_svm))
print('Precision on validation:'+str(prec_val_svm))
print('Recall on validation:'+str(rec_val_svm))
spec_svm = tn_val_svm / (tn_val_svm + fp_val_svm)
print('Specificity on validation:'+str(spec_svm))


# In[35]:


## Reliability on Validation set
is_border_val = [0] * X_val.shape[0]
for c in list(range(X_val.shape[0])):

    xval = X_val[c]

    bor = check_is_multidim_border(X_val, xval,
                                   isborder_train, mind_train, maxd_train, metric=m, r=1) # X, z, isborder, min_d, max_d
    if 'Outer' in bor or 'Inner' in bor:
        is_border_val[c] = 1


# In[36]:

# Reliability on the test set
rel_val = 1-np.array(is_border_val)
unreliable_val = np.argwhere(rel_val == 0).ravel()
reliable_val = np.argwhere(rel_val != 0).ravel() # or !=0


# In[37]:


reliable_val
unreliable_val


# In[38]:


acc_unrel_val_svm = accuracy_score(y_val[unreliable_val], ypred_val_svm[unreliable_val])
prec_unrel_val_svm = precision_score(y_val[unreliable_val], ypred_val_svm[unreliable_val])
mcc_unrel_val_svm = matthews_corrcoef(y_val[unreliable_val], ypred_val_svm[unreliable_val])

acc_rel_val_svm = accuracy_score(y_val[reliable_val], ypred_val_svm[reliable_val])
prec_rel_val_svm = precision_score(y_val[reliable_val], ypred_val_svm[reliable_val])
mcc_rel_val_svm = matthews_corrcoef(y_val[reliable_val], ypred_val_svm[reliable_val])

acc_val_svm = accuracy_score(y_val, ypred_val_svm)
prec_val_svm = precision_score(y_val, ypred_val_svm)
mcc_val_svm = matthews_corrcoef(y_val, ypred_val_svm)

tn_val_svm_rel, fp_val_svm_rel, fn_val_svm_rel, tp_val_svm_rel = confusion_matrix(y_val[reliable_val], 
                                                                                  ypred_val_svm[reliable_val]).ravel()
tn_val_svm_unrel, fp_val_svm_unrel, fn_val_svm_unrel, tp_val_svm_unrel = confusion_matrix(y_val[unreliable_val], 
                                                                                  ypred_val_svm[unreliable_val]).ravel()
spec_rel_val = tn_val_svm_rel/(tn_val_svm_rel+fp_val_svm_rel)
spec_unrel_val = tn_val_svm_unrel/(tn_val_svm_unrel+fp_val_svm_unrel)


sens_rel_val = tp_val_svm_rel/(tp_val_svm_rel+fn_val_svm_rel)
sens_unrel_val = tp_val_svm_unrel/(tp_val_svm_unrel+fn_val_svm_unrel)
sens_val = tp_val_svm/(tp_val_svm+fn_val_svm)


# In[39]:


print('Number of elements in validation:' + str(X_val.shape[0]))
print('Number of reliable examples in validation:' + str(len(reliable_val)))
print('Number of unreliable examples in validation:' + str(len(unreliable_val)))


# In[39]:





# In[40]:


print('Accuracy on validation:' + str(acc_val_svm))
print('Accuracy on reliable validation:' + str(acc_rel_val_svm))
print('Accuracy on unreliable validation:' + str(acc_unrel_val_svm))

print('Precision on validation:' + str(prec_val_svm))
print('Precision on reliable validation:' + str(prec_rel_val_svm))
print('Precision on unreliable validation:' + str(prec_unrel_val_svm))


print('MCC on validation:' + str(mcc_val_svm))
print('MCC on reliable validation:' + str(mcc_rel_val_svm))
print('MCC on unreliable validation:' + str(mcc_unrel_val_svm))

print('Sensitivity on validation: '+str(sens_val))
print('Sensitivity on reliable validation: '+str(sens_rel_val))
print('Sensitivity on unreliable validation: '+str(sens_unrel_val))


# In[41]:


z = 1.96
interval = z * np.sqrt(mcc_val_svm * (1 - mcc_val_svm) / X_val.shape[0])
conf_inf = [mcc_val_svm - interval, mcc_val_svm + interval]
print('Confidence intervals for MCC on Validation:' + str(conf_inf[0]) + '-' + str(conf_inf[1]))


# In[42]:


z = 1.96
interval = z * np.sqrt(prec_val_svm * (1 - prec_val_svm) / X_val.shape[0])
conf_inf = [prec_val_svm - interval, prec_val_svm + interval]
print('Confidence intervals for Precision on Validation:' + str(conf_inf[0]) + '-' + str(conf_inf[1]))


# In[43]:


z = 1.96
interval = z * np.sqrt(sens_val * (1 - sens_val) / X_val.shape[0])
conf_inf = [sens_val - interval, sens_val + interval]
print('Confidence intervals for Sensitivity on Validation:' + str(conf_inf[0]) + '-' + str(conf_inf[1]))

