import os
import pandas as pd
import numpy as np

path = '/Users/giovannanicora/Documents/single-cell_hierarchies/GSE116256_RAW/'
lfiles = os.listdir(path)
mcells = ['HSC', 'Prog', 'GMP', 'Promono', 'Mono', 'cDC', 'pDC']
lines = []
sep='\t'
lines.append('PATIENT\tDAY\tNMalign\tNbenign')
patients = []
days = []
nmal = []
nben = []
for f in lfiles:
    if 'anno' in f:
        pname = f.split('_')[1].split('-')[0]
        day = ''
        if 'D' in f:
            day = f.split('-')[1].split('.anno')[0]
        df = pd.read_csv(path+f, sep='\t')
        # Selecting only myeloid cells (normal or tumor)
        ind_myeloid_malign = [i for i, x in enumerate(df.CellType) if 'like' in x]
        ind_myeloid_benign = [i for i, x in enumerate(df.CellType) if x in mcells]
        lines.append(pname+sep+day+sep+str(len(ind_myeloid_malign))+sep+str(len(ind_myeloid_benign)))
        patients.append(pname)
        days.append(day)
        nmal.append(len(ind_myeloid_malign))
        nben.append(len(ind_myeloid_benign))

r = pd.DataFrame(dict(Patient=patients, Day=days, NMal=nmal, NBen=nben))
# Get percentage of malign
perc_mal = np.array(nmal)/(np.array(nmal)+np.array(nben))
r['PercMal'] = perc_mal


dir_path = '/Users/giovannanicora/Documents/single-cell_hierarchies/per_amia/GSE116256_RAW/ge_files/'
all_df_anno = []
all_ge = []
df_anno = pd.DataFrame()
df_ge = pd.DataFrame()
for l in os.listdir(dir_path):

  #if 'anno' in l:
  #     df_anno = pd.concat((df_anno, pd.read_csv(dir_path+l, sep='\t')))

  #  else:
  if 'AML32' in l:
    df_ge = pd.concat((df_ge, pd.read_csv(dir_path + l, sep='\t')))

df_anno.to_csv('/Users/giovannanicora/Documents/single-cell_hierarchies/per_amia/GSE116256_RAW/anno_files/all_anno.txt',
               sep='\t', index=False)