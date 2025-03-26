import os
import scanpy as sc
import anndata
import numpy as np
import scipy
import STAligner
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score as ari_score
warnings.filterwarnings("ignore")
import torch
used_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import pandas as pd
from STAligner import ST_utils
from scipy import sparse

Batch_list = []
adj_list = []
section_ids = ['slice_0','slice_1','slice_2','slice_3','slice_4','slice_5','slice_6','slice_7','slice_8','slice_9','slice_10','slice_11']
print(section_ids)

for section_id in section_ids:
    print(section_id)
    input_dir = os.path.join('data/', section_id)
    adata = sc.read_h5ad(input_dir+'.h5ad')
    adata.obs_names = [x + '_' + section_id for x in adata.obs_names]
    STAligner.Cal_Spatial_Net(adata, rad_cutoff=0.015)

    # Normalization
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000) #ensure enough common HVGs in the combined matrix
    adata = adata[:, adata.var['highly_variable']]

    adj_list.append(adata.uns['adj'])
    Batch_list.append(adata)

adata_concat = anndata.concat(Batch_list, label="slice_name", keys=section_ids)
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
print('adata_concat.shape: ', adata_concat.shape)

adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1,len(section_ids)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
adata_concat.uns['edgeList'] = np.nonzero(adj_concat) 

iter_comb = [(0, 5), (1, 5), (2, 5),(3,5),(4,5),(6,5),(7,5),(8,5),(9,5),(10,5),(11,5)] ## Fix slice 3 as reference to align

adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh = 10, iter_comb = iter_comb,
                                                        margin=1.0,  device=used_device)

sc.pp.neighbors(adata_concat, use_rep='STAligner', random_state=666)

sc.tl.louvain(adata_concat, random_state=666, key_added="louvain", resolution=0.4)

sc.tl.umap(adata_concat, random_state=666)

colors_default = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#aec7e8',
    '#ffbb78', '#98df8a', '#ff9896', '#bec1d4', '#bb7784', '#0000ff'
]

unique_clusters = np.sort(adata_concat.obs['louvain'].unique().astype(int))
adata_concat.uns['louvain_colors'] = [colors_default[i % len(colors_default)] for i in unique_clusters]

plt.rcParams['font.sans-serif'] = "DejaVu Sans"
plt.rcParams["figure.figsize"] = (2, 2)
plt.rcParams['font.size'] = 10

sc.pl.umap(adata_concat, color=['batch_name', 'louvain'], ncols=2, wspace=1, show=False)

plt.savefig("batch_louvain.png", dpi=300, bbox_inches='tight')

plt.show()

if isinstance(adata_concat.uns['edgeList'], tuple):
    adata_concat.uns['edgeList'] = np.array(adata_concat.uns['edgeList'])

output_path = "data/adata_concat_processed.h5ad"

adata_concat.write(output_path)

print(f"Processed adata saved at: {output_path}")

for ss in range(len(section_ids)):
    Batch_list[ss].obs['louvain'] = adata_concat[adata_concat.obs['batch_name'] == section_ids[ss]].obs['louvain'].values
    Batch_list[ss].uns['louvain_colors'] = [colors_default[0:][i] for i in np.sort(adata_concat[adata_concat.obs['batch_name'] == 
                                                                            section_ids[ss]].obs['louvain'].unique().astype('int'))]       

import matplotlib.pyplot as plt

spot_size = 0.01
title_size = 10

for adata in Batch_list:
    if 'spatial' not in adata.uns:
        adata.uns['spatial'] = {}

fig, ax = plt.subplots(4, 3, figsize=(12, 9), gridspec_kw={'wspace': 0.4, 'hspace': 0.3})    

for ss in range(len(section_ids)):
    row, col = divmod(ss, 3)  # ✅ 计算 (row, col) 位置
    _sc_0 = sc.pl.spatial(
        Batch_list[ss], img_key=None, color=['louvain'], title=['louvain'], 
        spot_size=spot_size, legend_fontsize=8, show=False, frameon=False, ax=ax[row, col]
    )
    _sc_0[0].set_title(section_ids[ss], size=title_size)

for i in range(len(section_ids), 12):  
    row, col = divmod(i, 3)
    fig.delaxes(ax[row, col])

plt.savefig("slice_figs.png", dpi=300, bbox_inches="tight")
plt.show()

