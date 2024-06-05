wget -P ./store/ https://github.com/dptech-corp/Uni-Core/releases/download/0.0.3/unicore-0.0.1+cu117torch2.0.0-cp38-cp38-linux_x86_64.whl
wget -P ./store/ https://data.pyg.org/whl/torch-2.0.0%2Bcu117/torch_scatter-2.1.1%2Bpt20cu117-cp38-cp38-linux_x86_64.whl
wget -P ./store/ https://data.pyg.org/whl/torch-2.0.0%2Bcu117/torch_sparse-0.6.17%2Bpt20cu117-cp38-cp38-linux_x86_64.whl
pip install ./store/unicore-0.0.1+cu117torch2.0.0-cp38-cp38-linux_x86_64.whl
pip install ./store/torch_scatter-2.1.1+pt20cu117-cp38-cp38-linux_x86_64.whl
pip install ./store/torch_sparse-0.6.17+pt20cu117-cp38-cp38-linux_x86_64.whl
pip install torch_geometric
pip install opt_einsum
pip install fair-esm
wget -P ./store/ https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
pip install freesasa