nvidia-smi
export MASTER_IP=$MLP_WORKER_0_HOST
export MASTER_PORT=$MLP_WORKER_0_PORT
export OMPI_COMM_WORLD_SIZE=$MLP_WORKER_NUM
export OMPI_COMM_WORLD_RANK=$MLP_ROLE_INDEX
apt update
apt install bc
mkdir data/
cp valid_similar_esm.lmdb data/valid_filter_alphaC_esm_cls.lmdb

export lr=1e-4
export total_steps=5000000
export warmup_steps=50000
export res2res_mask=0


export batch_size=2
export update_freq=4

export layers=12
export hidden_size=768
export ffn_size=768
export res2res_mask=15
export num_head=48
export num_layers=12
export node_dim=768
export knn=50
export edge_dim=128

export knn_res=15
export knn_atom2atom=15
export use_pairupdate=0
export cutoff=800

export res_pos_mask_prob=0.2
export res_noise_scale=0.5
export res_type_mask_prob=0.3

export ema_decay=0
export dihedral_pred=1
export pos_pred=1
export pocket_pred=0
export res_type_pred=1
export sas_pred=1
export local_all=0
export add_style=0
export preln=0

export learnable_pos=1
export use_esm_feat=1
export use_relative=1
export use_nerf_encoding=1
export esm_dim=1280
export span_mask=1
export residue_only=0

git --no-pager log -n 1
bash train_protein_pretrain.sh pt_new_data_all ../data/ ./store 2 