export lr=1e-5
export total_steps=100000
export warmup_steps=10000
export res2res_mask=0
export batch_size=1
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
export knn_res=30
export knn_atom2atom=30
export use_pairupdate=0
export cutoff=800
export res_pos_mask_prob=0.2
export res_noise_scale=0.5
export res_type_mask_prob=0.2
export ema_decay=0
export dihedral_pred=0
export pos_pred=0
export pocket_pred=1
export res_type_pred=0
export use_sas=0

export local_all=0

export learnable_pos=1
export use_esm_feat=1
export use_relative=1
export use_absolute=1
export use_nerf_encoding=1
export esm_dim=1280
export residue_only=0


git --no-pager log -n 1
bash train_protein_pocket_finetune.sh ft_poc_sm /mnt/vepfs/fs_ckps/zhaojiale/dataset/sm_dataset_official ./store 2 ./store/vabs_ckpt.pt