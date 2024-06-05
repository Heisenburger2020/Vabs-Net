[ -z "${MASTER_PORT}" ] && MASTER_PORT=10087
[ -z "${MASTER_IP}" ] && MASTER_IP=127.0.0.1
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=0

[ -z "${lr}" ] && lr=1e-5
[ -z "${end_lr}" ] && end_lr=1e-9
[ -z "${warmup_steps}" ] && warmup_steps=10000
[ -z "${total_steps}" ] && total_steps=100000
[ -z "${layers}" ] && layers=6
[ -z "${hidden_size}" ] && hidden_size=768
[ -z "${ffn_size}" ] && ffn_size=768
[ -z "${num_head}" ] && num_head=1
[ -z "${batch_size}" ] && batch_size=1 # G8 + export batch_size=8
[ -z "${update_freq}" ] && update_freq=1
[ -z "${seed}" ] && seed=1
[ -z "${data_seed}" ] && data_seed=$seed
[ -z "${clip_norm}" ] && clip_norm=5

[ -z "${dropout}" ] && dropout=0.0
[ -z "${act_dropout}" ] && act_dropout=0.1
[ -z "${attn_dropout}" ] && attn_dropout=0.1
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${droppath_prob}" ] && droppath_prob=0.1
[ -z "${num_3d_bias_kernel}" ] && num_3d_bias_kernel=128
[ -z "${num_block}" ] && num_block=1
[ -z "${gaussian_std_width}" ] && gaussian_std_width=1.0
[ -z "${gaussian_mean_start}" ] && gaussian_mean_start=0.0
[ -z "${guassian_mean_stop}" ] && guassian_mean_stop=9.0
[ -z "${pair_dropout}" ] && pair_dropout=0.25

[ -z "${ema_decay}" ] && ema_decay=0.999
[ -z "${preln}" ] && preln=0
[ -z "${notri}" ] && notri=1
[ -z "${usefl}" ] && usefl=1
[ -z "${fl_alpha}" ] && fl_alpha=0.25
[ -z "${fl_gamma}" ] && fl_gamma=4
[ -z "${gamma_pos}" ] && gamma_pos=0
[ -z "${gamma_neg}" ] && gamma_neg=4
[ -z "${pair_hidden_dim}" ] && pair_hidden_dim=16
[ -z "${pair_embed_dim}" ] && pair_embed_dim=64
[ -z "${angle_dim}" ] && angle_dim=34

[ -z "${batch_size_valid}" ] && batch_size_valid=4
[ -z "${validate_interval_updates}" ] && validate_interval_updates=1000
[ -z "${atom_base}" ] && atom_base=0
[ -z "${usemae}" ] && usemae=0

[ -z "${noise_scale}" ] && noise_scale=0.3
[ -z "${mask_prob}" ] && mask_prob=0.35

[ -z "${res2res_mask}" ] && res2res_mask=0
[ -z "${sample_atom}" ] && sample_atom=256

[ -z "${res_type_mask_prob}" ] && res_type_mask_prob=1
[ -z "${res_pos_mask_prob}" ] && res_pos_mask_prob=1
[ -z "${res_noise_scale}" ] && res_noise_scale=1

[ -z "${keep}" ] && keep=37
[ -z "${crop}" ] && crop=10
[ -z "${node_dim}" ] && node_dim=128
[ -z "${num_layers}" ] && num_layers=5
[ -z "${knn}" ] && knn=30
[ -z "${edge_dim}" ] && edge_dim=$node_dim
[ -z "${outer_product_dim}" ] && outer_product_dim=32
[ -z "${pair_update}" ] && pair_update=0


[ -z "${use_outer}" ] && use_outer=0
[ -z "${use_trimul}" ] && use_trimul=0
[ -z "${use_pairupdate}" ] && use_pairupdate=0
[ -z "${use_triangleattn}" ] && use_triangleattn=0
[ -z "${use_ipa_norm}" ] && use_ipa_norm=0
[ -z "${use_ipa}" ] && use_ipa=0
[ -z "${use_vec}" ] && use_vec=1
[ -z "${use_sas}" ] && use_sas=0
[ -z "${use_pos}" ] && use_pos=0
[ -z "${use_pointgnn}" ] && use_pointgnn=0
[ -z "${use_virtual_node}" ] && use_virtual_node=1
[ -z "${use_maskinter}" ] && use_maskinter=0
[ -z "${use_res2atom}" ] && use_res2atom=0
[ -z "${knn_res}" ] && knn_res=$knn
[ -z "${knn_atom2res}" ] && knn_atom2res=$knn
[ -z "${cutoff}" ] && cutoff=600
[ -z "${knn_atom2atom}" ] && knn_atom2atom=$knn
[ -z "${knn_res2atom}" ] && knn_res2atom=1
[ -z "${use_af}" ] && use_af=1
[ -z "${use_pdb}" ] && use_pdb=0
[ -z "${use_graphv2}" ] && use_graphv2=0
[ -z "${use_torsion}" ] && use_torsion=0

[ -z "${dihedral_pred}" ] && dihedral_pred=1
[ -z "${pos_pred}" ] && pos_pred=1
[ -z "${pocket_pred}" ] && pocket_pred=0
[ -z "${res_type_pred}" ] && res_type_pred=1
[ -z "${if_ratio}" ] && if_ratio=0.5
[ -z "${refine}" ] && refine=0
[ -z "${use_largest}" ] && use_largest=0
[ -z "${use_esm_feat}" ] && use_esm_feat=1
[ -z "${sas_pred}" ] && sas_pred=1
[ -z "${local_all}" ] && local_all=0
[ -z "${concat_style}" ] && concat_style=0
[ -z "${learnable_pos}" ] && learnable_pos=1
[ -z "${add_style}" ] && add_style=0
[ -z "${use_absolute}" ] && use_absolute=1
[ -z "${use_relative}" ] && use_relative=1
[ -z "${use_nerf_encoding}" ] && use_nerf_encoding=1
[ -z "${esm_dim}" ] && esm_dim=1280
[ -z "${mask_side_chain}" ] && mask_side_chain=0
[ -z "${residue_only_ratio}" ] && residue_only_ratio=0
[ -z "${span_mask}" ] && span_mask=1
[ -z "${atom_only}" ] && atom_only=0
[ -z "${residue_only}" ] && residue_only=1
[ -z "${v_attn}" ] && v_attn=0


# MP
echo -e "\n\n\n\n"
echo "==================================MP==========================================="

name=$1
# hyperparams=$name-lr$lr-bsz$((batch_size*n_gpu*OMPI_COMM_WORLD_SIZE*update_freq))-preln$preln-wp$warmup_steps-ts$total_steps-pairh$pair_hidden_dim-paire$pair_embed_dim-crop$crop-notri$notri
hyperparams=$name-prob$res_pos_mask_prob-scale$res_noise_scale-op$use_outer-pair$use_pairupdate-layers$num_layers-bsz$batch_size-update$update_freq-lr$lr-node_dim$node_dim-bsz$((batch_size*n_gpu*OMPI_COMM_WORLD_SIZE*update_freq))
save_dir=$3/$hyperparams

mkdir -p $save_dir

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "seed: ${seed}"
echo "data_seed: ${data_seed}"
echo "batch_size: ${batch_size}"
echo "n_layers: ${layers}"
echo "update_freq: ${update_freq}"
echo "lr: ${lr}"
echo "warmup_steps: ${warmup_steps}"
echo "total_steps: ${total_steps}"
echo "clip_norm: ${clip_norm}"
echo "BLOCKS: ${BLOCKS}"
echo "node_loss_weight: ${node_loss_weight}"
echo "save_dir: ${save_dir}"
echo "tsb_dir: ${tsb_dir}"
echo "pair_hidden_dim: ${pair_hidden_dim}"
echo "usefl: ${usefl}"
echo "fl_alpha: ${fl_alpha}"
echo "fl_gamma: ${fl_gamma}"
echo "==============================================================================="

more_args=""
if (( $(echo "$ema_decay > 0.0" | bc -l) )); then
more_args=$more_args" --ema-decay $ema_decay --validate-with-ema"
# save_dir=$save_dir"-ema"$ema_decay
fi

echo more_args 
echo $more_args

data_path=$2
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
run_name=$( echo ${hyperparams##*/} )
echo $run_name
torchrun --nproc_per_node=$n_gpu --master_port=$MASTER_PORT --nnodes=$OMPI_COMM_WORLD_SIZE --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_IP  \
       $(which unicore-train) $data_path --user-dir ./vabs --train-subset train --valid-subset valid --best-checkpoint-metric loss \
       --num-workers 8 --ddp-backend=c10d \
       --task protein_pretrain --loss protein_pretrain --arch protein_pocket_base  \
       --tensorboard-logdir $save_dir/tsb \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --log-interval 10 --log-format simple \
       --save-interval-updates 4000 --validate-interval-updates 4000 --keep-interval-updates 1 --no-epoch-checkpoints  \
       --save-dir $save_dir --batch-size $batch_size  \
       --data-buffer-size 16 --fixed-validation-seed 11 --batch-size-valid $4 \
       --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm $clip_norm \
       --lr $lr --end-learning-rate $end_lr --lr-scheduler polynomial_decay --power 1 \
       --warmup-updates $warmup_steps --total-num-update $total_steps --max-update $total_steps --update-freq $update_freq \
       --encoder-layers $layers --encoder-attention-heads $num_head --num-3d-bias-kernel $num_3d_bias_kernel \
       --encoder-embed-dim $hidden_size --encoder-ffn-embed-dim $ffn_size --droppath-prob $droppath_prob \
       --pair-embed-dim $pair_embed_dim --pair-hidden-dim $pair_hidden_dim --pair-dropout $pair_dropout \
       --attention-dropout $attn_dropout --act-dropout $act_dropout --dropout $dropout --weight-decay $weight_decay \
       --num-block $num_block \
       --gaussian-std-width $gaussian_std_width --gaussian-mean-start $gaussian_mean_start --gaussian-mean-stop $guassian_mean_stop \
       --seed $seed $more_args \
       --tmp-save-dir ../tmp/$hyperparams \
       --required-batch-size-multiple 1 \
       --preln $preln \
       --notri $notri \
       --noise-scale $noise_scale --mask-prob $mask_prob \
       --res2res-mask $res2res_mask \
       --sample-atom $sample_atom \
       --res-type-mask-prob $res_type_mask_prob \
       --res-pos-mask-prob $res_pos_mask_prob \
       --res-noise-scale $res_noise_scale \
       --node-dim $node_dim \
       --edge-dim $edge_dim \
       --outer-product-dim $outer_product_dim \
       --knn $knn \
       --num-layers $num_layers \
       --pair-update $pair_update \
       --use-outer $use_outer \
       --use-trimul $use_trimul \
       --use-pairupdate $use_pairupdate \
       --use-triangleattn $use_triangleattn \
       --use-ipa-norm $use_ipa_norm \
       --use-vec $use_vec \
       --use-pos $use_pos \
       --cutoff $cutoff \
       --knn-atom2atom $knn_atom2atom \
       --knn-res $knn_res \
       --use-virtual-node $use_virtual_node \
       --pos-pred $pos_pred \
       --dihedral-pred $dihedral_pred \
       --pocket-pred $pocket_pred \
       --res-type-pred $res_type_pred \
       --sas-pred $sas_pred \
       --use-torsion $use_torsion \
       --local-all $local_all \
       --concat-style $concat_style \
       --use-esm-feat $use_esm_feat \
       --learnable-pos $learnable_pos \
       --add-style $add_style \
       --use-absolute $use_absolute \
       --use-relative $use_relative \
       --esm-dim $esm_dim \
       --mask-side-chain $mask_side_chain \
       --use-nerf-encoding $use_nerf_encoding \
       --residue-only-ratio $residue_only_ratio \
       --span-mask $span_mask \
       --residue-only $residue_only \
       --v-attn $v_attn \
       --restore-file $5 \
