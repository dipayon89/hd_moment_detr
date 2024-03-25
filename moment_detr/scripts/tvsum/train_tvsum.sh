dset_name=tvsum
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=results_tvsum
exp_id=exp-qddec


######## data paths
train_path=data/tvsum/tvsum_train.jsonl
eval_path=data/tvsum/tvsum_val.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../Datasets/processed/tvsum

# # video features
v_feat_dim=2816
v_feat_dirs=()
v_feat_dirs+=(${feat_root}/video_features/)
v_feat_dirs+=(${feat_root}/video_features_blip/)

# # text features
t_feat_dirs=()
t_feat_dirs+=(${feat_root}/query_features/)
t_feat_dirs+=(${feat_root}/query_features_blip/)
t_feat_dim=1280

#### training
bsz=4
lr=1e-3

######## TVSUM domain name
for dset_domain in BK BT DS FM GA MS PK PR VT VU
do
    for seed in 0 1 2 3 2017
    do
        PYTHONPATH=$PYTHONPATH:. python moment_detr/train.py \
        --dset_name ${dset_name} \
        --ctx_mode ${ctx_mode} \
        --train_path ${train_path} \
        --eval_path ${eval_path} \
        --eval_split_name ${eval_split_name} \
        --v_feat_dirs ${v_feat_dirs[@]} \
        --v_feat_dim ${v_feat_dim} \
        --t_feat_dirs ${t_feat_dirs} \
        --t_feat_dim ${t_feat_dim} \
        --bsz ${bsz} \
        --results_root ${results_root}/${dset_domain} \
        --exp_id ${exp_id} \
        --max_v_l 1000 \
        --n_epoch 1500 \
        --lr_drop 2000 \
        --max_es_cnt -1 \
        --seed $seed \
        --lr ${lr} \
        --dset_domain ${dset_domain} \
        ${@:1}
    done
done
