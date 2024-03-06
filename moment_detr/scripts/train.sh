dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
#v_feat_types=clip
t_feat_type=clip
results_root=results
exp_id=moment_detr_cross_enc_conv_pred_head_sal_span_triplet

######## data paths
train_path=data/highlight_train_release.jsonl
#train_path=data/highlight_train_release_paraphrased.jsonl
eval_path=data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../QVHighlights/features

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text_features/
#  t_feat_dir=${feat_root}/clip_aug_text_features/
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

#### training
bsz=32
n_epoch=200

PYTHONPATH=$PYTHONPATH:. python moment_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--n_epoch ${n_epoch} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--device 0 \
--num_workers 4 \
--hidden_dim 256 \
${@:1}
