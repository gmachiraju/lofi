source /opt/conda/etc/profile.d/conda.sh
conda activate /home/envs

# ================= MODIFY =========================
# current models for backbone: VGG19_bn, VGG_att, ViT
model_class=ResNet18
ne=5 # usually like 1-10
overfit_flag=False

# model1="/home/codex_analysis/codex-analysis/models/cam/VGG19_bn-hdf5_random_loading-224-label_inherit-bce_loss-on_cam-cam16-filtration_background_epoch7.pt"
# model1="/home/codex_analysis/codex-analysis/models/cam/ResNet18-hdf5_triplets_random_loading-224-label_selfsup-custom_loss-on_cam-cam16-filtration_background_epoch4.pt"
# model1="/home/codex_analysis/codex-analysis/models/cam/CARTA-ResNet18-hdf5_triplets_random_loading-224-label_selfsup-custom_loss-on_cam-cam16-filtration_background.sd"
# model1="/home/codex_analysis/codex-analysis/models/cam/ResNet18-hdf5_triplets_random_loading-224-label_selfsup-custom_loss-on_cam-cam16-filtration_background_epoch13.sd"
# model1="/home/lofi/lofi/models/cam/ViT-hdf5_random_loading-224-label_inherit-bce_loss-on_cam-cam16-filtration_background.sd"

selfsup_mode=mix        # mix for regular self-sup  // sextuplet for carta
coaxial_flag=False      # False for regular self-sup // True for carta
bs=2                    # 15-20 for regular self-sup // 5 for carta // 36 for ViT


# gamified learning params
#-------------------------
save_embeds_flag=False
gamified_flag=False
backprop_level=none
# options are {none, blindfolded, full} --> none is the same as regularization term
pool_type=max # mean not yet efficiently implemented

# self-sup params
#----------------
selfsup_flag=False
trip0_path="/home/data/swedish_traffic_signs/swedish_streets/train0_idlist.obj"
trip1_path="/home/data/swedish_traffic_signs/swedish_streets/train1_idlist.obj" 

# dataset params
#----------------
dn=sts
cd=3
scenario=speed
ps=320
dp="/home/data/swedish_traffic_signs/swedish_streets/" 
# ===============================================

filtration=background #none OR background; background for MISO paper
plab=inherit
dlt=hdf5

hp=0.01
nf=False
pload=block
ploss=bce

cp=/home/cache/sts
ldp="/home/data/swedish_traffic_signs/swedish_streets/label_dict.obj"
plp="/home/data/swedish_traffic_signs/swedish_streets/allfiles_idlist.obj"
mp=/home/lofi/lofi/models/sts
desc=${model_class}"-"${dlt}"_"${pload}"_loading-"${ps}"-label_"${plab}"-"${ploss}"_loss-on_"${dn}"-"${scenario}"-filtration_"${filtration}
script=/home/lofi/lofi/src/train.py

# --model_to_load ${model1}
python ${script} --coaxial_flag ${coaxial_flag} --overfit_flag ${overfit_flag} --selfsup_flag ${selfsup_flag} --trip0_path ${trip0_path} --trip1_path ${trip1_path} --selfsup_mode ${selfsup_mode}  --description ${desc} --model_class ${model_class} --num_epochs ${ne} --hyperparameters ${hp} --batch_size ${bs} --channel_dim ${cd} --normalize_flag ${nf} --dataset_name ${dn} --dataloader_type ${dlt} --patch_size ${ps} --patch_loading ${pload} --patch_labeling ${plab} --patch_loss ${ploss} --data_path ${dp} --patchlist_path ${plp} --labeldict_path ${ldp} --model_path ${mp} --cache_path ${cp} --save_embeds_flag ${save_embeds_flag} --gamified_flag ${gamified_flag} --backprop_level ${backprop_level} --pool_type ${pool_type}
