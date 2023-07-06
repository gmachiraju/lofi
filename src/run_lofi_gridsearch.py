import utils
from embed_patches import class_tfidf
from embed_patches import diff_exp
from embed_patches import generate_model_outputs
from embed_patches import eval_classifier
from embed_patches import run_lofi_gridsearch

def run(modelstr):
    if modelstr == "tile2vec":
        Zs_path_train = "/home/data/tinycam/train/Zs"
        Zs_path_test = "/home/data/tinycam/test/clean_Zs"
        embed_path = "/home/lofi/lofi/src/outputs_tile2vec/train_sampled_inference_z_embeds.obj"
        crop_dict = utils.deserialize("/home/lofi/lofi/src/outputs_tile2vec/test_crop_coords.obj")
    elif modelstr == "vit_iid":
        Zs_path_train = "/home/data/tinycam/train/Zs_vit"
        Zs_path_test = "/home/data/tinycam/test/clean_Zs_vit"
        embed_path = "/home/lofi/lofi/src/outputs_vit_iid/train_vit_iid_sampled_inference_z_embeds.obj"
        mapping_path = "/home/lofi/lofi/src/outputs_vit_iid/train_vit_iid_chunkid_position.obj" # needed to use memmap
        crop_dict = utils.deserialize("/home/lofi/lofi/src/outputs_vit_iid/test_vit_iid_crop_coords.obj")
    elif modelstr == "clip":
        Zs_path_train = "/home/data/tinycam/train/Zs_clip"
        Zs_path_test = "/home/data/tinycam/test/clean_Zs_clip"
        embed_path = "/home/lofi/lofi/src/outputs_clip/train_clip_sampled_inference_z_embeds.obj"
        crop_dict = utils.deserialize("/home/lofi/lofi/src/outputs_clip/test_clip_crop_coords.obj")
    elif modelstr == "plip":
        Zs_path_train = "/home/data/tinycam/train/Zs_plip"
        Zs_path_test = "/home/data/tinycam/test/clean_Zs_plip"
        embed_path = "/home/lofi/lofi/src/outputs_plip/train_plip_sampled_inference_z_embeds.obj"
        crop_dict = utils.deserialize("/home/lofi/lofi/src/outputs_plip/test_plip_crop_coords.obj")

    label_dict_path_train = "/home/lofi/lofi/src/outputs/train-cam-cam16-224-background-labeldict.obj"
    label_dict_path_test = "/home/lofi/lofi/src/outputs/test-cam-cam16-224-background-labeldict.obj"

    gts_path = "/home/data/tinycam/test/gt_masks"
    csv_save_path = "/home/data/tinycam/test/csv_outputs"
    label_dict_train = utils.deserialize(label_dict_path_train)
    label_dict_test = utils.deserialize(label_dict_path_test)

    # run
    if modelstr == "vit_iid":
        run_lofi_gridsearch(modelstr, embed_path, "cpu", Zs_path_train, Zs_path_test, label_dict_path_train, label_dict_path_test, crop_dict, csv_save_path, gts_path, mode="memmap", mapping_path=mapping_path)
    else:
        run_lofi_gridsearch(modelstr, embed_path, "cpu", Zs_path_train, Zs_path_test, label_dict_path_train, label_dict_path_test, crop_dict, csv_save_path, gts_path)

def main():
    modelstr = "plip"
    run(modelstr)

if __name__ == "__main__":
	main()

