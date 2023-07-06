import os
import numpy as np
import pandas as pd
import glob
import torch
import torch.nn as nn
import h5py
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
# import clip

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
# import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from skimage.filters import threshold_otsu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import roc_auc_score

# import shap
from sklearn.ensemble import GradientBoostingClassifier
import scipy

# import gradcam_clip
import utils
from utils import serialize, deserialize
from dataloader import EmbedDataset, reduce_Z
from sod_utils import MeanAveragePrecision

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from dataloader import baggify, colocalization
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter
from vit_pytorch.recorder import Recorder
from vit_pytorch.extractor import Extractor
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn.functional as F

from tempfile import mkdtemp
# import os.path as path
threshs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

from lofi_utils import *


"""
X: raw data, images or time-series
x: tokens from X
z: token embedding (after inference on x)
Z: embedded data using concatenation of embedded tokens (z)
"""

def print_X_names(label_dict_path, arm="train"):
    """
    get the IDs of input data Xs
    """
    keys = []
    label_dict = utils.deserialize(label_dict_path)
    for k in label_dict.keys():
        if arm == "train":
            keys.append(k)
        elif arm == "val" and "node" in k:
            keys.append(k.split(".")[0])
    return keys


def parse_x_coords(x_name, arm="train"):
    """
    get the coordinates of a particular chunk, x
    """
    pieces = x_name.split("_")
    if arm == "train" or arm == "test":
        im_id = pieces[0] + "_" + pieces[1]
        pos = pieces[3]
    elif arm == "val":
        im_id = pieces[0] + "_" + pieces[1] + "_" + pieces[2] + "_" + pieces[3]
        pos = pieces[5]

    ij = pos.split("-")
    i = ij[0].split("coords")[1]
    j = ij[1]
    return im_id, int(i), int(j)


def gather_Z_dims(patch_dir, X_names, arm="train"):
    """
    Collect dimensions of the new embedded data (Zs) 
    -scans through patch/chunk indices to determine this
    """
    hf = h5py.File(patch_dir, 'r')
    files = list(hf.keys())
    print("Gathering dimensions...")
    dim_dict = {}
    for im in X_names:
        dim_dict[im] = [0,0]

    for idx,f in enumerate(files):
        im_id, i, j = parse_x_coords(f, arm)
        if i > dim_dict[im_id][0]:
            dim_dict[im_id][0] = i
        if j > dim_dict[im_id][1]:
            dim_dict[im_id][1] = j

    print("done!")
    return dim_dict


def inference_z(model_path, patch_dir, scope="all", cpu=True):
    """
    use trained model to embed chunks
    """
    # enable device
    print("GPU detected?", torch.cuda.is_available())
    if (cpu == False) and torch.cuda.is_available():
        device = torch.device('cuda')
        print("\nNote: gpu available & selected!")
    else:
        device = torch.device('cpu')
        print("\nNote: gpu NOT available!")

    # load data
    hf = h5py.File(patch_dir, 'r')
    files = list(hf.keys())
    print("We have", len(files), "unique patches to embed")
    print("loading model for inference...")
    model = torch.load(model_path, map_location=device)
    model.eval()
    embed_dict = {}

    scope_flag = False
    if isinstance(scope, list):
        print("Specifically embedding for data:", scope)
        scope_flag = True
    with torch.no_grad():
        x_batch, x_names = [], []
        for idx,f in enumerate(files):
            if scope_flag:
                im_id, _, _ = parse_x_coords(f)
                if im_id not in scope:
                    continue
                else:
                    # print("Found patch for", im_id)
                    x = hf[f][()]
                    x_names.append(f)
                    x_batch.append(x)
                    # print("current batch size:", len(x_batch))
            if len(x_batch) == 3:
                # print("we now have a batch!")
                x_batch = np.stack(x_batch, axis=0)
                # print(x_batch.shape)
                x_batch = torch.from_numpy(x_batch)
                if cpu == False:
                    x_batch = x_batch.cuda()
                x_batch = x_batch.to(device=device, dtype=torch.float)
                z_batch = model.encode(x_batch)
                for b,name in enumerate(x_names):
                    embed_dict[name] = z_batch[b,:].cpu().detach().numpy()
                x_batch, x_names = [], [] # reset
            if (idx+1) % 10000 == 0:
                print("finished inference for", (idx+1), "patches")
    
    utils.serialize(embed_dict, "inference_z_embeds.obj")
    print("serialized numpy embeds at: inference_z_embeds.obj")
    return 


def visualize_z(embed_dict_path, dim_dict, scope="all", K=8, mode="dict", mapping_path=None):
    """
    visualizes embedding space of individual chunk embeddings
    """
    if scope == "all":
        print("not yet implemented random sampling of all embeds~")
        exit()
    if isinstance(scope, list) and len(scope) > 1:
        print("not yet implemented scope of more than one X") 
        exit()

    embeds_list = []
    sources = []
    x_locs = []
    if mode == "dict":
        embed_dict = utils.deserialize(embed_dict_path)
        for k in embed_dict.keys():
            v = embed_dict[k]
            im_id, i, j = parse_x_coords(k)
            embeds_list.append(v)
            sources.append(im_id)
            x_locs.append([i,j])
        array = np.vstack(embeds_list)
        print("total embeds:", array.shape)
    elif mode == "memmap":
        mapping_dict = utils.deserialize(mapping_path)
        for k in mapping_dict.keys():
            v = mapping_dict[k]
            im_id, i, j = parse_x_coords(k)
            sources.append(im_id)
            x_locs.append([i,j])
        embed_dict = np.memmap(embed_dict_path, dtype='float32', mode='r', shape=(4440,1024)) # not an actual dict as you can see
        array = embed_dict

    # tsne - color by source
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30).fit_transform(array)
    print(X_embedded.shape)
    plt.figure()
    plt.xlabel("tSNE-0")
    plt.ylabel("tSNE-1")
    plt.scatter(X_embedded[:,0], X_embedded[:,1], s=2, alpha=0.1, cmap="Dark2")

    kmeans = KMeans(n_clusters=K, random_state=0).fit(array)
    cluster_labs = kmeans.labels_
    plt.figure()
    plt.xlabel("tSNE-0")
    plt.ylabel("tSNE-1")
    plt.scatter(X_embedded[:,0], X_embedded[:,1], s=2, c=cluster_labs, alpha=0.1, cmap="Dark2")

    plt.figure()
    plt.hist(cluster_labs)

    # plot clusters for image
    zero_id = np.max(cluster_labs) + 1
    our_id = sources[0]
    Z_dim = dim_dict[our_id]
    print("Z is of size:", Z_dim)
    Z = np.zeros((Z_dim[0]+1, Z_dim[1]+1)) + zero_id
    for coord_id,coord in enumerate(x_locs):
        i,j = coord[0], coord[1]
        Z[i,j] = cluster_labs[coord_id]
    
    plt.figure(figsize=(12, 6), dpi=80)
    plt.imshow(Z, vmin=0, vmax=zero_id, cmap="Dark2")
    plt.show()
    print(Z)


def construct_Z(embed_dict_path, X_id, Z_dim):
    """
    spatially orient embeddings to construct embedded data, Z
    """
    embed_dict = utils.deserialize(embed_dict_path)
    embeds_id = {}
    x_locs = []
    for k in embed_dict.keys():
        im_id, i, j = parse_x_coords(k)
        if im_id == X_id:
            v = embed_dict[k]
            embeds_id[(i,j)] = v
            x_locs.append([i,j])

    Z = np.zeros((Z_dim[0]+1, Z_dim[1]+1))
    for coord_id,coord in enumerate(x_locs):
        i,j = coord[0], coord[1]
        Z[i,j] = embeds_id[(i,j)]
    return Z


def pad_Z(Z, desired_dim=124):
    """
    Z: single embedded datum
    """
    h,w,d = Z.shape
    canvas = np.zeros((desired_dim, desired_dim, d))
    i_start = (desired_dim - h) // 2
    j_start = (desired_dim - w) // 2
    canvas[i_start:i_start+h, j_start:j_start+w, :] = Z
    coords = [(i_start, i_start+h), (j_start, j_start+w)]
    print("padding C (" + str(h) + "x" + str(w) + ") with H,W crops at:", coords)
    return canvas, coords


def construct_Zs(embed_dict_path, dim_dict, save_dir, scope="all"):
    """
    Reads embeds dict, gathers by image ID, and then creates tensor; saves in a save_dir 
    """
    crop_dict = {}
    print("Constructing Z tensors for", scope)
    for X_id in dim_dict.keys():
        print("On ID:", X_id)
        Z_dim = dim_dict[X_id]
        Z = construct_Z(embed_dict_path, X_id, Z_dim)
        Z, crop_coords = pad_Z(Z)
        np.save(Z, save_dir + "/Z-" + im_id + ".npy")
        crop_dict[X_id] = crop_coords
    print("Done!")
    print("saved Z tensors at:", save_dir)


#-------------------------------------------
def construct_Z_efficient(X_id, Z_dim, files, hf, model, device, sample_size=20, d=128, arm="train"):
    num_files = len(files)

    # grab triplets and do inference to get embeddings
    with torch.no_grad():
        x_locs_all = []
        x_batch, x_locs = [], []
        files_remaining = []
        # initialize Z array
        Z = np.zeros((Z_dim[0]+1, Z_dim[1]+1, d))
        save_backup = False

        # iterate through
        for idx, f in enumerate(files):
            im_id, i, j = parse_x_coords(f, arm)
            if im_id != X_id:
                if "noshift" in f:
                    files_remaining.append(f)
                continue
            else: # match with X_id
                # skipping 50shift for now to keep dimensionality low
                if "noshift" in f: 
                    # print("hit:", f)
                    x = hf[f][()]
                    # black background
                    summed = np.sum(x, axis=0)
                    if np.abs(np.mean(summed) - 0.0) < 100 and np.std(summed) < 10:
                        # print("black bg tile!")
                        continue 
                    if np.abs(np.mean(summed) - 765) < 100 and np.std(summed) < 10:
                        # print("white bg tile!")
                        continue 
                    x_locs.append((i,j))
                    x_locs_all.append((i,j))
                    x_batch.append(x)

            # Need to check for any straggler patches at end that may not be a full batch
            # Then we grab the first couple batch entries
            if idx == num_files and len(x_batch) < 3: 
                num_needed = 3 - len(x_batch)
                x_batch.extend([x_backup[i] for i in range(num_needed)])

            if len(x_batch) == 3:
                if save_backup == False:
                    x_backup = x_batch.copy() # for any stragglers
                    save_backup = True
                x_batch = np.stack(x_batch, axis=0)
                x_batch = torch.from_numpy(x_batch)
                x_batch = x_batch.to(device=device, dtype=torch.float)
                z_batch = model.encode(x_batch)
                for b,loc in enumerate(x_locs):
                    i, j = loc[0], loc[1]
                    # print(b,i,j)
                    try:
                        Z[i,j,:] = z_batch[b,:].cpu().detach().numpy()
                    except IndexError:
                        print(i, "or", j, "not in index range for Z:", Z.shape)
                        continue
                x_batch, x_locs = [], [] # reset

        # grab sample size emebds from id list
        sample_z_dict = {}
        sample_locs = np.random.choice(len(x_locs_all), size=sample_size, replace=False)
        sample_coords = [x_locs_all[sl] for sl in sample_locs]
        for sc in sample_coords:
            try:
                sample_z_dict[X_id+"_"+str(sc[0])+"_"+str(sc[1])] = Z[sc[0],sc[1],:]
            except IndexError:
                continue

    return Z, files_remaining, sample_z_dict


# Some functionality provided by library: 
# https://github.com/lucidrains/vit-pytorch#accessing-attention
def inference_transformer(X_id, Z_dim, files, hf, model, processor, tokenizer, device, sample_size=20, d=1024, arm="train", bs=3):
    num_files = len(files)
    model.to(device)
    model.eval()

    # grab triplets and do inference to get embeddings
    with torch.no_grad():
        x_locs_all = []
        x_batch, x_locs = [], []
        files_remaining = []

        # initialize Z array
        P = np.zeros((Z_dim[0]+1, Z_dim[1]+1, 1))
        G = np.zeros((Z_dim[0]+1, Z_dim[1]+1, 1)) # gradcam
        A = np.zeros((Z_dim[0]+1, Z_dim[1]+1, 1)) # attention
        Z = np.zeros((Z_dim[0]+1, Z_dim[1]+1, d))
        save_backup = False
        if device == "cpu":
            use_cuda = True
        else:
            use_cuda = False

        # gradcam setup
        # named_layers = dict(model.named_modules()) 
        # print(named_layers) 
        # pdb.set_trace() 
        # https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/extractor.py
        # target_layers = [model.transformer.layers[5][1].norm]  # .5.1.norm   
        # cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)

        # iterate through
        for idx, f in enumerate(files):
            im_id, i, j = parse_x_coords(f, arm)
            if im_id != X_id:
                if "noshift" in f:
                    files_remaining.append(f)
                continue
            else: # match with X_id
                # skipping 50shift for now to keep dimensionality low
                if "noshift" in f: 
                    x = hf[f][()]
                    summed = np.sum(x, axis=0)
                    if np.abs(np.mean(summed) - 0.0) < 100 and np.std(summed) < 10:
                        continue 
                    if np.abs(np.mean(summed) - 765) < 100 and np.std(summed) < 10:
                        continue 
                    x_locs.append((i,j))
                    x_locs_all.append((i,j))
                    x_batch.append(x)

            # Need to check for any straggler patches at end that may not be a full batch
            # Then we grab the first couple batch entries
            if idx == num_files and len(x_batch) < bs: 
                num_needed = bs - len(x_batch)
                x_batch.extend([x_backup[i] for i in range(num_needed)])

            if len(x_batch) == bs:
                if save_backup == False:
                    x_backup = x_batch.copy() # for any stragglers
                    save_backup = True
                x_batch = np.stack(x_batch, axis=0)
                x_batch = torch.from_numpy(x_batch)
                x_batch = x_batch.to(device=device, dtype=torch.float)

                # get embeddings
                #---------------
                model = Extractor(model)
                logits, z_batch = model(x_batch)
                # z_batch: (1, 65, 1024) - (batch x patches x model dim)
                z_batch = torch.mean(z_batch, dim=1) # take mean of all patch embeddings
                model = model.eject()  # wrapper is discarded and original ViT instance is returned

                if arm == "test":
                    # get probabilities and attention
                    #---------------------------------
                    model = Recorder(model)
                    outs, attn = model(x_batch)
                    # a_batch: (1, 6, 16, 65, 65) - (batch x layers x heads x patch x patch)
                    model = model.eject()  # wrapper is discarded and original ViT instance is returned

                    # gradcam
                    #---------
                    # with torch.enable_grad():
                    #     targets = [ClassifierOutputTarget(1)]
                    #     x_batch.requires_grad_()
                    #     gradcam = cam(input_tensor=x_batch, targets=targets)

                for b,loc in enumerate(x_locs):
                    i, j = loc[0], loc[1]
                    try:
                        Z[i,j,:] = z_batch[b,:].cpu().detach().numpy()
                        if arm == "test":
                            probs = F.softmax(outs, dim=1)
                            P[i,j,:] = probs[b,1].cpu().detach().numpy() # class 1
                            A[i,j,:] = np.mean(attn[b,:].cpu().detach().numpy())
                            
                            # G[i,j,:] = np.mean(gradcam[b,:].cpu().detach().numpy())
                    except IndexError:
                        print(i, "or", j, "not in index range for Z:", Z.shape)
                        continue
                x_batch, x_locs = [], [] # reset

        # grab sample size emebds from id list
        sample_z_dict = {}
        sample_locs = np.random.choice(len(x_locs_all), size=sample_size, replace=False)
        sample_coords = [x_locs_all[sl] for sl in sample_locs]
        for sc in sample_coords:
            try:
                sample_z_dict[X_id+"_"+str(sc[0])+"_"+str(sc[1])] = Z[sc[0],sc[1],:]
            except IndexError:
                continue

    return Z, P, A, G, files_remaining, sample_z_dict 


def inference_clip(X_id, Z_dim, files, hf, model, processor, tokenizer, device, sample_size=20, d=512, arm="train", text_to_encode=["normal lymph node", "lymph node metastasis"], bs=10):
    num_files = len(files)
    model.to(device)

    # grab triplets and do inference to get embeddings
    with torch.no_grad():
        x_locs_all = []
        x_batch, x_locs = [], []
        files_remaining = []

        # initialize Z array
        P = np.zeros((Z_dim[0]+1, Z_dim[1]+1, 1))
        E = np.zeros((Z_dim[0]+1, Z_dim[1]+1, 1))
        Z = np.zeros((Z_dim[0]+1, Z_dim[1]+1, d))
        save_backup = False

        # iterate through
        for idx, f in enumerate(files):
            im_id, i, j = parse_x_coords(f, arm)
            if im_id != X_id:
                if "noshift" in f:
                    files_remaining.append(f)
                continue
            else: # match with X_id
                # skipping 50shift for now to keep dimensionality low
                if "noshift" in f: 
                    # print("hit:", f)
                    x = hf[f][()]
                    # black background
                    summed = np.sum(x, axis=0)
                    if np.abs(np.mean(summed) - 0.0) < 100 and np.std(summed) < 10:
                        # print("black bg tile!")
                        continue 
                    if np.abs(np.mean(summed) - 765) < 100 and np.std(summed) < 10:
                        # print("white bg tile!")
                        continue 
                    x_locs.append((i,j))
                    x_locs_all.append((i,j))
                    x_batch.append(x)

            # Need to check for any straggler patches at end that may not be a full batch
            # Then we grab the first couple batch entries
            if idx == num_files and len(x_batch) < bs: 
                num_needed = bs - len(x_batch)
                x_batch.extend([x_backup[i] for i in range(num_needed)])

            if len(x_batch) == bs:
                if save_backup == False:
                    x_backup = x_batch.copy() # for any stragglers
                    save_backup = True
                x_batch = np.stack(x_batch, axis=0)
                x_batch = torch.from_numpy(x_batch)
                x_batch = x_batch.to(device=device, dtype=torch.float)

                img_inputs = processor(images=x_batch, return_tensors="pt", padding=True).to(device)
                z_batch = model.get_image_features(**img_inputs)
                
                if arm == "test":
                    inputs = processor(text=text_to_encode, images=x_batch, return_tensors="pt", padding=True).to(device)
                    outputs = model(**inputs)

                    #probabilities
                    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
                    probs = logits_per_image.softmax(dim=1) 

                    # gradcam
                    # text_input = clip.tokenize(["cancer"]).to(device)
                    # text_input = processor(text=text_to_encode[1], return_tensors="pt", padding=True).to(device)
                    # model.text_model(text_to_encode[1])
                    # model.text_model(text_input["input_ids"].to(device))
                    # pdb.set_trace()
                    # with torch.enable_grad():
                    #     text_forward = model.text_model(text_input["input_ids"])
                    #     img_forward = model.vision_model(img_inputs["pixel_values"])
                    #     attn_map = gradcam_clip.gradCAM(model, img_inputs, text_input, inputs, getattr(model, "visual_projection"))
                    #     attn = attn_map.squeeze().detach().cpu().numpy()
                    # print("Need to save attn and prob maps")
               
                for b,loc in enumerate(x_locs):
                    i, j = loc[0], loc[1]
                    try:
                        Z[i,j,:] = z_batch[b,:].cpu().detach().numpy()
                        if arm == "test":
                            P[i,j,:] = probs[b,1].cpu().detach().numpy() # class 1
                            # E[i,j,:] = np.mean(attn[b,:].cpu().detach().numpy())
                    except IndexError:
                        print(i, "or", j, "not in index range for Z:", Z.shape)
                        continue
                x_batch, x_locs = [], [] # reset

        # grab sample size emebds from id list
        sample_z_dict = {}
        sample_locs = np.random.choice(len(x_locs_all), size=sample_size, replace=False)
        sample_coords = [x_locs_all[sl] for sl in sample_locs]
        for sc in sample_coords:
            try:
                sample_z_dict[X_id+"_"+str(sc[0])+"_"+str(sc[1])] = Z[sc[0],sc[1],:]
            except IndexError:
                continue

    return Z, P, E, files_remaining, sample_z_dict 


def construct_Zs_efficient(model, patch_dir, dim_dict, save_dir, device, scope="all", arm="train", modelstr="", processor=None, tokenizer=None, text_to_encode=["normal lymph node", "lymph node metastasis"], overwrite_flag=False, embed_format="dict", sample_size=20):
    """
    Reads embeds dict, gathers by image ID, and then creates tensor; saves in a save_dir 
    """
    # iterate with whatever matches in scope
    if scope == "all":
        scope = dim_dict.keys()
        print("we have this # of test set images:", len(scope))
    elif isinstance(scope, list):
        pass

    # load data
    hf = h5py.File(patch_dir, 'r')
    files = list(hf.keys())
    print("We have", len(files), "unique patches to embed")
    model.eval()

    if modelstr not in ["", "tile2vec"]:
        embed_path = arm + "_" + modelstr + "_sampled_inference_z_embeds.obj"
        crop_path =  arm + "_" + modelstr + "_crop_coords.obj"
    else:
        embed_path = arm + "_sampled_inference_z_embeds.obj"
        crop_path =  arm + "_crop_coords.obj"

    if embed_format == "dict":
        if overwrite_flag == False:
            try:
                sampled_embed_dict = utils.deserialize(embed_path)
                crop_dict = utils.deserialize(crop_path)
                print("found embeds and crop coords!")
            except FileNotFoundError:
                sampled_embed_dict = {} # for sampling of embeds 
                crop_dict = {}
        else:
            sampled_embed_dict = {} # for sampling of embeds 
            crop_dict = {}
    elif embed_format == "memmap":
        chunkid_position_path = arm + "_" + modelstr + "_chunkid_position.obj"
        position_chunkid_path = arm + "_" + modelstr + "_position_chunkid.obj"
        model_dims = {"tile2vec": 128, "vit_iid": 1024, "clip": 512, "plip":512}
        storage_shape = ((len(scope) * sample_size), model_dims[modelstr])
        if overwrite_flag == False:
            try:
                sampled_embed_dict = np.memmap(embed_path, dtype='float32', mode='w+', shape=storage_shape) # not an actual dict as you can see
                crop_dict = utils.deserialize(crop_path)
                chunkid_position_dict = utils.deserialize(chunkid_position_path)
                position_chunkid_dict = utils.deserialize(position_chunkid_path)
                print("found embeds and crop coords!")
            except FileNotFoundError:
                sampled_embed_dict = np.memmap(embed_path, dtype='float32', mode='w+', shape=storage_shape)
                crop_dict = {}
                chunkid_position_dict = {}
                position_chunkid_dict = {}
        else:
            sampled_embed_dict = np.memmap(embed_path, dtype='float32', mode='w+', shape=storage_shape)
            crop_dict = {}
            chunkid_position_dict = {}
            position_chunkid_dict = {}

    # iterate and construct
    for idx, X_id in enumerate(scope):
        print("On sample number", idx, "of", len(scope))
        if os.path.exists(save_dir + "/Z-" + X_id + ".npy") and (X_id in crop_dict.keys()) and (overwrite_flag == False):
            print("Already have {Z, embeds, and crop coordinates} for", X_id)
            continue
        
        Z_dim = dim_dict[X_id]
        print(Z_dim)
        if Z_dim[0] == 0 or Z_dim[1] == 0:
            continue

        print("On ID:", X_id)
        if modelstr == "" or modelstr == "tile2vec":
            Z, files, sample_z_dict = construct_Z_efficient(X_id, Z_dim, files, hf, model, device, arm=arm)
        elif modelstr == "vit_iid":
            Z, P, A, G, files, sample_z_dict = inference_transformer(X_id, Z_dim, files, hf, model, processor, tokenizer, device, arm=arm)
        elif modelstr == "plip" or modelstr == "clip":
            Z, P, G, files, sample_z_dict = inference_clip(X_id, Z_dim, files, hf, model, processor, tokenizer, device, arm=arm, text_to_encode=text_to_encode)
        Z, crop_coords = pad_Z(Z)

        crop_dict[X_id] = crop_coords
        np.save(save_dir + "/Z-" + X_id + ".npy", Z)
        if arm == "test":
            save_root = save_dir.split("/")[:-1]
            save_root = '/'.join(save_root)
            if modelstr in ["plip", "clip"]:
                np.save(save_root + "/probs_"+ modelstr + "/P-" + X_id + ".npy", P)
                # np.save(save_root + "/gradcam_" + modelstr + "/G-" + X_id + ".npy", G)
            elif modelstr == "vit_iid":
                np.save(save_root + "/probs_"+ modelstr + "/P-" + X_id + ".npy", P)
                np.save(save_root + "/attn_" + modelstr + "/A-" + X_id + ".npy", A)
                # np.save(save_root + "/gradcam_" + modelstr + "/G-" + X_id + ".npy", G)
            
        if arm == "train":
            if embed_format == "dict":
                for z in sample_z_dict.keys():
                    sampled_embed_dict[z] = sample_z_dict[z] # load embeddings
                utils.serialize(sampled_embed_dict, embed_path)
                print("We now have", len(sampled_embed_dict.keys()), "embeddings stored as a sample")
                print()
            elif embed_format == "memmap":
                for sample_idx, z in enumerate(sample_z_dict.keys()):
                    position = (idx * sample_size) + sample_idx
                    sampled_embed_dict[position, :] = sample_z_dict[z] # load embeddings
                sampled_embed_dict.flush() # flush
                for sample_idx, z in enumerate(sample_z_dict.keys()):
                    position = (idx * sample_size) + sample_idx
                    chunkid_position_dict[z] = position # position --> chunk name
                    position_chunkid_dict[position] = z # position --> chunk name
                utils.serialize(chunkid_position_dict, chunkid_position_path)
                utils.serialize(position_chunkid_dict, position_chunkid_path)
                print("We now have", np.sum(np.sum(sampled_embed_dict, axis=1) != 0.0), "embeddings stored as a sample")
                print()
        utils.serialize(crop_dict, crop_path)
    
    if arm == "train":
        if embed_format == "dict":
            utils.serialize(sampled_embed_dict, embed_path)
        elif embed_format == "memmap":
            #already flushed memmap
            utils.serialize(chunkid_position_dict, chunkid_position_path)
    utils.serialize(crop_dict, crop_path)
    print("serialized sampled numpy embeds at:", embed_path)
    print("saved Z tensors at:", save_dir)
    print("Done!")


# function returns WSS score for k values
def calculate_ideal_k(embed_dict_path, ks, mode="dict"):
    if mode == "dict":
        embed_dict = utils.deserialize(embed_dict_path)
        embeds_list = []
        for k in embed_dict.keys():
            v = embed_dict[k]
            embeds_list.append(v)
        points = np.vstack(embeds_list)
    elif mode == "memmap":
        embed_dict = np.memmap(embed_dict_path, dtype='float32', mode='r', shape=(4440,1024)) # not an actual dict as you can see
        points = embed_dict

    # scaler = StandardScaler()
    # points = scaler.fit_transform(points)

    sse, sil = [], []
    for k in ks:
        # print("fitting k=" + str(k))
        kmeans = KMeans(n_clusters=k, random_state=0).fit(points)
        
        # WSS/elbow method
        #------------------
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0
        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
        sse.append(curr_sse)

        # Silhouette method
        #-------------------
        # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
        labels = kmeans.labels_
        sil.append(silhouette_score(points, labels, metric = 'euclidean'))

    return sse, sil


def get_plot_markers(id_list):
    sal_counter = 0
    so_dict = utils.deserialize("/home/lofi/lofi/src/outputs/train_so_dict.obj")
    ms = []
    for id_val in id_list:
        pieces = id_val.split("_")
        lab = pieces[0]
        id_num = lab+"_"+pieces[1]
        coordi, coordj = int(pieces[2]), int(pieces[3])
        val = 0
        if id_num in so_dict.keys():
            val = so_dict[id_num][(coordi,coordj)]
        if val == 0 and lab == "normal":
            m = "o"
        elif val == 0 and lab == "tumor":
            m = "x"
        elif val == 1 and lab == "tumor":
            sal_counter += 1
            m = "X"
        ms.append(m)
    
    print("sampled", str(sal_counter), "salient objects!")
    return ms


def get_magnitudes(embed_dict_path, mode="dict"):
    if mode == "dict":
        embed_dict = utils.deserialize(embed_dict_path)
        embeds_list = []
        id_list = []
        for k in embed_dict.keys():
            v = embed_dict[k]
            embeds_list.append(v)
            id_list.append(k)
        array = np.vstack(embeds_list)

    elif mode == "memmap":
        embed_dict = np.memmap(embed_dict_path, dtype='float32', mode='r', shape=(4440,1024)) # not an actual dict as you can see
        array = embed_dict

    mags = []
    n,d = array.shape
    print(n,d)
    for i in range(n):
        x = array[i,:]
        # mag = np.sqrt(x.dot(x))
        mag = np.linalg.norm(x, ord=2)
        mags.append(mag)

    plt.figure()
    plt.title("Histogram of sampled embedding L2 norms")
    plt.hist(mags)


def fit_clustering(embed_dict_path, K=20, alg="kmeans_euc", verbosity="full", mode="dict", mapping_path=None):
    
    if mode == "dict":
        embed_dict = utils.deserialize(embed_dict_path)
        id_list = []
        for k in embed_dict.keys():
            id_list.append(k)
    elif mode == "memmap":
        embed_dict = np.memmap(embed_dict_path, dtype='float32', mode='r', shape=(4440,1024)) # not an actual dict as you can see
        mapping_dict = utils.deserialize(mapping_path)
        id_list = mapping_dict.keys()

    # create df and partition by marker
    ms = get_plot_markers(id_list)
    uniques_ms = set(ms)
    ms_dict = dict(zip(id_list, ms))
    embed_df = pd.DataFrame.from_dict(ms_dict, orient='index', columns=["marker"])
    o_df = embed_df.loc[embed_df['marker'] == "o"]
    x_df = embed_df.loc[embed_df['marker'] == "x"]
    X_df = embed_df.loc[embed_df['marker'] == "X"]

    embeds_list_o = []
    for k in list(o_df.index):
        if mode == "dict":
            v = embed_dict[k]
        elif mode == "memmap":
            pos = mapping_dict[k]
            v = embed_dict[pos,:]
        embeds_list_o.append(v)
    array_o = np.vstack(embeds_list_o)
    idx_o = len(embeds_list_o)

    embeds_list_x = []
    for k in list(x_df.index):
        if mode == "dict":
            v = embed_dict[k]
        elif mode == "memmap":
            pos = mapping_dict[k]
            v = embed_dict[pos,:]
        embeds_list_x.append(v)
    array_x = np.vstack(embeds_list_x)
    idx_x = len(embeds_list_x) + idx_o

    embeds_list_X = []
    for k in list(X_df.index):
        if mode == "dict":
            v = embed_dict[k]
        elif mode == "memmap":
            pos = mapping_dict[k]
            v = embed_dict[pos,:]
        embeds_list_X.append(v)
    array_X = np.vstack(embeds_list_X)
    idx_X = len(embeds_list_X) + idx_x

    arrays = [array_o, array_x, array_X]
    array = np.vstack(arrays).astype("double")

    if verbosity == "full":
        print("total embeds:", array.shape[0])
        print("collapsing from dim", array.shape[1], "--> 2")

    # tsne - color by source
    for perplexity in [5,10,20]:
        X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=perplexity).fit_transform(array)
        if verbosity == "full":
            fig, ax1 = plt.subplots(1, 1)
            fig.suptitle('Sampled embeddings for cluster assignment')
            # fig, (ax1, ax2) = plt.subplots(1, 1)
            # fig.suptitle('Sampled embeddings for cluster assignment')
            ax1.set_xlabel("tSNE-0")
            ax1.set_ylabel("tSNE-1")
            # ax1.set_title("t-SNE (perplexity="+str(perplexity)+")")
            # ax1.scatter(X_embedded[:,0], X_embedded[:,1], alpha=0.3, s=1, cmap="Dark2")
        if alg == "kmeans_euc":
            cluster_algo = KMeans(n_clusters=K, random_state=0).fit(array)
        elif alg == "hierarchical_euc":
            # linkage default = ward
            cluster_algo = AgglomerativeClustering(n_clusters=K).fit(array)
        else:
            print("Error: only supporting 'kmeans_euc' and 'hierarchical' for now. No other algos/distance metrics supported as of now")
            exit()
        cluster_labs = cluster_algo.labels_
        if verbosity == "full":
            ax1.set_xlabel("tSNE-0")
            ax1.set_title("t-SNE with K="+str(K)+" clusters (perplexity="+str(perplexity)+")")
            X_embedded_o = X_embedded[0:idx_o,:]
            X_embedded_x = X_embedded[idx_o:idx_x,:]
            X_embedded_X = X_embedded[idx_x:idx_X,:]
            ax1.scatter(X_embedded_o[:,0], X_embedded_o[:,1], c=cluster_labs[0:idx_o], alpha=0.3, s=5, marker="o", cmap="Dark2")
            ax1.scatter(X_embedded_x[:,0], X_embedded_x[:,1], c=cluster_labs[idx_o:idx_x], alpha=0.3, s=30, marker="x", cmap="Dark2")
            ax1.scatter(X_embedded_X[:,0], X_embedded_X[:,1], c=cluster_labs[idx_x:idx_X], alpha=0.6, s=300, edgecolors="k", marker="X", cmap="Dark2") 
            # ax1.scatter(X_embedded[:,0], X_embedded[:,1], c=cluster_labs, alpha=0.3, s=2, marker=ms, cmap="Dark2")
            
    unique, counts = np.unique(cluster_labs, return_counts=True)
    if verbosity == "full":
        plt.figure()
        plt.title("Cluster bar chart")
        plt.bar(unique, height=counts)

    return cluster_algo


def visualize_Z(Z_path, kmeans_model, mode="dict"):
    Z = np.load(Z_path)
    Z_viz, zero_id = reduce_Z(Z, kmeans_model, mode=mode)    
    plt.figure(figsize=(12, 6), dpi=80)
    plt.yticks([])
    plt.xticks([])
    plt.imshow(Z_viz, vmin=0, vmax=zero_id, cmap="Dark2")
    plt.show()


def get_contigs(row):
    contig_lengths = []
    for idx,el in enumerate(row):
        if idx == 0:
            prev_token = 0
        if prev_token == 0: # start new
            if el == 1:
                contig = 1
                prev_token = 1
            elif el == 0:
                continue
        elif prev_token == 1: # continue
            if el == 1:
                contig += 1
                prev_token = 1
            elif el == 0: # store and start over
                contig_lengths.append(contig)
                prev_token = 0
    return contig_lengths


def clean_pcm(Z, Z_id):
    d = Z.shape[2]
    flatZ = Z
    H, W = flatZ.shape[0], flatZ.shape[1]
    print("H, W, d:", H, W, d)

    # clip crop edges
    crop_dict = utils.deserialize("/home/lofi/lofi/src/outputs_tile2vec/test_crop_coords.obj")
    Z_id_trim = Z_id
    if Z_id_trim in crop_dict.keys():
        crop_coords = crop_dict[Z_id_trim]
        i0, i1 = crop_coords[0][0], crop_coords[0][1]
        j0, j1 = crop_coords[1][0], crop_coords[1][1]
        H_crop, W_crop = i1-i0, j1-j0 
        print("H_crop, W_crop:", H_crop, W_crop)
        if (H_crop / 1.75 > W_crop) or (W_crop / 1.75 > H_crop):
            # print("detecting extra long/wide WSI, performing edge trim")
            lr_trim = W_crop // 8
            tb_trim = H_crop // 8
            Z[i0:i0+tb_trim,:,:] = np.zeros((tb_trim,W,d)) # top rows
            Z[i1-1-tb_trim:i1-1,:,:] = np.zeros((tb_trim,W,d)) # bot row
            Z[:,j0:j0+lr_trim,:] = np.zeros((H,lr_trim,d)) # left col
            Z[:,j1-1-lr_trim:j1-1,:] = np.zeros((H,lr_trim,d)) # right col
        else:
            # print("detecting square WSI, performing edge trim")
            lr_trim = W_crop // 15
            tb_trim = H_crop // 12
            print("lr_trim, tb_trim:", lr_trim, tb_trim)
            Z[i0:i0+tb_trim,:,:] = np.zeros((tb_trim,W,d)) # top rows
            print(i1-1-tb_trim, i1-1)
            Z[i1-1-tb_trim:i1-1,:,:] = np.zeros((tb_trim,W,d)) # bot row
            Z[:,j0:j0+lr_trim,:] = np.zeros((H,lr_trim,d)) # left col
            Z[:,j1-1-lr_trim:j1-1,:] = np.zeros((H,lr_trim,d)) # right col
    return Z


def clean_Z(Z, Z_id):
    d = Z.shape[2]
    flatZ = np.sum(Z, axis=2)
    H, W = flatZ.shape[0], flatZ.shape[1]

    # clip crop edges
    crop_dict = utils.deserialize("/home/lofi/lofi/src/outputs/test_crop_coords.obj")
    Z_id_trim = Z_id.split("Z-")[1]
    if Z_id_trim in crop_dict.keys():
        crop_coords = crop_dict[Z_id_trim]
        i0, i1 = crop_coords[0][0], crop_coords[0][1]
        j0, j1 = crop_coords[1][0], crop_coords[1][1]
        H_crop, W_crop = i1-i0, j1-j0 
        if (H_crop / 1.75 > W_crop) or (W_crop / 1.75 > H_crop):
            # print("detecting extra long/wide WSI, performing edge trim")
            lr_trim = W_crop // 8
            tb_trim = H_crop // 8
            Z[i0:i0+tb_trim,:,:] = np.zeros((tb_trim,W,d)) # top rows
            Z[i1-1-tb_trim:i1-1,:,:] = np.zeros((tb_trim,W,d)) # bot row
            Z[:,j0:j0+lr_trim,:] = np.zeros((H,lr_trim,d)) # left col
            Z[:,j1-1-lr_trim:j1-1,:] = np.zeros((H,lr_trim,d)) # right col
        else:
            # print("detecting square WSI, performing edge trim")
            lr_trim = W_crop // 15
            tb_trim = H_crop // 12
            Z[i0:i0+tb_trim,:,:] = np.zeros((tb_trim,W,d)) # top rows
            Z[i1-1-tb_trim:i1-1,:,:] = np.zeros((tb_trim,W,d)) # bot row
            Z[:,j0:j0+lr_trim,:] = np.zeros((H,lr_trim,d)) # left col
            Z[:,j1-1-lr_trim:j1-1,:] = np.zeros((H,lr_trim,d)) # right col
    return Z



def clean_Zs(Z_test_path, save_dir):
    Z_list = os.listdir(Z_test_path)
    plt.figure(figsize=(12,9))
    for idx, file in enumerate(Z_list):
        # pdb.set_trace()
        Z = np.load(Z_test_path + "/" + file)
        flatZ = np.sum(Z, axis=2) != 0.0
        plt.subplot(13,10,idx+1)
        plt.imshow(flatZ, cmap = "bone")
        plt.axis('off')
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.show()
    
    plt.figure(figsize=(12,9))
    for idx, file in enumerate(Z_list):
        Z_id = file.split(".npy")[0]
        Z = np.load(Z_test_path + "/" + file)
        Z = clean_Z(Z, Z_id)
        np.save(save_dir + "/" + Z_id, Z)
        
        flatZ = np.sum(Z, axis=2) != 0.0
        plt.subplot(13,10,idx+1)
        plt.imshow(flatZ, cmap = "bone")
        plt.axis('off')
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.show()

    # plt.figure(figsize=(12,9))
    # for idx, file in enumerate(Z_list):
    #     Z = np.load(Z_test_path + "/" + file)
    #     Zn = clean_Z(Z)
    #     np.save(save_dir + "/" + file.split(".npy")[0], Zn)

    #     flatZ = np.sum(Z, axis=2) > 0
    #     flatZn = np.sum(Zn, axis=2) > 0
    #     diff = flatZ.astype(int) + flatZn.astype(int)
    #     plt.subplot(13,10,idx+1)
    #     plt.imshow(diff, cmap = "jet")
    #     plt.axis('off')
    #     plt.subplots_adjust(wspace=0.01, hspace=0.01)
    # plt.show()
    return


def validation_performance(val_loader, model, device):
    val_losses = []
    model.eval()
    for idx, (Z,y) in enumerate(val_loader):
        B,H,W,D = Z.shape
        Z = Z.to(torch.float)
        y = y.to(torch.long)
        Z = torch.reshape(Z, (B,D,H,W))
        Z.to(device=device)
        y.to(device=device)
        scores = model(Z.cuda())
        loss = F.cross_entropy(scores, y.cuda())
        val_losses.append(loss.item())
    model.train()
    return np.mean(val_losses)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def class_tfidf(Z_path, label_dict, kmeans_model, K, r, mode="mean"):
    # modes include: mean, concat
    # compute class-based stats for training set --> apply to test set
    class0, class1 = [], []
    class0_norm_tuples, class1_norm_tuples = [], []   
    for Z_file in os.listdir(Z_path):
        Z_id = Z_file.split("Z-")[1].split(".npy")[0]
        lab = label_dict[Z_id]
        Z = np.load(Z_path + "/" + Z_file)
        C, _ = reduce_Z(Z, kmeans_model)
        all_cluster_labs, _ = np.unique(kmeans_model.labels_, return_counts=True)
        bag_dict, num_valid_bag = baggify(C, all_cluster_labs)
        num_valid = (num_valid_bag)
        feat_names = [key for key in sorted(bag_dict.keys())]
        counts = np.array([bag_dict[key] for key in sorted(bag_dict.keys())]) # no normalization -- do it later
        if r > 0:
            coloc_dict, num_valid_coloc = colocalization(C, all_cluster_labs, nbhd_size=r)
            num_valid = (num_valid_bag, num_valid_coloc)
            feat_names_add = [key for key in sorted(coloc_dict.keys())]
            feat_names = feat_names + feat_names_add
            counts_annex = np.array([coloc_dict[key] for key in sorted(coloc_dict.keys())]) # no normalization -- do it later
            counts = np.concatenate([counts, counts_annex])       
        if lab == 0:
            class0.append(counts)
            class0_norm_tuples.append(num_valid)
        else:
            class1.append(counts)
            class1_norm_tuples.append(num_valid)
    
    if mode == "concat":  
        class0 = np.array(class0)
        class1 = np.array(class1)
        class0_tf = np.sum(class0, axis=0) / np.sum(class0) # coloc features are amplified
        class1_tf = np.sum(class1, axis=0) / np.sum(class1) 
        doc_freq0 = np.sum(class0, axis=0) > 0.0
        doc_freq1 = np.sum(class1, axis=0) > 0.0
        # print(class1_tf.shape, doc_freq1.shape)
        class0_idf = np.log(1 / 1 + (doc_freq0))
        class1_idf = np.log(1 / 1 + (doc_freq1))
        # this sholuld be over both classes^

        class0_tfidf = class0_tf * class0_idf
        class1_tfidf = class1_tf * class1_idf
        p_vals = None

    elif mode == "mean":
        class0_tf, class1_tf = [], []
        for idx, doc in enumerate(class0):
            if r == 0:
                norm_idx = doc / class0_norm_tuples[idx]
                class0_tf.append(norm_idx)
            else:
                norm_idx = np.concatenate([doc[:K] / class0_norm_tuples[idx][0], doc[K:] / class0_norm_tuples[idx][1]])
                class0_tf.append(norm_idx)
        for idx, doc in enumerate(class1):
            if r == 0:
                norm_idx = doc / class1_norm_tuples[idx]
                class1_tf.append(norm_idx)
            else:
                norm_idx = np.concatenate([doc[:K] / class1_norm_tuples[idx][0], doc[K:] / class1_norm_tuples[idx][1]])
                class1_tf.append(norm_idx)
       
        class0 = np.array(class0_tf)
        class1 = np.array(class1_tf)
        corpus_tf = np.concatenate([class0, class1], axis=0)
        idf = idf_scale(corpus_tf)
        # print(idf)
        class0 = np.array([doc * idf for doc in class0_tf]) 
        class1 = np.array([doc * idf for doc in class1_tf])
        
        # create outputs to return
        class0_tfidf = np.mean(class0, axis=0)
        class1_tfidf = np.mean(class1, axis=0)
        pvals = []
        p = class0.shape[1]
        for idx in range(p):
            # _, pval = scipy.stats.ttest_ind(class1[:,idx], class0[:,idx], equal_var=False)
            _, pval = scipy.stats.mannwhitneyu(class1[:,idx], class0[:,idx])
            pvals.append(pval / p)
        
    return class0_tfidf, class1_tfidf, pvals


def idf_scale(Z):
    """
    TF-IDF normalization
    """
    try:
        Z = Z.numpy() # for inputs that are torch
    except:
        pass
    n = Z.shape[0]
    doc_freq = np.count_nonzero(Z, axis=0)
    idf = np.log(n / (1+doc_freq))
    return idf


def train_on_Z(model, device, optimizer, Z_path_train, Z_path_val, label_dict_path_train, label_dict_path_val, train_set, val_set, kmeans_model, epochs=30, batch_size=10, mode="fullZ", nbhd_size=2, verbosity="low"):
    if mode != "fullZ" or mode != "clusterZ":
        if train_set == None or val_set == None:
            batch_size_train = len(os.listdir(Z_path_train)) # overwrite to make sure we keep all in matrix
            batch_size_val = len(os.listdir(Z_path_val))
        elif train_set == [] or val_set == []:
            batch_size_train = len(os.listdir(Z_path_train))
            batch_size_val = len(os.listdir(Z_path_val))
        else:
            batch_size_train = len(train_set) 
            batch_size_val = len(val_set)
    else:
        batch_size_train = batch_size
        batch_size_val = batch_size

    # instantiate dataloader
    train_dataset = EmbedDataset(Z_path_train, label_dict_path_train, split_list=train_set, mode=mode, kmeans_model=kmeans_model, arm="train", nbhd_size=nbhd_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)
    val_dataset = EmbedDataset(Z_path_val, label_dict_path_val, split_list=val_set, mode=mode, kmeans_model=kmeans_model, arm="test", nbhd_size=nbhd_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, num_workers=1)

    if mode != "fullZ" or mode != "ClusterZ":
        for idx, (Z_train, y_train) in enumerate(train_loader):
            break
        for idx, (Z_val, y_val) in enumerate(val_loader):
            break
        if verbosity == "high":
            print("Z train is shape:", Z_train.shape)
            print("Z val is shape:", Z_val.shape)
            print("Fitting sci-kit learn models: returning y_probs, ys")
        
        # idf scaling
        if mode == "clusterbag" or mode == "coclusterbag":
            idf = idf_scale(Z_train)
            Z_train = Z_train * idf
            Z_val = Z_val * idf

        scaler = StandardScaler()
        Z_train = scaler.fit_transform(Z_train.numpy())
        Z_val = scaler.transform(Z_val.numpy())
        y_train = y_train.numpy().astype(int)
        y_val = y_val.numpy().astype(int)
        # print(Z_train)
        # pdb.set_trace()

        model = model.fit(Z_train, y_train)
        score = model.score(Z_val, y_val)
        print("\toverall mean accuracy", score)
        val_losses = y_val
        train_losses = model.predict_proba(Z_val)
        # placeholder names, really ys, y_preds

        FI = model.coef_[0]
        # explainer = shap.KernelExplainer(model.predict_proba, Z_train)
        # FI = explainer.shap_values(Z_val)
        # print(FI)

    else:
        early_stopper = EarlyStopper(patience=2, min_delta=0.05)
        model.to(device=device)
        model.train()
        train_losses, train_losses_epoch = [], []
        val_losses = []
        for e in range(epochs):
            for idx, (Z,y) in enumerate(train_loader):
                # print("On minibatch:", idx)
                B,H,W,D = Z.shape
                Z = Z.to(torch.float)
                y = y.to(torch.long)
                Z = torch.reshape(Z, (B,D,H,W))
                Z.to(device=device)
                y.to(device=device)
                
                optimizer.zero_grad()
                scores = model(Z.cuda())
                loss = F.cross_entropy(scores, y.cuda())
                loss.backward()
                optimizer.step()
                train_losses_epoch.append(loss.item())
        
            print("end of epoch", e, "train loss:", loss.item())
            train_loss_epoch = np.mean(train_losses_epoch)
            print("end of epoch", e, "AVG train loss:", train_loss_epoch)
            train_losses.append(train_loss_epoch)

            # validation 
            with torch.no_grad(): 
                val_loss = validation_performance(val_loader, model, device)                  
                print("end of epoch", e, "AVG val loss:", val_loss)
                val_losses.append(val_loss)
            print()
            
            if early_stopper.early_stop(val_loss): 
                print("Early stopping triggered!")            
                break
        print("Fitting torch models: returning train losss and val losses")

    return model, FI, train_losses, val_losses


def eval_classifier(ys, y_probs):
    preds = y_probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(ys, preds)
    roc_auc = metrics.auc(fpr, tpr)
    precision, recall, threshold = precision_recall_curve(ys, preds)
    prc_auc = metrics.auc(recall, precision)
    return roc_auc, prc_auc


def diff_exp(tfidfs, K, r, tau=1.0, alpha=0.05, plot_flag=False):
    class0, class1, pvals = tfidfs[(K,r)]
    if plot_flag == True:
        print("any nans for class0?", np.isnan(class0).any())
        print("any nans for class1?", np.isnan(class1).any())
        print("any nans for pvals?", np.isnan(pvals).any())
    
    x = [str(f) for f in range(len(class0))]
    eps = 1e-10
    log2fc = np.log2((class1+eps) / (class0+eps))

    if plot_flag == True:
        print("mean:", np.mean(log2fc))
        print("min:", np.min(log2fc))
        print("max:", np.max(log2fc))
        print("number of + peaks > 1: ", np.sum(log2fc > tau))
        print("number of - peaks < -1:", np.sum(log2fc < -tau))
   
        plt.figure()
        plt.bar(x, height=log2fc)
        plt.title("Class TF-IDFs features (K=" + str(K) + ",r=" + str(r) + ")")
        plt.xlabel("feature")
        plt.xticks([])
        plt.ylabel("Log2FC of class tfidfs")
        plt.show()

    colors = []
    for idx,l in enumerate(log2fc):
        pval = pvals[idx]
        if (pval > alpha) or (np.abs(l) < tau):
            colors.append("gray")
        elif (pval <= alpha) and (l >= tau):
            colors.append("r")
        elif (pval <= alpha) and (l <= -tau):
            colors.append("b")
        else: # catch all
            colors.append("gray")
        
    neglog10p = -np.log10(pvals)
    min_fc = np.min(log2fc)
    max_fc = np.max(log2fc)
    min_p = np.min(neglog10p)
    max_p = np.max(neglog10p)

    if plot_flag == True:
        plt.figure()
        plt.scatter(log2fc, neglog10p, c=colors, alpha=0.3)
        plt.plot([-1, -1],[min_p,max_p], "k--")
        plt.plot([1, 1],[min_p,max_p], "k--")
        plt.plot([min_fc, max_fc],[min_p,min_p], "k--")
        plt.title("Volcano plot for TF-IDFs features (K=" + str(K) + ",r=" + str(r) + ")")
        plt.xlabel("log2(Fold change)")
        plt.ylabel("-log10(p-value)")
        plt.show()

    # print(len(log2fc), len(neglog10p), len(colors))
    # pdb.set_trace()
    return log2fc, neglog10p, colors


def lofi_map(Z_path, kmeans_model, FI, mode="clusterbag", nbhd_size=4):
    Z = np.load(Z_path)
    # print("pre reduce")
    C, zero_id = reduce_Z(Z, kmeans_model)
    # print("post reduce")
    all_cluster_labs, _ = np.unique(kmeans_model.labels_, return_counts=True)
    # print("pre bag")
    bag_dict, _ = baggify(C, all_cluster_labs)
    # print("post bag") 
    feat_names = [key for key in sorted(bag_dict.keys())]
    if mode == "coclusterbag":
        # print("pre coloc") 
        bag_dict, _ = colocalization(C, all_cluster_labs)
        # print("post coloc") 
        feat_names_add = [key for key in sorted(bag_dict.keys())]
        feat_names = feat_names + feat_names_add
    # print(feat_names)

    # build dict of FIs
    score_dict = {}
    for idx, fn in enumerate(feat_names):
        score_dict[fn] = FI[idx]

    # pad and search all 3x3 nbhds
    # combo_dict = {}
    # combos = itertools.combinations_with_replacement(all_cluster_labs, 2)
    bg = np.max(all_cluster_labs) + 1.0

    H,W,_ = C.shape
    C_pad = np.ones((H+nbhd_size+1, W+nbhd_size+1)) * -1
    C_pad[nbhd_size:H+nbhd_size, nbhd_size:W+nbhd_size] = C[:,:,0]
    M_pad = np.zeros((H+nbhd_size+1, W+nbhd_size+1)) # output

    for i in range(1,H+1):
        for j in range(1,W+1):
            cij = C_pad[i,j]
            if (cij == -1.0) or (cij == bg): # padding or bg labels
                continue
            M_pad[i,j] += score_dict[cij]
            # co clusters keep adding attributions
            if mode == "coclusterbag":
                nbhd = C_pad[i-nbhd_size:i+nbhd_size+1, j-nbhd_size:j+nbhd_size+1]
                unique, counts = np.unique(nbhd, return_counts=True)
                for idx,u in enumerate(unique):
                    if (u == -1.0) or (u == bg): # padding or bg labels
                        continue
                    if u > cij:
                        M_pad[i,j] += score_dict[(cij,u)] * counts[idx]
                    elif u < cij:
                        M_pad[i,j] += score_dict[(u,cij)] * counts[idx]
                    elif u == cij: 
                        M_pad[i,j] += score_dict[(cij,u)] * (counts[idx] - 1)           
    return M_pad


def sod_map_generator(Z_path, M, kmeans_model, crop_dict, mask_path=None, nbhd_size=2, highres_flag=False, viz_flag=True, model_details=None):
    Z_id = Z_path.split("/")[-1].split(".npy")[0].split("Z-")[1]
    Z = np.load(Z_path)
    Z_viz, zero_id = reduce_Z(Z, kmeans_model)

    minM = np.min(M)
    maxM = np.max(M)
    maxmag = np.max([np.abs(minM), np.abs(maxM)])
    
    mask = None
    if viz_flag == True and Z_id == "test_001":
        if mask_path is not None:
            try:
                mask = np.load(mask_path)
            except FileNotFoundError:
                print("mask not found, thus also decreasing to low-res")
                highres_flag = False

        fig = plt.figure(figsize=(16, 8))
        plt.suptitle("Slide ID: "+Z_id + "\n"+model_details, fontsize=24)
        if mask_path is None or mask is None:
            grid = ImageGrid(fig, 111,
                            nrows_ncols = (1,3),
                            axes_pad = 0.1,
                            cbar_location = "right",
                            cbar_mode="single",
                            cbar_size="5%",
                            cbar_pad=0.1
                            )
        else:
            grid = ImageGrid(fig, 111,
                            nrows_ncols = (1,4),
                            axes_pad = 0.1,
                            cbar_location = "right",
                            cbar_mode="single",
                            cbar_size="5%",
                            cbar_pad=0.1
                            )

    filename = Z_path.split("/")[-1]
    dropnpy = filename.split(".npy")[0] 
    X_id = dropnpy.split("Z-")[1] 
    i0, i1 = crop_dict[X_id][0]
    j0, j1 = crop_dict[X_id][1]
    C_crop = Z_viz[i0:i1, j0:j1]
    # fg = C_crop < zero_id 
    fg = np.ma.masked_where(C_crop == zero_id, C_crop)
    custom_cmap = plt.get_cmap("Dark2")
    custom_cmap.set_bad(color='white')
    if highres_flag == True:
        fg = cv2.resize(fg, (mask.shape[1],mask.shape[0]), interpolation=cv2.INTER_AREA)
    if viz_flag == True and Z_id == "test_001":
        grid[0].imshow(fg, vmin=0, vmax=zero_id, cmap=custom_cmap)
        #used to be C_crop
        grid[0].set_yticks([])
        grid[0].set_xticks([])
        grid[0].set_title('Sprite', fontsize=20)

    if nbhd_size > 0:
        M_noborder = M[nbhd_size:-nbhd_size, nbhd_size:-nbhd_size]
    else:
        M_noborder = M
    M_crop = M_noborder[i0:i1, j0:j1]
    if highres_flag == True:
        M_crop = cv2.resize(M_crop, (mask.shape[1],mask.shape[0]), interpolation=cv2.INTER_AREA)
    if viz_flag == True and Z_id == "test_001":
        im2 = grid[1].imshow(M_crop, vmin=-maxmag, vmax=maxmag, cmap="bwr")
        grid[1].set_yticks([])
        grid[1].set_xticks([])
        grid[1].set_title('K2 Map', fontsize=20)
        plt.colorbar(im2, cax=grid.cbar_axes[0])

    # bg_val = 0 #1e10
    # M_pos = np.where(M_crop > 0, M_crop, bg_val)
    # vals_to_search = M_crop[M_crop > 0] 
    # thresh = threshold_otsu(vals_to_search) # M_pos)
    # M_thresh = M_pos > thresh
    # fgbg = M_thresh.astype(float)*2 + fg.squeeze().astype(float)
    # print("thresholding at", thresh) 
    # grid[2].imshow(fgbg, cmap="gray")
    
    vals_to_search = M_crop[M_crop > 0.0] # zero is bg, != 0?
    if len(vals_to_search) > 0:
        M_crop = np.where(M_crop == 0.0, minM, M_crop)
        if len(set(vals_to_search)) == 1: # all the same value, rare
            normalized = np.where(M_crop > 0.0, M_crop, 0)
        else:
            normalized = (M_crop-np.min(vals_to_search)) / (np.max(vals_to_search)-np.min(vals_to_search))
            normalized = np.where(normalized > 0.0, normalized, 0)
    else:
        normalized = np.zeros(M_crop.shape)
    if highres_flag == True:
        normalized = cv2.resize(normalized, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
    if viz_flag == True and Z_id == "test_001":
        grid[2].imshow(normalized, cmap="gray")
        grid[2].set_yticks([])
        grid[2].set_xticks([])
        grid[2].axis("off")
        grid[2].set_title('P(z=1)', fontsize=20)

    if mask is not None and viz_flag == True and Z_id == "test_001":
        # print("mask shape:", mask.shape)
        # print("map shape: ", M_crop.shape)
        # mask = cv2.resize(mask, (fgbg.shape[1], fgbg.shape[0]), interpolation=cv2.INTER_AREA)
        if highres_flag == False:
            mask = cv2.resize(mask, (M_crop.shape[1], M_crop.shape[0]), interpolation=cv2.INTER_AREA)
            # print("mask shape:", mask.shape)
        grid[3].imshow(mask, cmap="bone")
        grid[3].set_yticks([])
        grid[3].set_xticks([])
        grid[3].set_title('Ground Truth', fontsize=20)
        # plt.subplots_adjust(top=0.925)
    
    return C_crop, zero_id, M_crop, normalized, mask


def prob_map_to_coords(Mp):
    idxs_i, idxs_j = np.nonzero(Mp > 0)
    values = Mp[Mp > 0]
    d = {"Confidence":values, "X coordinate":idxs_i, "Y coordinate":idxs_j}
    df = pd.DataFrame(data=d)
    return df


def grid_search_elastic(device, embed_path, Z_train_path, lab_train_path, Z_val_path, lab_val_path):
    Ks = [5,7,10] # used to do 5,10,15,20
    clusterers = ["kmeans_euc"] # ["hierarchical_euc"] # 
    featurizers = ["coclusterbag_4"] #[ "clusterbag", "coclusterbag_1", "coclusterbag_2", "meanpool", "maxpool", "meanmaxpool"]
    l1_mixes = [0.5] #elastic net mixing param: 0-lasso, 1-ridge, 0.5EN
    
    y_probs_all = []
    ys_all = []
    model_strs = []
    adapters_trained = []
    FIs = []
    total_models = len(Ks) * len(clusterers) * len(featurizers) * len(l1_mixes)
    kmeans_models = {}

    print("Beginning shallow training on suite of", total_models, "models")
    model_num = 1
    for K in Ks:
        for clusterer in clusterers:
            print("Fitting clusterer:", clusterer + "...")
            kmeans_model = fit_clustering(embed_path, K=K, alg=clusterer, verbosity="none")
            kmeans_models[str(K)+"-"+clusterer] = kmeans_model
            p = (K*K + 3*K)/2
            for featurizer in featurizers:
                if "_" in featurizer:
                    (m, ns) = featurizer.split("_")
                else:
                    m,ns = featurizer, 0
                for l1_mix in l1_mixes:
                    model_str = m + "-K"+str(K)+"-"+clusterer+"-N"+str(ns)+"-L"+str(l1_mix)
                    print("Training [", model_str, "] --> model #", model_num, "/", total_models)
                    model_num += 1
                    adapter = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=l1_mix, random_state=0, max_iter=3000)
                    # adapter = MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(2*p, 2), random_state=0, max_iter=3000)
                    # adapter = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
                    adapter_trained, FI, y_probs, ys = train_on_Z(adapter, device, None, Z_train_path, Z_val_path, lab_train_path, lab_val_path, None, None, epochs=20, mode=m, kmeans_model=kmeans_model, nbhd_size=int(ns))

                    y_probs_all.append(y_probs)
                    ys_all.append(ys)
                    adapters_trained.append(adapter_trained)
                    FIs.append(FI)
                    model_strs.append(model_str)

    return model_strs, y_probs_all, ys_all, FIs, kmeans_models
    

def calc_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[True, False])
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tp+tn+fn+fp)
    ravel = (tp, tn, fp, fn)
    return acc, ravel

def calc_sod_accs(threshs, probs, y_true):    
    accs = []
    ravels = []
    for t in threshs:
        y_pred = probs > t
        y_true = y_true > 0 # extra check on bool enforcement
        acc, ravel = calc_confusion(y_true, y_pred)
        ravels.append(ravel)
        accs.append(acc)
    return accs, ravels

def sod_acc_corpus(threshs, accs_0_dict, accs_1_dict, class1_only=False):
    maxs_1 = []
    means_1 = np.zeros(len(threshs))
    for j,key in enumerate(accs_1_dict.keys()):
        accs = accs_1_dict[key]
        means_1 += accs
        maxs_1.append(np.max(accs))
    
    if class1_only == True:
        norm_means_1 = means_1 / (j+1)
        mean_maxs_1 = np.mean(maxs_1)
        return None, mean_maxs_1, None, None, norm_means_1, None

    maxs_0 = []
    means_0 = np.zeros(len(threshs))
    for i,key in enumerate(accs_0_dict.keys()):
        accs = accs_0_dict[key]
        means_0 += accs
        maxs_0.append(np.max(accs))

    maxs = maxs_0 + maxs_1
    means = means_0 + means_1
    mean_maxs_0, mean_maxs_1, mean_maxs = np.mean(maxs_0), np.mean(maxs_1), np.mean(maxs)

    norm_means_0 = means_0 / (i+1)
    norm_means_1 = means_1 / (j+1)
    norm_means = means / (i+j+2)

    return mean_maxs_0, mean_maxs_1, mean_maxs, norm_means_0, norm_means_1, norm_means


def calc_clf_stats(map_ys):
    preds_fh, preds_gs, ys = [], [], []
    for key in map_ys.keys():
        (y_prob_fh, y_prob_gs, lab) = map_ys[key]
        preds_fh.append(y_prob_fh)
        preds_gs.append(y_prob_gs)
        ys.append(lab)

    fpr, tpr, threshold = metrics.roc_curve(ys, preds_fh)
    roc_auc_fh = metrics.auc(fpr, tpr)
    fpr, tpr, threshold = metrics.roc_curve(ys, preds_gs)
    roc_auc_gs = metrics.auc(fpr, tpr)
    
    precision, recall, threshold = precision_recall_curve(ys, preds_fh)
    prc_auc_fh = metrics.auc(recall, precision)
    precision, recall, threshold = precision_recall_curve(ys, preds_gs)
    prc_auc_gs = metrics.auc(recall, precision)
    return roc_auc_fh, roc_auc_gs, prc_auc_fh, prc_auc_gs


def few_hot_classification(M_probs, few=5):
    # vals_to_search = M_crop[M_crop != 0.0]
    #normed_lofi_fg = (vals_to_search - np.min(vals_to_search)) / (np.max(vals_to_search) - np.min(vals_to_search))
    # y_prob = np.mean(normed_lofi_fg) 
    # relu_lofi = M_crop[M_crop > 0.0]
    relu_prob = M_probs[M_probs > 0.0]
    if len(relu_prob) == 0:
        return 0.0
    sorted_index_array = np.argsort(relu_prob)
    sorted_array = relu_prob[sorted_index_array]
    top_few = sorted_array[-few:]
    return np.mean(top_few)
        

def gauss_smooth_classification(M_probs, s=1):
    smoothed = gaussian_filter(M_probs, sigma=s)
    return np.max(smoothed)

def get_adaptive_threshold(M, valid_coords):
    # threshold like Borji et al / Achanta et al
    # p = (2 / (M.shape[0] * M.shape[1])) * M.sum() # for images
    # pdb.set_trace()
    t = (2 / len(valid_coords)) * M[valid_coords].sum()
    return t

def feature_scale_normalization(M, valid_coords=None):
    if valid_coords is None:
        M = (M - np.min(M)) / (np.max(M) - np.min(M))
    else:
        M[valid_coords] = (M[valid_coords] - np.min(M[valid_coords])) / (np.max(M[valid_coords]) - np.min(M[valid_coords]))
    return M

def locate_valid_coords():
    pass

def process_map(M, valid_coords, mode="attn"):
    """
    M is a raw map
    """
    t = get_adaptive_threshold(M, valid_coords)
    M_af = M > t # adaptive forward
    M_ab = M < t # adaptive backward
    M_pr = None # probabilities from raw
    M_pp = None # probabilities from positive part
    M_pa = None # probabilities from absolute val
    
    if mode == "attn":
        M_pr = feature_scale_normalization(M, valid_coords) # probabilities from raw
        M_pa = feature_scale_normalization(np.abs(M), valid_coords) # probabilities from absolute val
    if mode in ["attn", "lofi"]:
        M_pp = feature_scale_normalization(np.maximum(M, 0), valid_coords) # probabilities from positive part

    return M_af, M_ab, M_pr, M_pp, M_pa


def eval_pcms(Ms_path, gts_path, label_dict, valid_coords_dict, mode="pcm"):
    threshs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    accs_0_valid, accs_1_valid = {}, {}
    accs_0_so, accs_1_so = {}, {}
    ravels_0_so, ravels_1_so = {}, {} 
    ravels_0_valid, ravels_1_valid = {}, {}
    ravels_0_full, ravels_1_full = {}, {}
    ravels_0_static, ravels_1_static = {}, {} # for static attn mask
    ravels_0_static_back, ravels_1_static_back = {}, {} # for static attn mask, backward thresholding
    sod_auroc, sod_auprc, sod_ap = {}, {}, {}
    mod_forward, mod_backward = {}, {}
    mcc_forward, mcc_backward = {}, {}
    prec_forward, prec_backward = {}, {}
    ba_forward, ba_backward = {}, {}

    for idx, M_file in enumerate(os.listdir(Ms_path)):
        if mode == "pcm":
            M_id = M_file.split(".npy")[0].split("P-")[1]
        elif mode == "attn":
            M_id = M_file.split(".npy")[0].split("A-")[1]
        
        lab = label_dict[M_id]
        if lab == 0:
            continue

        # a few class-1 images dont have ground truths
        try:
            gt = np.load(gts_path + "/" + M_id + "_gt.npy")
        except FileNotFoundError:
            continue    

        valid_coords = valid_coords_dict[M_id]
        M_raw = np.load(Ms_path + "/" + M_file)
        M_af, M_ab, M_pr, M_pp, M_pa = process_map(M_raw, valid_coords, mode)
                
        gt_patch = cv2.resize(np.float32(gt), (M_raw.shape[1], M_raw.shape[0]), interpolation=cv2.INTER_AREA) > 0
        
        gt_valid = gt_patch[valid_coords]
        M_valid_af = M_af[valid_coords]
        M_valid_ab = M_ab[valid_coords]
        
        # MULTI-THRESHOLDING: selected and near-continuous
        if mode == "attn":
            Ms_list = [M_pr, M_pp, M_pa]
            ravels_valid_list = []
            for M in Ms_list:
                _, ravels_valid = calc_sod_accs(threshs, M[valid_coords], gt_valid)
                ravels_valid_list.append(ravels_valid)
        elif mode == "pcm":
            Ms_list = [M_raw]
            _, ravels_valid = calc_sod_accs(threshs, M_raw[valid_coords], gt_valid)
            ravels_valid_list = [ravels_valid]

        # load list
        ravels_1_valid[M_id] = ravels_valid_list

        aurocs, auprcs, aps = [], [], []
        for idx, M in enumerate(Ms_list):
            M_valid = np.squeeze(M[valid_coords])
            gt_valid = np.squeeze(gt_valid)
            auroc = roc_auc_score(gt_valid, M_valid)
            precision, recall, _ = precision_recall_curve(gt_valid, M_valid)
            auprc = auc(recall, precision)
            ap = average_precision_score(gt_valid, M_valid)
            aurocs.append(auroc)
            auprcs.append(auprc)
            aps.append(ap)
        
        # load list
        sod_auprc[M_id] = auprcs
        sod_auroc[M_id] = aurocs
        sod_ap[M_id] = aps

        #---------------
        # adaptive eval
        #---------------
        _, ravel_static_f = calc_confusion(gt_valid.astype("bool"), M_valid_af.astype("bool")) # true, preds
        _, ravel_static_b = calc_confusion(gt_valid.astype("bool"), M_valid_ab.astype("bool")) # true, preds
        ravels_1_static[M_id] = ravel_static_f
        ravels_1_static_back[M_id] = ravel_static_b

        # precision
        prec_forward[M_id] = compute_precision(ravel_static_f)
        prec_backward[M_id] = compute_precision(ravel_static_b)  

        # balanced acc
        ba_forward[M_id] = compute_balanced_acc(ravel_static_f)
        ba_backward[M_id] = compute_balanced_acc(ravel_static_b)

        # phi/mcc
        mcc_forward[M_id] = compute_phi(ravel_static_f)
        mcc_backward[M_id] = compute_phi(ravel_static_b)

        #------------
        # structural
        #------------
        im_f = np.array(M_af * 255, dtype = np.uint8)
        im_b = np.array(M_ab * 255, dtype = np.uint8)
            
        numLabels_f, _, stats_f, _ = cv2.connectedComponentsWithStats(im_f, 8, cv2.CV_32S) # 4- or 8- connectivity; 8 is time-efficient 
        numLabels_b, _, stats_b, _ = cv2.connectedComponentsWithStats(im_b, 8, cv2.CV_32S) # 4- or 8- connectivity; 8 is time-efficient
        areas_f = [stats_f[i, cv2.CC_STAT_AREA] for i in range(numLabels_f) if i > 0]
        areas_b = [stats_b[i, cv2.CC_STAT_AREA] for i in range(numLabels_b) if i > 0]
        if areas_f == [] and numLabels_f < 2:
            mod_f = 0
        else:
            mean_area_f = np.mean(areas_f)
            mod_f =  mean_area_f / (numLabels_f - 1)
        mod_forward[M_id] = mod_f

        if areas_b == [] and numLabels_b < 2:
            mod_b = 0
        else:
            mean_area_b = np.mean(areas_b)
            mod_b =  mean_area_b / (numLabels_b - 1)
        mod_backward[M_id] = mod_b

        
    result_dict = {}
    result_dict["ravels_0_so"], result_dict["ravels_1_so"] = ravels_0_so, ravels_1_so
    result_dict["ravels_0_valid"], result_dict["ravels_1_valid"] = ravels_0_valid, ravels_1_valid
    result_dict["ravels_0_full"], result_dict["ravels_1_full"] = ravels_0_full, ravels_1_full
    result_dict["ravels_0_static"], result_dict["ravels_1_static"] = ravels_0_static, ravels_1_static
    result_dict["ravels_0_static_back"], result_dict["ravels_1_static_back"] = ravels_0_static_back, ravels_1_static_back

    result_dict["sod_auroc"], result_dict["sod_auprc"], result_dict["sod_ap"] = sod_auroc, sod_auprc, sod_ap
    result_dict["mod_forward"], result_dict["mod_backward"] = mod_forward, mod_backward
    result_dict["mcc_forward"], result_dict["mcc_backward"] = mcc_forward, mcc_backward
    result_dict["prec_forward"], result_dict["prec_backward"] = prec_forward, prec_backward
    result_dict["ba_forward"], result_dict["ba_backward"] = ba_forward, ba_backward

    # # Acc - all valid patches
    # mean_max_accs_0, mean_max_accs_1, mean_max_accs, mean_accs_thresh_0, mean_accs_thresh_1, mean_accs_thresh = sod_acc_corpus(threshs, accs_0_valid, accs_1_valid)
    # result_dict["mean_max_accs_0_valid"], result_dict["mean_max_accs_1_valid"], result_dict["mean_max_accs_valid"] = mean_max_accs_0, mean_max_accs_1, mean_max_accs
    # result_dict["mean_accs_thresh_0_valid"], result_dict["mean_accs_thresh_1_valid"], result_dict["mean_accs_thresh_valid"] = mean_accs_thresh_0, mean_accs_thresh_1, mean_accs_thresh 
    # # Acc - so patches
    # mean_max_accs_0, mean_max_accs_1, mean_max_accs, mean_accs_thresh_0, mean_accs_thresh_1, mean_accs_thresh = sod_acc_corpus(threshs, accs_0_so, accs_1_so, class1_only=True)
    # result_dict["mean_max_accs_0_so"], result_dict["mean_max_accs_1_so"], result_dict["mean_max_accs_so"] = mean_max_accs_0, mean_max_accs_1, mean_max_accs
    # result_dict["mean_accs_thresh_0_so"], result_dict["mean_accs_thresh_1_so"], result_dict["mean_accs_thresh_so"] = mean_accs_thresh_0, mean_accs_thresh_1, mean_accs_thresh 
    return result_dict


def mean_stat_from_dict(stat_dict):
    # pdb.set_trace()
    vals = []
    replicates_flag = False
    for idx, val in enumerate(stat_dict.values()):
        if idx == 0:
            if type(val) is list:
                if len(val) > 1:
                    replicates_flag = True
                else:
                    replicates_flag = False
        vals.append(val)
    vals = np.array(vals)

    if replicates_flag == True:
        means = np.mean(vals, axis=0)
    else:
        means = np.mean(vals)
    return means

def contiguity_eval():
    pass

def calc_ravels_full(threshs, probs, y_true):
    accs = []
    ravels = []
    for t in threshs:
        y_pred = probs > t
        y_true = y_true > 0 # extra check on bool enforcement
        cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=[True, False])
        tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / (tp+tn+fn+fp)
        ravel = (tp, tn, fp, fn)
        ravels.append(ravel)
        accs.append(acc)
    return accs, ravels


def structural_eval(M_forward, M_backward):
    im_f = np.array(M_forward * 255, dtype = np.uint8)
    im_b = np.array(M_backward * 255, dtype = np.uint8)

    numLabels_f, _, stats_f, _ = cv2.connectedComponentsWithStats(im_f, 8, cv2.CV_32S) # 4- or 8- connectivity; 8 is time-efficient 
    numLabels_b, _, stats_b, _ = cv2.connectedComponentsWithStats(im_b, 8, cv2.CV_32S) # 4- or 8- connectivity; 8 is time-efficient
    areas_f = [stats_f[i, cv2.CC_STAT_AREA] for i in range(numLabels_f) if i > 0]
    areas_b = [stats_b[i, cv2.CC_STAT_AREA] for i in range(numLabels_b) if i > 0]
    if areas_f == [] and numLabels_f < 2:
        mod_f = 0
    else:
        mean_area_f = np.mean(areas_f)
        mod_f =  mean_area_f / (numLabels_f - 1)

    if areas_b == [] and numLabels_b < 2:
        mod_b = 0
    else:
        mean_area_b = np.mean(areas_b)
        mod_b =  mean_area_b / (numLabels_b - 1)
    return mod_f, mod_b


def continuous_eval(tissue_ys_valid, tissue_preds_valid):
    auroc = roc_auc_score(tissue_ys_valid, tissue_preds_valid)
    precision, recall, thresholds = precision_recall_curve(tissue_ys_valid, tissue_preds_valid)
    auprc = auc(recall, precision)
    ap = average_precision_score(tissue_ys_valid, tissue_preds_valid)
    return auroc, auprc, ap

def adaptive_eval(ravel_f, ravel_b):
    # precision
    prec_f = compute_precision(ravel_f)
    prec_b = compute_precision(ravel_b)  

    # balanced acc
    bacc_f = compute_balanced_acc(ravel_f)
    bacc_b = compute_balanced_acc(ravel_b)

    # phi/mcc
    mcc_f = compute_phi(ravel_f)
    mcc_b = compute_phi(ravel_b)
    return prec_f, prec_b, bacc_f, bacc_b, mcc_f, mcc_b

def parse_lofi_models(model_str):
    pieces = model_str.split("-")
    K = int(pieces[0].split("K")[1])
    r = int(pieces[1].split("r")[1])
    g = pieces[2]
    if len(pieces) > 3:
        sig = float(pieces[3].split("a")[1])
        thresh = int(pieces[4].split("t")[1])
    else:
        sig, thresh = None, None
    return K,r,g,sig,thresh


def eval_lofi_top_models(lofi_model_strings, Zs_paths, kmeans_models, elastic_models, detok_models, crop_dicts, gt_path, label_dict, encoder_names):
    all_results = {}
    for stat_idx, lms_group in enumerate(lofi_model_strings):
        print("performing analyses on stat:", stat_idx)
        for e_idx, lms in enumerate(lms_group):
            encoder_name = encoder_names[e_idx]
            full_model_name = encoder_name + "-" + lms
            print("performing analyses on model:", full_model_name)
            Zs_path = Zs_paths[e_idx]
            kmeans_dict = utils.deserialize(kmeans_models[e_idx])
            elastic_dict = utils.deserialize(elastic_models[e_idx])
            detok_dict = utils.deserialize(detok_models[e_idx])
            crop_dict = utils.deserialize(crop_dicts[e_idx])

            K,r,g,sig,thresh = parse_lofi_models(lms)
            kmeans_model = kmeans_dict[K]
            if g == "elastic":
                FI = elastic_dict[(K,r)][0]
            else:
                # [class0_tfidf, class1_tfidf, pvals] = detok_dict[(K,r)]
                log2fc, neglog10p, colors = diff_exp(detok_dict, K, r, tau=thresh, alpha=sig)        
                significance_mask = [0 if c=="gray" else 1 for c in colors]
                FI = log2fc * np.array(significance_mask)

            results = evaluate_adaptive_continuous_structure(Zs_path, kmeans_model, FI, r, crop_dict, gt_path, label_dict, full_model_name)
            all_results[encoder_name + "-" + lms] = results
            
    return all_results
    


def crop_image(arr, crop_coords):
    i0,i1 = crop_coords[0]
    j0,j1 = crop_coords[1]
    return arr[i0:i1, j0:j1, :]


def get_valid_coordinates(Zs_path, gts_path, crop_dict_path):
    crop_dict = utils.deserialize(crop_dict_path)

    valid_coords = {}
    for Z_file in os.listdir(Zs_path):
        Z_id = Z_file.split(".npy")[0].split("Z-")[1]
        gt_path = gts_path + "/" + Z_id + "_gt.npy"
        Z_path = Zs_path + "/" + Z_file
        
        Z = np.load(Z_path)
        crop_coords = crop_dict[Z_id]
        Z_crop = crop_image(Z, crop_coords)

        try:
            gt = np.load(gt_path)
        except FileNotFoundError: # not class 1
            continue
        gt_patch = cv2.resize(np.float32(gt), (Z_crop.shape[1], Z_crop.shape[0]), interpolation=cv2.INTER_AREA) > 0

        # pdb.set_trace()
        # foreground (legit tissue patches):
        idxs_i_C, idxs_j_C = np.nonzero(np.sum(Z_crop,axis=2) != 0.0)
        idxs_i_gt, idxs_j_gt = np.nonzero(gt_patch == True)
        idxs_i_total, idxs_j_total = np.concatenate([idxs_i_C, idxs_i_gt]), np.concatenate([idxs_j_C, idxs_j_gt])
        
        # check gt for duplicate indices
        unique = set()
        for idx in range(len(idxs_i_total)):
            i,j = idxs_i_total[idx],idxs_j_total[idx]
            if (i,j) not in unique:
                unique.add((i,j))
        idxs_i = np.array([el[0] for el in unique])
        idxs_j = np.array([el[1] for el in unique])

        valid_coords[Z_id] = (idxs_i, idxs_j)
        
    return valid_coords


def evaluate_adaptive_continuous_structure(Zs_path, kmeans_model, FI, nbhd_size, crop_dict, gt_path, label_dict, model_details):
    results = {}
    
    if nbhd_size > 0:
        mode = "coclusterbag"
    else:
        mode = "clusterbag"
    
    for Z_file in os.listdir(Zs_path):
        Z_id = Z_file.split(".npy")[0].split("Z-")[1]
        lab = label_dict[Z_id]
        if lab == 0:
            continue
        gt_path_Z = gt_path + "/" + Z_id + "_gt.npy"

        # create LOFI Map
        #-----------------
        Z_path = Zs_path + "/" + Z_file
        # C, zero_id = reduce_Z(np.load(Z_path), kmeans_model)
        M = lofi_map(Z_path, kmeans_model, FI=FI, mode=mode, nbhd_size=nbhd_size)
        C_crop, zero_id, M_crop, M_probs, mask = sod_map_generator(Z_path, M, kmeans_model, crop_dict, nbhd_size=nbhd_size, mask_path=gt_path_Z, viz_flag=True, model_details=model_details)

        # Preprocess Mask
        #-----------------
        if lab == 1:
            try:
                gt = np.load(gt_path + "/" + Z_id + "_gt.npy")
            except FileNotFoundError:
                pass
            gt_patch = cv2.resize(np.float32(gt), (M_probs.shape[1], M_probs.shape[0]), interpolation=cv2.INTER_AREA) > 0
        else:
            continue # skip class-0

        # foreground (legit tissue patches):
        idxs_i_C, idxs_j_C = np.nonzero(C_crop[:,:,0] != zero_id)
        idxs_i_gt, idxs_j_gt = np.nonzero(gt_patch > 0)
        idxs_i_total, idxs_j_total = np.concatenate([idxs_i_C, idxs_i_gt]), np.concatenate([idxs_j_C, idxs_j_gt])

        # check gt for duplicate indices
        unique = set()
        for idx in range(len(idxs_i_total)):
            i,j = idxs_i_total[idx],idxs_j_total[idx]
            if (i,j) not in unique:
                unique.add((i,j))
        idxs_i = np.array([el[0] for el in unique])
        idxs_j = np.array([el[1] for el in unique])

        # valid tissue subset
        tissue_preds_valid = M_probs[idxs_i, idxs_j]
        tissue_ys_valid = gt_patch[idxs_i, idxs_j]

        # threshold
        p = get_adaptive_threshold(M_probs, (idxs_i, idxs_j)) # use valid subset
        M_forward = M_crop > p
        M_backward = M_crop < p
        M_f_valid = M_forward[idxs_i, idxs_j]
        M_b_valid = M_backward[idxs_i, idxs_j]

        y_true = tissue_ys_valid > 0 # extra check on bool enforcement
        _, ravel_f = calc_confusion(y_true, M_f_valid)
        _, ravel_b = calc_confusion(y_true, M_b_valid)

        # adaptive eval
        prec_f, prec_b, bacc_f, bacc_b, mcc_f, mcc_b = adaptive_eval(ravel_f, ravel_b)

        # continuous eval
        auroc, auprc, ap = continuous_eval(tissue_ys_valid, tissue_preds_valid)

        # structural
        mod_f, mod_b = structural_eval(M_forward, M_backward)

        # store so we can get averages for test stat
        results[Z_id] = [prec_f, prec_b, bacc_f, bacc_b, mcc_f, mcc_b, auroc, auprc, ap, mod_f, mod_b]
    
    return results


def generate_model_outputs(Zs_path, kmeans_model, FI, mode, nbhd_size, crop_dict, save_path, gt_path, label_dict):
    threshs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    lofi_clf = {}
    accs_0_valid, accs_1_valid = {}, {}
    accs_0_so, accs_1_so = {}, {}
    ravels_0_so, ravels_1_so = {}, {} 
    ravels_0_valid, ravels_1_valid = {}, {}
    ravels_0_full, ravels_1_full = {}, {}

    metric_0 = MeanAveragePrecision() # kick off COCO mAP
    metric_1 = MeanAveragePrecision() # kick off COCO mAP
    metric_2 = MeanAveragePrecision() # kick off COCO mAP
    metric_3 = MeanAveragePrecision() # kick off COCO mAP
    metric_4 = MeanAveragePrecision() # kick off COCO mAP
    metric_5 = MeanAveragePrecision() # kick off COCO mAP
    metric_6 = MeanAveragePrecision() # kick off COCO mAP
    metric_7 = MeanAveragePrecision() # kick off COCO mAP
    metric_8 = MeanAveragePrecision() # kick off COCO mAP
    metric_9 = MeanAveragePrecision() # kick off COCO mAP
    metric_10 = MeanAveragePrecision() # kick off COCO mAP
    metrics = [metric_0, metric_1, metric_2, metric_3, metric_4, metric_5, metric_6, metric_7, metric_8, metric_9, metric_10]
    update_hit = False
    hits = [0] * 11

    gt_exists = False
    for Z_file in os.listdir(Zs_path):
        Z_id = Z_file.split(".npy")[0].split("Z-")[1]
        lab = label_dict[Z_id]
        Z_path = Zs_path + "/" + Z_file
        # C, zero_id = reduce_Z(np.load(Z_path), kmeans_model)
        M = lofi_map(Z_path, kmeans_model, FI=FI, mode=mode, nbhd_size=nbhd_size)
        C_crop, zero_id, M_crop, M_probs, mask = sod_map_generator(Z_path, M, kmeans_model, crop_dict, nbhd_size=nbhd_size, mask_path=None, viz_flag=False)
        
        # Preprocess Mask
        #-----------------
        if lab == 1:
            try:
                gt = np.load(gt_path + "/" + Z_id + "_gt.npy")
                gt_exists = True
            except FileNotFoundError:
                gt_exists = False
                pass
            gt_patch = cv2.resize(np.float32(gt), (M_probs.shape[1], M_probs.shape[0]), interpolation=cv2.INTER_AREA) > 0
        else:
            gt_exists = False
            gt_patch = np.zeros((M_probs.shape))

        # Preprocess Mosaic
        #--------------------
        # foreground (legit tissue patches):
        idxs_i_C, idxs_j_C = np.nonzero(C_crop[:,:,0] != zero_id)
        idxs_i_gt, idxs_j_gt = np.nonzero(gt_patch > 0)
        idxs_i_total, idxs_j_total = np.concatenate([idxs_i_C, idxs_i_gt]), np.concatenate([idxs_j_C, idxs_j_gt])
        # print("# tissue patches from our two sources:", len(idxs_i_C), len(idxs_i_gt))

        # check gt for duplicate indices
        unique = set()
        for idx in range(len(idxs_i_total)):
            i,j = idxs_i_total[idx],idxs_j_total[idx]
            if (i,j) not in unique:
                unique.add((i,j))
        idxs_i = np.array([el[0] for el in unique])
        idxs_j = np.array([el[1] for el in unique])

        # valid tissue subset
        tissue_preds_valid = M_probs[idxs_i, idxs_j]
        tissue_ys_valid = gt_patch[idxs_i, idxs_j]

        # SO tissue subset
        if gt_exists == True:
            tissue_preds_so = M_probs[idxs_i_gt, idxs_j_gt]
            tissue_ys_so = gt_patch[idxs_i_gt, idxs_j_gt]

        # Img-level clf - based on map
        #-----------------------------
        # print("Label:", lab)
        y_prob_fh = few_hot_classification(M_probs)
        y_prob_gs = gauss_smooth_classification(M_probs)
        lofi_clf[Z_id] = (y_prob_fh, y_prob_gs, lab)

        # SOD Accuracy
        #--------------
        accs_valid, ravels_valid = calc_sod_accs(threshs, tissue_preds_valid, tissue_ys_valid)
        if lab == 1:
            accs_1_valid[Z_id] = accs_valid
            ravels_1_valid[Z_id] = ravels_valid
            accs_full, ravels_full = calc_ravels_full(threshs, M_probs, gt_patch)
            ravels_1_full[Z_id] = (accs_full, ravels_full)
        else:
            accs_0_valid[Z_id] = accs_valid
            ravels_0_valid[Z_id] = ravels_valid
            accs_full, ravels_full = calc_ravels_full(threshs, M_probs, gt_patch)
            ravels_0_full[Z_id] = (accs_full, ravels_full)

        if gt_exists == True:
            accs_so, ravels_so = calc_sod_accs(threshs, tissue_preds_so, tissue_ys_so)
            if lab == 1:
                accs_1_so[Z_id] = accs_so
                ravels_1_so[Z_id] = ravels_so
            else: # useless
                accs_0_so[Z_id] = accs_so
                ravels_0_so[Z_id] = ravels_so

        # SOD mAP
        #---------
        if gt_exists == True:
            M_bin = M_probs > 0.0  # enforce binarization
            gt_bin = gt_patch > 0.0  # enforce binarization
            if (np.sum(M_bin) == 0.0) or (np.sum(gt_patch) == 0.0):
                pass
            else:
                # batch size of 1
                for idx, metric in enumerate(metrics):
                    M_thresh = M_probs > threshs[idx]
                    if np.sum(M_thresh) == 0.0:
                        pass
                    else:
                        update_hit = True
                        metric.update(preds=M_thresh, target=gt_bin)
                        # pdb.set_trace()
                        hits[idx] += 1
        
        # SOD FROC - analysis from Camelyon
        #-----------------------------------
        # if gt_exists == True:
        #     print("FROC analysis: processing mask->csv for:", Z_id)
        #     # resize to level 5
        #     M_probs = cv2.resize(np.float32(M_probs), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
        #     df = prob_map_to_coords(M_probs)
        #     df.to_csv(save_path + "/" + Z_id + ".csv", index=False)
        # print()

    result_dict = {}
    result_dict["ravels_0_so"], result_dict["ravels_1_so"] = ravels_0_so, ravels_1_so
    result_dict["ravels_0_valid"], result_dict["ravels_1_valid"] = ravels_0_valid, ravels_1_valid
    result_dict["ravels_0_full"], result_dict["ravels_1_full"] = ravels_0_full, ravels_1_full

    # Acc - all valid patches
    mean_max_accs_0, mean_max_accs_1, mean_max_accs, mean_accs_thresh_0, mean_accs_thresh_1, mean_accs_thresh = sod_acc_corpus(threshs, accs_0_valid, accs_1_valid)
    result_dict["mean_max_accs_0_valid"], result_dict["mean_max_accs_1_valid"], result_dict["mean_max_accs_valid"] = mean_max_accs_0, mean_max_accs_1, mean_max_accs
    result_dict["mean_accs_thresh_0_valid"], result_dict["mean_accs_thresh_1_valid"], result_dict["mean_accs_thresh_valid"] = mean_accs_thresh_0, mean_accs_thresh_1, mean_accs_thresh 
    # Acc - so patches
    mean_max_accs_0, mean_max_accs_1, mean_max_accs, mean_accs_thresh_0, mean_accs_thresh_1, mean_accs_thresh = sod_acc_corpus(threshs, accs_0_so, accs_1_so, class1_only=True)
    result_dict["mean_max_accs_0_so"], result_dict["mean_max_accs_1_so"], result_dict["mean_max_accs_so"] = mean_max_accs_0, mean_max_accs_1, mean_max_accs
    result_dict["mean_accs_thresh_0_so"], result_dict["mean_accs_thresh_1_so"], result_dict["mean_accs_thresh_so"] = mean_accs_thresh_0, mean_accs_thresh_1, mean_accs_thresh 
    # image clf
    roc_auc_fh, roc_auc_gs, prc_auc_fh, prc_auc_gs = calc_clf_stats(lofi_clf)
    result_dict["roc_auc_fh"], result_dict["roc_auc_gs"] = roc_auc_fh, roc_auc_gs
    result_dict["prc_auc_fh"], result_dict["prc_auc_gs"] = prc_auc_fh, prc_auc_gs
    # mAP
    if update_hit == False:
        print("warning: some threhsolds of mAP were not generating evaluable SOD prediction maps")
    for idx in range(len(metrics)):
        map_dict_idx = metrics[idx].compute()
        result_dict["map_stats_" + str(idx)] = map_dict_idx
    # run_test(gt_path, save_path, label_dict)
    print("mAP hits:", hits)

    return result_dict


def run_lofi_gridsearch(modelstr, embed_path, device, Zs_path_train, Zs_path_test, label_dict_path_train, label_dict_path_test, crop_dict, csv_save_path, gts_path, mode="dict", mapping_path=None):
    label_dict_train = utils.deserialize(label_dict_path_train)
    label_dict_test = utils.deserialize(label_dict_path_test)

    root_path = embed_path.split("/")[:-1]
    root_path = "/".join(root_path)

    kmeans_filename = root_path + "/" + modelstr + "_kmeans_models_dict.obj"
    try:
        kmeans_models = utils.deserialize(kmeans_filename)
    except:
        kmeans_models = {}
    print("K-means models cached (K):", list(kmeans_models.keys()))

    tfidf_filename = root_path + "/" + modelstr + "_tfidfs_dict.obj"
    try:
        tfidfs = utils.deserialize(tfidf_filename)
    except:
        tfidfs = {}
    print("tf-idf models cached (K,r):", list(tfidfs.keys()))

    elastic_filename = root_path + "/" + modelstr + "_elastic_dict.obj"
    try:
        elastic = utils.deserialize(elastic_filename)
    except:
        elastic = {}
    print("elasticNet models cached (K,r):", list(elastic.keys()))

    # big doodad -- dict of all model runs and stats
    results_filename = root_path + "/" + modelstr + "_all_results_lofi.obj"
    try:
        all_results = utils.deserialize(results_filename)
    except:
        all_results = {}
    print("all models cached so far:", list(all_results.keys()))
    print("saving results to:", results_filename + "...")

    # hyperparameter sweep
    Ks = [10,15,20,25,30]
    rs = [0,1,2,4,8]
    alphas = [0.01, 0.025, 0.05, 1e10]
    taus = [0,1,2]

    num_models = (len(Ks) * len(rs) * len(alphas) * len(taus)) + (len(Ks) * len(rs))
    print("Expecting", num_models, "for this grid_search... get strapped in")
    model_counter = 0
    hf = False

    for K in Ks:
        if K in kmeans_models.keys():
            kmeans_model = kmeans_models[K]
            print("Using existing k-means model")
        else:
            print("Fitting new k-means model...")
            kmeans_model = fit_clustering(embed_path, K=K, alg="kmeans_euc", verbosity="none", mode=mode, mapping_path=mapping_path)
            kmeans_models[K] = kmeans_model
            serialize(kmeans_models, kmeans_filename)

        for r in rs:
            if r > 0:
                m = "coclusterbag"
            else:
                m = "clusterbag"

            # ElasticNet
            #------------
            if (K,r) in elastic.keys():
                print("Using precomputed elasticNet! skipping run for (K,r):", K, r)
                FI, roc_auc, prc_auc = elastic[(K,r)]
            else:
                print("Training elasticNet for (K,r):", K, r)
                model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=0, max_iter=3000)
                _, FI, y_probs, ys = train_on_Z(model, device, None, Zs_path_train, Zs_path_test, label_dict_path_train, label_dict_path_test, None, None, epochs=20, mode=m, kmeans_model=kmeans_models[K], nbhd_size=r)
                roc_auc, prc_auc = eval_classifier(ys, y_probs) # test set clf stats
                elastic[(K,r)] = [FI, roc_auc, prc_auc]
                serialize(elastic, elastic_filename)

            model_str = "K"+str(K)+"-r"+str(r)+"-elastic"
            if model_str in all_results.keys():
                print("Using cached model:", model_str)
            else:
                result_dict = generate_model_outputs(Zs_path_test, kmeans_models[K], FI, m, r, crop_dict, csv_save_path, gts_path, label_dict_test)
                all_results[model_str] = result_dict
                serialize(all_results, results_filename)
            model_counter += 1

            # DE
            #-------
            if (K,r) in tfidfs.keys():
                print("Using precomputed tfidf! skipping run for (K,r):", K, r)
                class0_tfidf, class1_tfidf, pvals = tfidfs[(K,r)]
            else:
                print("Calculating tfidf for (K,r):", K, r)
                class0_tfidf, class1_tfidf, pvals = class_tfidf(Zs_path_train, label_dict_train, kmeans_models[K], K, r)
                tfidfs[(K,r)] = [class0_tfidf, class1_tfidf, pvals]
                serialize(tfidfs, tfidf_filename)  
            
            for alpha in alphas:
                for tau in taus:
                    log2fc, neglog10p, colors = diff_exp(tfidfs, K, r, tau=tau, alpha=alpha)        
                    significance_mask = [0 if c=="gray" else 1 for c in colors]
                    FI = log2fc * np.array(significance_mask)

                    model_str = "K"+str(K)+"-r"+str(r)+"-DEtok"+"-a"+str(alpha)+"-t"+str(tau)
                    if model_str in all_results.keys():
                        print("Using cached model:", model_str)
                    else:
                        result_dict = generate_model_outputs(Zs_path_test, kmeans_models[K], FI, m, r, crop_dict, csv_save_path, gts_path, label_dict_test)
                        all_results[model_str] = result_dict
                        serialize(all_results, results_filename)
                    model_counter += 1   

            print("We've completed/stored", model_counter, "models on this fucntion call so far...")
            print("Another check for model count (previously cached):", len(all_results.keys()))
            print("Models so far:", list(all_results.keys()))


def grab_stats(all_results, elastic):
    search_plot_dict = {}

    for model_str in all_results.keys():
        pieces = model_str.split("-")
        K, r, model_class = int(pieces[0].split("K")[1]), int(pieces[1].split("r")[1]), pieces[2]
        alpha = None
        tau = None
        if len(pieces) > 3:
            alpha = float(pieces[3].split("a")[1])
            tau = int(pieces[4].split("t")[1])
        key = (K,r,model_class,alpha,tau)
        model_vals = all_results[model_str]

        # classification
        roc_auc, prc_auc = 0.0, 0.0 # non-elastic
        if model_class == "elastic":
            _, roc_auc, prc_auc = elastic[(K,r)]
        roc_auc_fh, roc_auc_gs = model_vals["roc_auc_fh"], model_vals["roc_auc_gs"]
        prc_auc_fh, prc_auc_gs = model_vals["prc_auc_fh"], model_vals["prc_auc_gs"]
        max_auc_roc, max_auc_prc = np.max([roc_auc, roc_auc_fh, roc_auc_gs]), np.max([prc_auc, prc_auc_fh, prc_auc_gs]) 

        # sod
        acc_valid = model_vals["mean_max_accs_1_valid"]
        acc_so = model_vals["mean_max_accs_1_so"]

        vals = (max_auc_roc, max_auc_prc, acc_valid, acc_so)
        search_plot_dict[key] = vals

        # convert to df
        df_rows = [list(key) + list(val) for key,val in search_plot_dict.items()]
        cols = ["K", "r", "model_class", "a", "t", "auroc", "auprc", "acc_fg", "acc_so"]
        results_df = pd.DataFrame.from_dict(df_rows, orient="columns")
        results_df.columns = cols

    return results_df


def grab_stats_multithresh(all_results, elastic):
    search_plot_dict = {}
    mean_precision, _ = compute_top_cm_metric_lofi(all_results, "precision", mode="thresh_max")
    mean_bacc, _ = compute_top_cm_metric_lofi(all_results, "balanced_accuracy", mode="thresh_max")

    for model_str in all_results.keys():
        pieces = model_str.split("-")
        K, r, model_class = int(pieces[0].split("K")[1]), int(pieces[1].split("r")[1]), pieces[2]
        alpha = None
        tau = None
        if len(pieces) > 3:
            alpha = float(pieces[3].split("a")[1])
            tau = int(pieces[4].split("t")[1])
        key = (K,r,model_class,alpha,tau)
        model_vals = all_results[model_str]

        # classification
        roc_auc, prc_auc = 0.0, 0.0 # non-elastic
        if model_class == "elastic":
            _, roc_auc, prc_auc = elastic[(K,r)]
        roc_auc_fh, roc_auc_gs = model_vals["roc_auc_fh"], model_vals["roc_auc_gs"]
        prc_auc_fh, prc_auc_gs = model_vals["prc_auc_fh"], model_vals["prc_auc_gs"]
        max_auc_roc, max_auc_prc = np.max([roc_auc, roc_auc_fh, roc_auc_gs]), np.max([prc_auc, prc_auc_fh, prc_auc_gs]) 

        # sod
        prec = mean_precision[model_str]["ravels_1_valid"]
        bacc = mean_bacc[model_str]["ravels_1_valid"]

        vals = (max_auc_roc, max_auc_prc, prec, bacc)
        search_plot_dict[key] = vals

        # convert to df
        df_rows = [list(key) + list(val) for key,val in search_plot_dict.items()]
        cols = ["K", "r", "model_class", "a", "t", "auroc", "auprc", "prec", "bacc"]
        results_df = pd.DataFrame.from_dict(df_rows, orient="columns")
        results_df.columns = cols

    return results_df



def plot_gridsearch_encoder(detok_df, elastic_df, other_results=None):
    fig, axs = plt.subplots(2, 2, figsize=(12,12))
    fig.suptitle('Classification vs wsSOD Performance', fontsize=24)

    SCALE = 15
    c_detok = detok_df["K"]
    s_detok = (np.array(detok_df["r"]) + 2) * SCALE
    c_elastic = elastic_df["K"]
    s_elastic = (np.array(elastic_df["r"]) + 2) * SCALE
    ec="none"

    if other_results:
        mean_max_accs_1_valid = other_results["mean_max_accs_1_valid"]
        mean_max_accs_1_so = other_results["mean_max_accs_1_so"]

    if other_results:
        xmin_roc = np.min([np.min(detok_df["auroc"]), np.min(elastic_df["auroc"])])
        xmax_roc = np.max([np.max(detok_df["auroc"]), np.max(elastic_df["auroc"])])
        xmin_prc = np.min([np.min(detok_df["auprc"]), np.min(elastic_df["auprc"])])
        xmax_prc = np.max([np.max(detok_df["auprc"]), np.max(elastic_df["auprc"])])
            
    axs[0,0].scatter(detok_df["auroc"], detok_df["acc_fg"], alpha=0.2, c=c_detok, edgecolors=ec, s=s_detok, marker="o")
    axs[0,0].scatter(elastic_df["auroc"], elastic_df["acc_fg"], alpha=0.8, c=c_elastic, s=s_elastic, marker="x")
    axs[0,0].set_xlabel('AUROC', fontsize=18)
    axs[0,0].set_ylabel('Mean-max Accuracy (FG)', fontsize=18)

    axs[0,1].scatter(detok_df["auprc"], detok_df["acc_fg"], alpha=0.2, c=c_detok, edgecolors=ec, s=s_detok, marker="o")
    axs[0,1].scatter(elastic_df["auprc"], elastic_df["acc_fg"], alpha=0.8, c=c_elastic, s=s_elastic, marker="x")
    axs[0,1].set_xlabel('AUPRC', fontsize=18)
    # axs[0,1].set_ylabel('Mean-max Accuracy (FG)', fontsize=18)

    axs[1,0].scatter(detok_df["auroc"], detok_df["acc_so"], alpha=0.2, c=c_detok, edgecolors=ec, s=s_detok, marker="o")
    axs[1,0].scatter(elastic_df["auroc"], elastic_df["acc_so"], alpha=0.8, c=c_elastic, s=s_elastic, marker="x")
    axs[1,0].set_xlabel('AUROC', fontsize=18)
    axs[1,0].set_ylabel('Mean-max Accuracy (SO)', fontsize=18)

    axs[1,1].scatter(detok_df["auprc"], detok_df["acc_so"], alpha=0.2, c=c_detok, edgecolors=ec, s=s_detok, marker="o")
    axBR = axs[1,1].scatter(elastic_df["auprc"], elastic_df["acc_so"], alpha=0.8, c=c_elastic, s=s_elastic, marker="x")
    axs[1,1].set_xlabel('AUPRC', fontsize=18)
    # axs[1,1].set_ylabel('Mean-max Accuracy (SO)', fontsize=18)

    if other_results:
        # other results
        axs[0,0].plot([xmin_roc, xmax_roc], [mean_max_accs_1_valid, mean_max_accs_1_valid], color="k", linestyle="--", linewidth=1)
        axs[0,1].plot([xmin_prc, xmax_prc], [mean_max_accs_1_valid, mean_max_accs_1_valid], color="k", linestyle="--", linewidth=1)
        axs[1,0].plot([xmin_roc, xmax_roc], [mean_max_accs_1_so, mean_max_accs_1_so], color="k", linestyle="--", linewidth=1)
        axs[1,1].plot([xmin_prc, xmax_prc], [mean_max_accs_1_so, mean_max_accs_1_so], color="k", linestyle="--", linewidth=1)


    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(axBR, cax=cbar_ax)
    plt.text(0.32,9.3,"K", {"fontsize":16})

    handles, labels = axBR.legend_elements(prop="sizes", alpha=0.6, num=6)     
    labels = ["0", "1", "2", "4", "8"]
    legend = plt.legend(handles, labels, loc=(0,1.01), title="r")

    plt.subplots_adjust(top=0.925)
    plt.show()
    return



def plot_gridsearch_encoder_fg(detok_df, elastic_df, other_results=None, model_str=None):
    fig, axs = plt.subplots(2, 2, figsize=(12,12))
    fig.suptitle(model_str + ': Classification vs SOD Performance', fontsize=24)

    SCALE = 15
    c_detok = detok_df["K"]
    s_detok = (np.array(detok_df["r"]) + 2) * SCALE
    c_elastic = elastic_df["K"]
    s_elastic = (np.array(elastic_df["r"]) + 2) * SCALE
    ec="none"

    if other_results:
        mean_max_accs_1_valid = other_results["mean_max_accs_1_valid"]
        mean_max_accs_1_so = other_results["mean_max_accs_1_so"]

    if other_results:
        xmin_roc = np.min([np.min(detok_df["auroc"]), np.min(elastic_df["auroc"])])
        xmax_roc = np.max([np.max(detok_df["auroc"]), np.max(elastic_df["auroc"])])
        xmin_prc = np.min([np.min(detok_df["auprc"]), np.min(elastic_df["auprc"])])
        xmax_prc = np.max([np.max(detok_df["auprc"]), np.max(elastic_df["auprc"])])
            
    axs[0,0].scatter(detok_df["auroc"], detok_df["prec"], alpha=0.2, c=c_detok, edgecolors=ec, s=s_detok, marker="o")
    axs[0,0].scatter(elastic_df["auroc"], elastic_df["prec"], alpha=0.8, c=c_elastic, s=s_elastic, marker="x")
    axs[0,0].set_xlabel('AUROC', fontsize=18)
    axs[0,0].set_ylabel('Precision', fontsize=18)

    axs[0,1].scatter(detok_df["auprc"], detok_df["prec"], alpha=0.2, c=c_detok, edgecolors=ec, s=s_detok, marker="o")
    axs[0,1].scatter(elastic_df["auprc"], elastic_df["prec"], alpha=0.8, c=c_elastic, s=s_elastic, marker="x")
    axs[0,1].set_xlabel('AUPRC', fontsize=18)
    # axs[0,1].set_ylabel('Mean-max Accuracy (FG)', fontsize=18)

    axs[1,0].scatter(detok_df["auroc"], detok_df["bacc"], alpha=0.2, c=c_detok, edgecolors=ec, s=s_detok, marker="o")
    axs[1,0].scatter(elastic_df["auroc"], elastic_df["bacc"], alpha=0.8, c=c_elastic, s=s_elastic, marker="x")
    axs[1,0].set_xlabel('AUROC', fontsize=18)
    axs[1,0].set_ylabel('Balanced Accuracy', fontsize=18)

    axs[1,1].scatter(detok_df["auprc"], detok_df["bacc"], alpha=0.2, c=c_detok, edgecolors=ec, s=s_detok, marker="o")
    axBR = axs[1,1].scatter(elastic_df["auprc"], elastic_df["bacc"], alpha=0.8, c=c_elastic, s=s_elastic, marker="x")
    axs[1,1].set_xlabel('AUPRC', fontsize=18)
    # axs[1,1].set_ylabel('Mean-max Accuracy (SO)', fontsize=18)

    if other_results:
        # other results
        axs[0,0].plot([xmin_roc, xmax_roc], [mean_max_accs_1_valid, mean_max_accs_1_valid], color="k", linestyle="--", linewidth=1)
        axs[0,1].plot([xmin_prc, xmax_prc], [mean_max_accs_1_valid, mean_max_accs_1_valid], color="k", linestyle="--", linewidth=1)
        axs[1,0].plot([xmin_roc, xmax_roc], [mean_max_accs_1_so, mean_max_accs_1_so], color="k", linestyle="--", linewidth=1)
        axs[1,1].plot([xmin_prc, xmax_prc], [mean_max_accs_1_so, mean_max_accs_1_so], color="k", linestyle="--", linewidth=1)


    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(axBR, cax=cbar_ax)
    plt.text(0.32,9.3,"K", {"fontsize":16})

    handles, labels = axBR.legend_elements(prop="sizes", alpha=0.6, num=6)     
    labels = ["0", "1", "2", "4", "8"]
    legend = plt.legend(handles, labels, loc=(0,1.01), title="r")

    plt.subplots_adjust(top=0.925)
    plt.show()
    return





def is_pareto_efficient_dumb(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
    return is_efficient


def get_top_models_encoder(results_df):
    top_auroc = results_df.nlargest(3,"auroc")
    top_auprc = results_df.nlargest(3,"auprc")
    top_accso = results_df.nlargest(3,"acc_so")
    top_accfg = results_df.nlargest(3,"acc_fg")
    top_all = pd.concat([top_auroc, top_auprc, top_accso, top_accfg])
    print("we have {} top models".format(len(top_all)))

    numbers_all = results_df[["auroc","auprc","acc_fg","acc_so"]].values
    iseff_all = is_pareto_efficient_dumb(numbers_all)
    paretto_all = results_df[pd.Series(iseff_all).values]
    print("we have {} paretto models".format(len(paretto_all)))

    top_dict = {}
    top_dict["top"] = top_all
    top_dict["paretto"] = paretto_all
    return top_dict



def plot_top_explainers(top_model_dict_list, modelstr_list, other_results=None):
    fig, axs = plt.subplots(2, 2, figsize=(12,12))
    fig.suptitle('Classification vs wsSOD Performance', fontsize=24)
    SCALE = 15
    markers = {"plip":"o", "tile2vec":"x"}

    # from PLIP results on PCM
    # mean_max_accs_1_valid = 0.8800603070954025
    # mean_max_accs_1_so = 0.6903719562767577
    if other_results:
        mean_max_accs_1_valid = other_results["mean_max_accs_1_valid"]
        mean_max_accs_1_so = other_results["mean_max_accs_1_so"]

    for idx, top_model_dict in enumerate(top_model_dict_list):
        modelstr = modelstr_list[idx]
        marker = markers[modelstr]

        top_df = top_model_dict["top"]
        paretto_df = top_model_dict["paretto"]

        c_top = top_df["K"]
        s_top = (np.array(top_df["r"]) + 2) * SCALE
        c_paretto = paretto_df["K"]
        s_paretto = (np.array(paretto_df["r"]) + 2) * SCALE

        if marker != "x":
            ec = "k"
            alpha = 0.4
        else:
            ec = None
            alpha = 1
        
        viridis_big = mpl.colormaps['viridis']
        cmap = ListedColormap(viridis_big(np.linspace(0, 0.85, 128)))
        # cmap = "brg"
        if other_results:
            xmin_roc = np.min([np.min(paretto_df["auroc"]), np.min(top_df["auroc"])])
            xmax_roc = np.max([np.max(paretto_df["auroc"]), np.max(top_df["auroc"])])
            xmin_prc = np.min([np.min(paretto_df["auprc"]), np.min(top_df["auprc"])])
            xmax_prc = np.max([np.max(paretto_df["auprc"]), np.max(top_df["auprc"])])
            
        axs[0,0].scatter(top_df["auroc"], top_df["acc_fg"], alpha=alpha, c=c_top, edgecolors=ec, s=s_top, marker=marker, cmap=cmap)
        axs[0,0].scatter(paretto_df["auroc"], paretto_df["acc_fg"], alpha=alpha, c=c_paretto, edgecolors=ec, s=s_paretto, marker=marker, cmap=cmap)
        axs[0,0].set_xlabel('AUROC', fontsize=18)
        axs[0,0].set_ylabel('Mean-max Accuracy (FG)', fontsize=18)
       
        axs[0,1].scatter(top_df["auprc"], top_df["acc_fg"], alpha=alpha, c=c_top, edgecolors=ec, s=s_top, marker=marker, cmap=cmap)
        axs[0,1].scatter(paretto_df["auprc"], paretto_df["acc_fg"], alpha=alpha, c=c_paretto, edgecolors=ec, s=s_paretto, marker=marker, cmap=cmap)
        axs[0,1].set_xlabel('AUPRC', fontsize=18)
        # axs[0,1].set_ylabel('Mean-max Accuracy (FG)', fontsize=18)
 
        axs[1,0].scatter(top_df["auroc"], top_df["acc_so"], alpha=alpha, c=c_top, edgecolors=ec, s=s_top, marker=marker, cmap=cmap)
        axs[1,0].scatter(paretto_df["auroc"], paretto_df["acc_so"], alpha=alpha, c=c_paretto, edgecolors=ec, s=s_paretto, marker=marker, cmap=cmap)
        axs[1,0].set_xlabel('AUROC', fontsize=18)
        axs[1,0].set_ylabel('Mean-max Accuracy (SO)', fontsize=18)

        axBR = axs[1,1].scatter(top_df["auprc"], top_df["acc_so"], alpha=alpha, c=c_top, edgecolors=ec, s=s_top, marker=marker, cmap=cmap)
        axs[1,1].scatter(paretto_df["auprc"], paretto_df["acc_so"], alpha=alpha, c=c_paretto, edgecolors=ec, s=s_paretto, marker=marker, cmap=cmap)
        axs[1,1].set_xlabel('AUPRC', fontsize=18)
        
        if other_results:
            # other results
            axs[0,0].plot([xmin_roc, xmax_roc], [mean_max_accs_1_valid, mean_max_accs_1_valid], color="k", linestyle="--", linewidth=1)
            axs[0,1].plot([xmin_prc, xmax_prc], [mean_max_accs_1_valid, mean_max_accs_1_valid], color="k", linestyle="--", linewidth=1)
            axs[1,0].plot([xmin_roc, xmax_roc], [mean_max_accs_1_so, mean_max_accs_1_so], color="k", linestyle="--", linewidth=1)
            axs[1,1].plot([xmin_prc, xmax_prc], [mean_max_accs_1_so, mean_max_accs_1_so], color="k", linestyle="--", linewidth=1)

        # axs[1,1].set_ylabel('Mean-max Accuracy (SO)', fontsize=18)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(axBR, cax=cbar_ax)
    plt.text(0.32,9.3,"K", {"fontsize":16})

    handles, labels = axBR.legend_elements(prop="sizes", alpha=0.6, num=6)     
    labels = ["0", "1", "2", "4", "8"]
    legend = plt.legend(handles, labels, loc=(0,1.01), title="r")
    plt.subplots_adjust(top=0.925)
    plt.show()

def compute_sensitivity(ravel):
    """
    ravel: (tp, tn, fp, fn)
    """
    # tp, tn, fp, fn = ravel
    tn, tp, fn, fp = ravel
    if tp + fn == 0:
        return 1 # np.nan
    return tp / (tp + fn)

def compute_specificity(ravel):
    """
    ravel: (tp, tn, fp, fn)
    """
    # tp, tn, fp, fn = ravel
    tn, tp, fn, fp = ravel
    if tn + fp == 0:
        return 1 # np.nan
    return tn / (tn + fp)

def compute_dice(ravel):
    """
    ravel: (tp, tn, fp, fn)
    """
    # tp, tn, fp, fn = ravel
    tn, tp, fn, fp = ravel
    if ((2*tp) + fp + fn) == 0:
        return 1 # np.nan
    return (2*tp) / ((2*tp) + fp + fn)

def compute_jaccard(ravel):
    """
    ravel: (tp, tn, fp, fn)
    """
    # tp, tn, fp, fn = ravel
    tn, tp, fn, fp = ravel
    if (tp + fp + fn) == 0:
        return 1 # np.nan
    return tp / (tp + fp + fn)

def compute_accuracy(ravel):
    """
    ravel: (tp, tn, fp, fn)
    """
    # tp, tn, fp, fn = ravel
    tn, tp, fn, fp = ravel
    return (tp + tn) / (tp + tn + fp + fn)

def compute_precision(ravel):
    """
    ravel: (tp, tn, fp, fn)
    """
    # tp, tn, fp, fn = ravel
    tn, tp, fn, fp = ravel
    if (tp + fp) == 0:
        return 0
    return tp / (tp + fp)

def compute_recall(ravel):
    """
    ravel: (tp, tn, fp, fn)
    """
    return compute_sensitivity(ravel)

def compute_phi(ravel):
    # tp, tn, fp, fn = ravel
    # num = (tp * tn) - (fp * fn)
    # denom = np.sqrt((tp + fp)*(tp+fn)*(tn+fp)*(tn+fn))
    # if denom == 0:
    #     return 0 # see wikipedia
    # return num / denom

    # tp, tn, fp, fn = ravel
    tn, tp, fn, fp = ravel
    # C = np.array([[tn, fp], [fn, tp]])
    C = np.array([[tp, fn], [fp, tn]])

    # https://github.com/scikit-learn/scikit-learn/blob/364c77e04/sklearn/metrics/_classification.py#L848
    t_sum = C.sum(axis=1, dtype=np.float64)
    p_sum = C.sum(axis=0, dtype=np.float64)
    n_correct = np.trace(C, dtype=np.float64)
    n_samples = p_sum.sum()
    cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
    cov_ypyp = n_samples**2 - np.dot(p_sum, p_sum)
    cov_ytyt = n_samples**2 - np.dot(t_sum, t_sum)

    if cov_ypyp * cov_ytyt == 0:
        return 0.0
    else:
        return cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)

def compute_threats(ravel):
    # tp, tn, fp, fn = ravel
    tn, tp, fn, fp = ravel
    return tp / (tp + fn + fp)


def compute_balanced_acc(ravel, adjusted=False):
    # tp, tn, fp, fn = ravel
    # tpr = tp / (tp + fn)
    # tnr = tn / (tn + fp)
    # return (tpr + tnr) / 2
    # https://github.com/scikit-learn/scikit-learn/blob/364c77e04/sklearn/metrics/_classification.py#L2111
    # tp, tn, fp, fn = ravel
    tn, tp, fn, fp = ravel
    # C = np.array([[tn, fp], [fn, tp]])
    # C = np.array([[tp, tn], [fp, tn]]) 
    C = np.array([[tp, fn], [fp, tn]])

    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.diag(C) / C.sum(axis=1)
    score = np.mean(per_class)
    if adjusted:
        n_classes = len(per_class)
        chance = 1 / n_classes
        score -= chance
        score /= 1 - chance
    return score

def compute_prevalence(ravel):
    # tp, tn, fp, fn = ravel
    tn, tp, fn, fp = ravel
    return (tp + fn) / (tp + tn + fp + fn)

def compute_top_cm_metric(results_dict, k_config, metric, mode="thresh_max", map_type="lofi"):
    mean_metric_stat = {}
    trajectory_dict = {}

    threshs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    # print(k_config)

    if k_config == None:
        dict_to_scan = results_dict
    else:
        dict_to_scan = results_dict[k_config]

    for k_stat in dict_to_scan.keys(): # stat class: e.g. ravels
        if k_stat not in ["ravels_1_valid", "ravels_1_static", "ravels_1_static_back"]:
             continue
        if (not k_stat.startswith("ravels")) or ("full" in k_stat):
            continue
        if k_config == None:
            cms_dict = results_dict[k_stat]
        else:
            cms_dict = results_dict[k_config][k_stat] # confusion matrices
        if len(cms_dict) == 0:
            continue
        # main computation per config
        metrics = []
        # if map_type == "attn":
        #     pdb.set_trace()

        # for idx in range(3): # for diff indices of attn
        # if k_stat == "ravels_1_valid" and map_type == "attn":
        #     k_stat = k_stat + "_" + str(idx)

        for k_id in cms_dict.keys():
            cms = cms_dict[k_id]
            senss, specs, specs1 = [], [], [] # sens, spec, 1-spec
            dices, jaccards = [], []
            accs, baccs, precisions, recalls = [], [], [], []
            phis, threats, prevs = [], [], []
            if map_type == "lofi":
                cms_unpacked = cms
            elif map_type == "pcm":
                if k_stat == "ravels_1_valid":
                    cms_unpacked = cms[0]
                else:
                    cms_unpacked = [cms]
            elif map_type == "attn":
                if k_stat == "ravels_1_valid":
                    cms_unpacked = cms[2] # 0 is full feature scaling, 1 is ReLU only, 2 is abs val
                else:
                    cms_unpacked = [cms]

            for confusion_mat in cms_unpacked:
                if (k_stat.startswith("ravels")) and ("static" in k_stat):
                    cm = confusion_mat #[c for c in cms_unpacked]
                else:
                    cm = confusion_mat

                # if map_type == "pcm":
                #     pdb.set_trace()
                
                # if type(cm) is list:
                #     pdb.set_trace()
                #     cm_list = cm
                # else:
                #     cm_list = [cm]

                # senss_temp, threats_temp, specs_temp, specs1_temp = [],[],[],[]
                # dices_temp, jaccards_temp, accs_temp, baccs_temp = [], [], [], []
                # recalls_temp, precisions_temp, phis_temp, prevs_temp = [], [], [], []
                # for cm in cm_list:
                senss.append(compute_sensitivity(cm)) 
                threats.append(compute_threats(cm)) 
                specs.append(compute_specificity(cm))
                specs1.append(1-compute_specificity(cm)) 
                dices.append(compute_dice(cm))
                jaccards.append(compute_jaccard(cm))
                accs.append(compute_accuracy(cm))
                baccs.append(compute_balanced_acc(cm))
                recalls.append(compute_recall(cm))
                precisions.append(compute_precision(cm))
                phis.append(compute_phi(cm))
                prevs.append(compute_prevalence(cm))
                if (k_stat.startswith("ravels")) and ("static" in k_stat):
                    break
                # senss.append(np.max(senss_temp))
                # threats.append(np.max(threats_temp))
                # specs.append(np.max(specs_temp))
                # specs1.append(np.max(specs1_temp))
                # dices.append(np.max(dices_temp))
                # jaccards.append(np.max(jaccards_temp))
                # accs.append(np.max(accs_temp))
                # baccs.append(np.max(baccs_temp))
                # recalls.append(np.max(recalls_temp))
                # precisions.append(np.max(precisions_temp))
                # phis.append(np.max(phis_temp))
                # prevs.append(np.max(prevs_temp))
                # if (k_stat.startswith("ravels")) and ("static" in k_stat):
                #     break
                
            # if len(to_plot_x) == 0 or len(to_plot_y) == 0:
            #     continue
            # auroc = auc(to_plot_x, to_plot_y)
            if metric == "sensitivity":
                metric_list = senss
            elif metric == "threat_score":
                metric_list = threats
            elif metric == "specificity":
                metric_list = specs
            elif metric == "dice":
                metric_list = dices
            elif metric == "jaccard":
                metric_list = jaccards
            elif metric == "accuracy":
                metric_list = accs
            elif metric == "balanced_accuracy":
                metric_list = baccs
            elif metric == "precision":
                metric_list = precisions
            elif metric == "recall":
                metric_list = recalls
            elif metric == "prevalence":
                metric_list = prevs
            elif metric == "phi":
                if mode == "auc":
                    metric_list = np.abs(phis)
                else:
                    metric_list = phis
            else:
                print("metric not implemented")
                return

            if mode == "auc":
                score = auc(threshs, metric_list)
            elif mode == "mean":
                score = np.mean(metric_list)
            elif mode == "median":
                score = np.median(metric_list)
            elif mode == "max":
                score = np.max(metric_list)
            elif mode == "thresh_max":
                score = metric_list # need to conatenate
            else:
                print("mode not implemented")
                return
            metrics.append(score)

        if mode == "thresh_max":
            # turn into array (rows=samples, cols=threshs)
            # then take column means over samples
            # then take max over those column means
            # if k_config == None:
            #     print(k_stat)
            # if map_type == "pcm":
            #     pdb.set_trace()
            metrics = np.array(metrics)
            # if k_config == None:
            #     print(metrics.shape)
            # if (k_stat.startswith("ravels")) and ("static" in k_stat):
            #     pdb.set_trace()
            metrics = np.nanmean(metrics, axis=0)
            # if k_config == None and (k_stat in ["ravels_1_static", "ravels_1_valid"]):
            #     print(metrics)
                # pdb.set_trace()
            trajectory_dict[k_stat] = metrics
            metrics = np.max(metrics)
            mean_metric_stat[k_stat] = metrics
        else:
            mean_metric_stat[k_stat] = np.nanmean(metrics)
    return mean_metric_stat, trajectory_dict


def compute_top_cm_metric_lofi(results_dict, metric, mode="thresh_max"):
    """
    mode: mean, median, max, auc (w.r.t. thresh)
    """
    mean_metric = {}
    trajectories = {}
    for k_config in results_dict.keys(): # lofi config
        mean_metric_stat, trajectory_dict = compute_top_cm_metric(results_dict, k_config, metric, mode)
        mean_metric[k_config] = mean_metric_stat
        trajectories[k_config] = trajectory_dict
    return mean_metric, trajectories


def get_max_performances(results_list, elastic_list, encoder_list, other_results, stat="precision", type="thresh_max"):
    bars = {}
    top_configs = []
    if stat == "acc_fg":
        stat_alias = "mean_max_accs_1_valid"

    for idx, e in enumerate(encoder_list):
        # if e != "vit_iid":
        #     continue

        results = utils.deserialize(results_list[idx])
        elastic = utils.deserialize(elastic_list[idx])

        if stat == "acc_fg":
            results_df = grab_stats(results, elastic)
            top_lofi = results_df.nlargest(1,stat)
            bars[e+"-"+"lofi"] = [top_lofi[stat].values[0], e, "lofi"]
            if e == "vit_iid" or e == "clip" or e == "plip":
                pcm_results = other_results[e+"_pcm"]
                top_pcm = pcm_results[stat_alias]
                bars[e+"-"+"pcm"] = [top_pcm, e, "pcm"]
            if e == "vit_iid":
                attn_results = other_results[e+"_sam"]
                top_attn = attn_results[stat_alias]
                bars[e+"-"+"sam"] = [top_attn, e, "sam"]
        else:
            mean_metric, trajectories = compute_top_cm_metric_lofi(results, metric=stat, mode=type)
            vals = []
            for k_config in mean_metric.keys():
                for k_stat in mean_metric[k_config].keys():
                    if k_stat == 'ravels_1_valid':
                        vals.append([k_config, mean_metric[k_config][k_stat]])
            top_hit = max(vals, key=lambda x: x[1])
            top_lofi = top_hit[1]
            top_config = top_hit[0]
            print("TOP CONFIG:", top_config)
            top_configs.append(top_config)

            bars[e+"-"+"lofi"] = [top_lofi, e, "lofi"]
            print("")
            print("MULTI-THREHOLDING:")
            print(e, "=", top_lofi, "(k2)")

            if e == "vit_iid" or e == "clip" or e == "plip":
                pcm_results = other_results[e+"_pcm"]
                top_pcm, trajectory_pcm = compute_top_cm_metric(pcm_results, k_config=None, metric=stat, mode=type, map_type="pcm")
                valid_pcm = top_pcm["ravels_1_valid"]
                bars[e+"-"+"pcm"] = [valid_pcm, e, "pcm"]
                print(e, "=", valid_pcm, "(pcm)")
            if e == "vit_iid":
                attn_results = other_results[e+"_sam"]
                top_attn, trajectory_attn = compute_top_cm_metric(attn_results, k_config=None, metric=stat, mode=type, map_type="attn")
                valid_attn = top_attn["ravels_1_valid"]
                bars[e+"-"+"sam"] = [valid_attn, e, "sam"]
                print(e, "=", valid_attn, "(attn)")
      
            # plot line plot to show thresholding sensitivity
            k2_deltas = []
            sam_delta, pcm_delta = None, None
            plot_dict = {}
            for k_config in trajectories.keys():
                points = trajectories[k_config]["ravels_1_valid"]
                plot_dict[k_config] = ["lofi"] + [p for p in points]
                k2_deltas.append(np.max(points) - np.min(points))

            if e == "vit_iid" or e == "clip" or e == "plip":
                points = trajectory_pcm["ravels_1_valid"]
                plot_dict["pcm"] = ["pcm"] + [p for p in points]
                pcm_delta = np.max(points) - np.min(points)
                # points = trajectory_pcm["ravels_1_static"]
                # plot_dict["tpcm"] = ["tpcm"] + [p for p in points] * len(threshs)
                # points = trajectory_pcm["ravels_1_static_back"]
                # plot_dict["tpcm_b"] = ["tpcm_b"] + [p for p in points] * len(threshs)
            if e == "vit_iid":
                points = trajectory_attn["ravels_1_valid"]
                plot_dict["sam"] = ["sam"] + [p for p in points]
                sam_delta = np.max(points) - np.min(points)
                # points = trajectory_attn["ravels_1_static"]
                # plot_dict["tsam"] = ["tsam"] + [p for p in points] * len(threshs)
                # points = trajectory_attn["ravels_1_static_back"]
                # plot_dict["tsam_b"] = ["tsam_b"] + [p for p in points] * len(threshs)

            print()
            print("k2 min delta:", np.min(k2_deltas))
            print("k2 mean delta:", np.mean(k2_deltas))
            print("k2 median delta:", np.median(k2_deltas))
            print("k2 max delta:", np.max(k2_deltas))
            print()
            print("pcm delta:", pcm_delta)
            print("sam delta:", sam_delta)


            print()
            print("ADAPTIVE THRESHOLDING")
            if e == "vit_iid" or e == "clip" or e == "plip":
                thresh_pcm = top_pcm["ravels_1_static"]
                thresh_pcm_b = top_pcm["ravels_1_static_back"]
                print(e, "=", thresh_pcm, "(pcm-thresh forward)")
                print(e, "=", thresh_pcm_b, "(pcm-thresh backward)")
                bars[e+"-"+"pcm-thresh"] = [thresh_pcm, e, "pcm"]
            if e == "vit_iid":
                thresh_sam = top_attn["ravels_1_static"]
                thresh_sam_b = top_attn["ravels_1_static_back"]
                print(e, "=", thresh_sam, "(attn-thresh forward)")  
                print(e, "=", thresh_sam_b, "(attn-thresh backward)")  
                bars[e+"-"+"sam-thesh"] = [thresh_sam, e, "sam"]

            # plot trajectories
            #------------------
            cols_names = ["class"] + ["v_"+str(t) for t in threshs]
            lines_df = pd.DataFrame.from_dict(plot_dict, orient='index', columns=cols_names)
            melt_df = pd.melt(lines_df.reset_index(), id_vars=["index", "class"], value_vars=["v_"+str(t) for t in threshs])
            melt_df["t"] = melt_df["variable"].apply(lambda x: float(x.split("_")[1]))
            melt_df["alpha"] = np.where(melt_df["class"] == "lofi", 0.2, 1)
            melt_df["size"] = np.where(melt_df["class"] == "lofi", 1, 300)
            
            alphas = melt_df.alpha.sort_values().unique()
            plt.figure()
            sns.color_palette("tab10")
            for idx,alpha in enumerate(alphas):
                if idx == 0:
                    ax = sns.lineplot(data=melt_df[melt_df.alpha == alpha], x="t", y="value", color="gray", style="index", dashes=False, alpha=alpha, legend=False, lw=1)
                else:
                    ax = sns.lineplot(data=melt_df[melt_df.alpha == alpha], x="t", y="value", hue="class", style="index", dashes=False, alpha=alpha, legend=False, lw=2) # size="size"
                ax.margins(x=0, y=0.01, tight=True)
            sns.despine()

            if "_" in e:
                e = e.split("_")[0]
            if e in ["clip", "plip"]:
                e = e.upper()
            elif e == "vit":
                e = "ViT"
            if "_" in stat:
                stat_str = stat.replace("_", " ")
            else:
                stat_str = stat
            if "phi" in stat:
                stat_str = "MCC"
            else:
                stat_str = stat_str.title()

            plt.title(e + " " + stat_str + " over thresholds", fontsize=18)
            plt.ylabel(stat_str, fontsize=15)
            plt.xlabel("Threshold", fontsize=15)
            plt.tight_layout()
            plt.show()
    return bars, top_configs
    

def plot_top_performances(bars):
    width = 1      
    groupgap=1
    df = pd.DataFrame.from_dict(bars, orient='index', columns=['score', 'encoder', 'map_type'])
    # pdb.set_trace()
    print(df)
    import matplotlib
    cmap = matplotlib.cm.get_cmap('Accent')

    y_tile2vec = df[df["encoder"] == "tile2vec"]["score"].values
    y_vit_iid = df[df["encoder"] == "vit_iid"]["score"].values
    y_clip = df[df["encoder"] == "clip"]["score"].values
    y_plip = df[df["encoder"] == "plip"]["score"].values

    x1 = np.arange(len(y_tile2vec))
    x2 = np.arange(len(y_vit_iid))+groupgap+len(y_tile2vec)
    x3 = np.arange(len(y_clip))+groupgap+len(y_tile2vec)+groupgap+len(y_vit_iid)
    x4 = np.arange(len(y_plip))+groupgap+len(y_tile2vec)+groupgap+len(y_vit_iid)+groupgap+len(y_clip)

    ind = np.concatenate((x1,x2,x3,x4))
    _, ax = plt.subplots()
    rects1 = ax.bar(x1, y_tile2vec, width, color=cmap(0), edgecolor="black" , label="USL")
    rects2 = ax.bar(x2, y_vit_iid, width, color=cmap(0.25), edgecolor="black", label="WSL")
    rects3 = ax.bar(x3, y_clip, width, color=cmap(0.5), edgecolor="black", label="VLM-general")
    rects4 = ax.bar(x4, y_plip, width, color=cmap(1), edgecolor="black", label="VLM-domain")

    ax.set_title('Top performances over varying thresholds',fontsize=14)
    ax.set_ylabel('MMA-FG',fontsize=14)
    ax.legend(loc="lower right")
    ax.set_xticks(ind)
    ax.set_xticklabels(('LOFI', 'LOFI','Prob','Attn', 'LOFI','Prob', 'LOFI','Prob'),fontsize=10)
    plt.show()



def display_adaptive_continuous_structure_eval_pcm(other_results):
    for k in other_results.keys():
        results = other_results[k]
        auroc, auprc, ap = mean_stat_from_dict(results["sod_auroc"]), mean_stat_from_dict(results["sod_auprc"]), mean_stat_from_dict(results["sod_ap"])
        mod_forward, mod_backward = mean_stat_from_dict(results["mod_forward"]), mean_stat_from_dict(results["mod_backward"])
        mcc_forward, mcc_backward = mean_stat_from_dict(results["mcc_forward"]), mean_stat_from_dict(results["mcc_backward"])
        prec_forward, prec_backward = mean_stat_from_dict(results["prec_forward"]), mean_stat_from_dict(results["prec_backward"]) 
        ba_forward, ba_backward = mean_stat_from_dict(results["ba_forward"]), mean_stat_from_dict(results["ba_backward"])
        print()
        print(k)
        print("-----------")
        print("CONTINUOUS THRESHOLDING:")
        print("auroc, auprc, ap:", auroc, auprc, ap)
        print("ADAPTIVE THRESHOLDING:")
        print("mcc f/b:", mcc_forward, mcc_backward)
        print("prec f/b:", prec_forward, prec_backward)
        print("ba f/b:", ba_forward, ba_backward)
        print("STRUCTURE:")
        print("mod f/b:", mod_forward, mod_backward)


