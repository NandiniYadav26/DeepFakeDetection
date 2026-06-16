import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve
import torch
import numpy as np
import argparse
from tqdm import tqdm
import math
import yaml
from utils import check_correct, aggregate_attentions, save_attention_plots, count_parameters
from torch.optim.lr_scheduler import LambdaLR
from datetime import datetime, timedelta
from statistics import mean
import collections
import os
import json
from sklearn import metrics
from sklearn.metrics import f1_score
from itertools import chain
import random
from einops import rearrange, reduce
import pandas as pd
from os import cpu_count
from multiprocessing.pool import Pool
from functools import partial
from multiprocessing import Manager
from progress.bar import ChargingBar
from torch.optim import lr_scheduler
from deepfakes_dataset import DeepFakesDataset
from models.size_invariant_timesformer import SizeInvariantTimeSformer
from models.efficientnet.efficientnet_pytorch import EfficientNet
from torch.utils.tensorboard import SummaryWriter
import torch_optimizer as optim
from timm.scheduler.cosine_lr import CosineLRScheduler





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test_list_file', default="../../datasets/FaceForensics/faces/test.csv", type=str,
                        help='Test List txt file path)')  
    parser.add_argument('--data_path', default="../../datasets/FaceForensics/faces", type=str,
                        help='Path to the dataset converted into identities.')
    parser.add_argument('--video_path', default="../../datasets/FaceForensics/videos", type=str,
                        help='Path to the dataset original videos (.mp4 files).')
    parser.add_argument('--deepfake_methods', nargs='*', required=False,
                        help="For FaceForensics dataset, filter some deepfake methods for partial training.")
    parser.add_argument('--workers', default=0, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--random_state', default=42, type=int,
                        help='Random state value')
    parser.add_argument('--model_weights', type=str,
                        help='Model weights.')
    parser.add_argument('--extractor_model', type=int, default=0, 
                        help="Which model use for features extraction (0: EfficientNet).")
    parser.add_argument('--extractor_weights', default='ImageNet', type=str,
                        help='Path to extractor weights or "imagenet".')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='ID of GPU to be used.')
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for training (default: all).")                  
    parser.add_argument('--only_multiidentity', default=False, action="store_true",
                        help='Use only multiidentity videos.')
    parser.add_argument('--config', type=str, 
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--model', type=int, 
                        help="Which model to use. (1: Size Invariant TimeSformer).")
    parser.add_argument('--identities_ordering', type=int,  default = 0,
                        help="Which ordering rule to use. (0: Size-based | 1: Frequency-based | 2: Random).")
    parser.add_argument('--save_attentions', default=False, action="store_true",
                        help='Save attentions plots.')
    opt = parser.parse_args()
    
    print(opt)
    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)


    # Check for integrity
    if config['model']['num-frames'] != 8 and config['model']['num-frames'] != 16:
        raise Exception("Invalid number of frames.")
        
        
   
    random.seed(opt.random_state)
    torch.manual_seed(opt.random_state)
    np.random.seed(opt.random_state)
   
    # Load required weights for feature extractor
    if opt.extractor_model == 0:  # EfficientNet-B0
        features_extractor = EfficientNet.from_name('efficientnet-b0')

        extractor_path = '"outputs/models"/Extractor_checkpoint16'

        if os.path.exists(extractor_path):
            features_extractor.load_state_dict(
                torch.load(extractor_path, map_location=device)
            )
            print("Loaded trained extractor weights.")
        else:
            print("Extractor checkpoint not found, using ImageNet weights.")
    else:
        features_extractor = None

    # Init the required model
    if opt.model == 1:
        model = SizeInvariantTimeSformer(config=config, require_attention=True)
        num_patches = config['model']['num-patches']



    if os.path.exists(opt.model_weights):
        checkpoint = torch.load(opt.model_weights, map_location=device)

        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)

        print("Checkpoint loaded successfully.")
    else:
        raise Exception("No checkpoint loaded for the model.")

    loss_fn = torch.nn.BCEWithLogitsLoss().to(device)

    # Move into GPU
    if features_extractor != None:
        features_extractor = features_extractor.to(device)   
        features_extractor.eval()
        print("Extractor Parameters: ", count_parameters(features_extractor))  
    print("Model Parameters: ", count_parameters(model)) 
    model = model.to(device)
    model.eval()
       
    # Read all the paths and initialize data loaders for train and validation
    paths = []
    col_names = ["video", "label"]
    df_test = pd.read_csv(opt.test_list_file, sep=' ', names=col_names)
    df_test = df_test.sample(frac=1, random_state=opt.random_state).reset_index(drop=True)

    
    # Filter out deepfake methods if requested for ForgeryNet
    
    
    # Filter out non-multi-identity videos if requested
    if opt.only_multiidentity:
        indexes_to_drop = []
        for index, row in df_test.iterrows():
            video_path = os.path.join(opt.data_path, row['video']) 
            folders = os.listdir(video_path)
            if len(folders) < 2:
                indexes_to_drop.append(index)
            else:
                counter = 0
                for folder in folders:
                    if os.path.isdir(os.path.join(opt.data_path, row['video'], folder)):
                        counter += 1
                if counter < 2:
                    indexes_to_drop.append(index)
                
        df_test.drop(df_test.index[indexes_to_drop], inplace=True)
            
    # Split videos and labels and reduce to the required number of videos
    test_videos = df_test['video'].tolist()
    test_labels = df_test['label'].tolist()
    multiclass_labels = [0] * len(df_test)
    class_counter = collections.Counter(test_labels)

    if opt.max_videos > -1:
        test_videos = test_videos[:opt.max_videos]
        test_labels = test_labels[:opt.max_videos]
    
    test_samples = len(test_videos)

    # Create the data loaders 
    test_dataset = DeepFakesDataset(test_videos, test_labels, multiclass_labels = multiclass_labels, image_size=config['model']['image-size'], data_path=opt.data_path, video_path=opt.video_path, num_frames=config['model']['num-frames'], num_patches=num_patches, max_identities=config['model']['max-identities'], enable_identity_attention=config['model']['enable-identity-attention'], identities_ordering = opt.identities_ordering, mode='test')
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=config['test']['bs'], shuffle=False, sampler=None,
                                    batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None, prefetch_factor=2,
                                    persistent_workers=False)

    # Print some useful statistics
    print("Test videos:", test_samples)
    print("__TEST STATS__")
    test_counters = collections.Counter(test_labels)
    print(test_counters)

    # Init variables
    total_test_loss = 0
    test_correct = 0
    test_positive = 0
    test_negative = 0
    test_counter = 0

    
    bar = ChargingBar('PREDICT', max=(len(test_dl)))
    preds = []
    videos_errors = []

    # Test loop
    for index, (videos, size_embeddings, masks, identities_masks, positions, tokens_per_identity, labels, multiclass_labels, video_ids) in enumerate(test_dl):
        b, f, h, w, c = videos.shape
        labels = labels.unsqueeze(1).float().to(device)
        identities_masks = identities_masks.to(device)
        masks = masks.to(device)
        positions = positions.to(device)

        with torch.no_grad():
            
            if opt.model != 2: # Use the features extractor
                videos = rearrange(videos, "b f h w c -> (b f) c h w")
                videos = videos.to(device)
                
                features = features_extractor(videos)  
                if opt.model == 0: 
                    test_pred = model(features)
                    test_pred = torch.mean(test_pred.reshape(-1, config["model"]["num-frames"]), axis=1).unsqueeze(1)
                elif opt.model == 1:
                    features = rearrange(features, '(b f) c h w -> b f c h w', b = b, f = f)
                    test_pred, attentions = model(features, mask=masks, size_embedding=size_embeddings, identities_mask=identities_masks, positions=positions)
                    if opt.save_attentions:
                        identity_names = [row[0] for row in tokens_per_identity]
                        frames_per_identity = [int(row[1] / config["model"]["num-patches"]) for row in tokens_per_identity]
                        
                        aggregated_attentions, identity_attentions = aggregate_attentions(attentions, config['model']['heads'], config['model']['num-frames'], frames_per_identity)
                        
                        save_attention_plots(aggregated_attentions, identity_names, frames_per_identity, config['model']['num-frames'], video_ids[0])
            
                    
    


        
        test_loss = loss_fn(test_pred, labels)
        total_test_loss += round(test_loss.item(), 2)
        pred_labels = (torch.sigmoid(test_pred) > 0.5).int()

        corrects = (pred_labels == labels.int()).sum().item()
        positive_class = (pred_labels == 1).sum().item()
        negative_class = (pred_labels == 0).sum().item()

        batch_errors = []
        videos_errors.extend(batch_errors)
        test_correct += corrects
        test_positive += positive_class
        test_counter += 1
        test_negative += negative_class
        preds.extend(test_pred.detach().cpu())
        bar.next()
    
    preds = torch.sigmoid(torch.stack(preds)).cpu().numpy().flatten()

    fpr, tpr, th = metrics.roc_curve(test_labels, preds)
    auc = metrics.auc(fpr, tpr)

    f1 = f1_score(test_labels, (preds > 0.5).astype(int))
    bar.finish()
    total_test_loss /= test_counter
    test_correct /= test_samples
    print("Videos errors", videos_errors)
    print(str(opt.model_weights) + " test loss:" +
            str(total_test_loss) + " f1 score: " + str(f1) + " test accuracy:" + str(test_correct) + " test_0s:" + str(test_negative) + "/" + str(test_counters[0]) + " test_1s:" + str(test_positive) + "/" + str(test_counters[1]) + " AUC " + str(auc))
    # Convert predictions to labels
    pred_labels = (preds > 0.5).astype(int)
    cm = confusion_matrix(test_labels, pred_labels)

    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()

    classes = ["Real", "Fake"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.savefig("confusion_matrix.png")
    plt.close()



    fpr, tpr, _ = metrics.roc_curve(test_labels, preds)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1])
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig("roc_curve.png")
    plt.close()

    precision, recall, _ = precision_recall_curve(test_labels, preds)

    plt.figure()
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig("precision_recall_curve.png")
    plt.close()

    print("Plots saved:")
    print("confusion_matrix.png")
    print("roc_curve.png")
    print("precision_recall_curve.png") 