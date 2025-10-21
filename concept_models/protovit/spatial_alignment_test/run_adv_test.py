import torch 
import torch.utils.data
import os
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as datasets
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import re
from helpers import makedir,find_high_activation_crop
import model
import train_and_test as tnt
from pathlib import Path
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function
from preprocess import undo_preprocess_input_function
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from typing import List, Optional
import copy
import pickle
import json
import argparse
from collections import defaultdict
from typing import List
import pandas as pd
import numpy as np
from torch.utils.data import Subset
from tqdm import tqdm

#######################################################################
# put help function modules on top of everything 
###############This chunck is for function run model on data and batch 
#### adversarial model wrapper 
#### this chunck is for adversarial modification
from typing import List, Dict

import cv2
import torch
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from torch import nn
import numpy as np
from preprocess import mean, std
from settings import img_size

def retrieve_heatmap(model, sample):
    '''
    We follow the definition of PLC in location misalignment to compare the location change in the top 10 percentile 
    of the activation area 
    '''
    _, dist_all = model.subpatch_dist(sample)# bsz, 2000, 196, 4
    percentiles = torch.quantile(dist_all.detach().cpu(), 0.9, dim=2, keepdim=True)
    greater_than_percentile = dist_all.detach().cpu() >= percentiles # simulated heatmap with TF 
    #2indices = torch.nonzero(greater_than_percentile)
    return greater_than_percentile#.numpy()

def get_all_class_proto_low_activation_bbox_mask(
        proto_nums: List[np.ndarray],
        activations: np.ndarray,
        bbox: np.ndarray) -> np.ndarray:
    """
    Get a mask that has 0 on pixels within high activation bounding box of any of the ground truth prototypes.
    :param proto_nums: list of prototype numbers to attack for each image
    :param activations: a tensor of prototype activations over image patches of shape [B, P, Wp, Hp]
    :param epsilon_pixels: number of border pixels to add to the high activation bounding box
    """
    assert len(proto_nums) == activations.shape[0]
    proto_mask = np.ones((activations.shape[0], 1, img_size, img_size), dtype=np.float32)
    for sample_i in range(activations.shape[0]):
        min_indices = bbox[sample_i]
        #print('indices shape:', min_indices.shape)
        proto_bound_boxes = np.full(shape=[5, 4],
                                            fill_value=-1)
        for proto_num in proto_nums[sample_i]:
            #proto_patch_activation = activations[sample_i, proto_num, :, :]
            min_j_indice = min_indices[proto_num] # 5, 4 
            #print('indice j shape:',min_j_indice.shape)
            min_j_indice = np.unravel_index(min_j_indice.astype(int), (14,14))
            grid_width = 16
            for k in range(4):
                fmap_height_start_index_k = min_j_indice[0][k]* 1
                fmap_height_end_index_k = fmap_height_start_index_k + 1
                fmap_width_start_index_k = min_j_indice[1][k] * 1
                fmap_width_end_index_k = fmap_width_start_index_k + 1
                bound_idx_k = np.array([[fmap_height_start_index_k, fmap_height_end_index_k],
                [fmap_width_start_index_k, fmap_width_end_index_k]])
                pix_bound_k= bound_idx_k*grid_width
                proto_bound_boxes[0] = -1
                proto_bound_boxes[1,k] = pix_bound_k[0][0]
                proto_bound_boxes[2,k] = pix_bound_k[0][1]
                proto_bound_boxes[3,k] = pix_bound_k[1][0]
                proto_bound_boxes[4,k] = pix_bound_k[1][1]
                bbox_height_start_k = proto_bound_boxes[1,k]
                bbox_height_end_k = proto_bound_boxes[2,k]
                bbox_width_start_k = proto_bound_boxes[3,k]
                bbox_width_end_k = proto_bound_boxes[4,k]
                proto_mask[sample_i, :, bbox_height_start_k:bbox_height_end_k-1, bbox_width_start_k:bbox_width_end_k-1] = 0 

    return proto_mask 


def attack_images_target_class_prototypes(
        model: nn.Module,
        img: torch.tensor,
        activations: np.ndarray,
        attack_type: str,
        cls: np.ndarray,
        bb_box: np.ndarray,
        epsilon: float = 0.1,
        epsilon_iter: float = 0.01,
        nb_iter: int = 20,
        ) -> Dict:
    """
    Adversarially attack activations of prototypes of the ground truth class for a given image.
    We will exclude from the region of the attack the high activation bounding box of the ground truth class prototypes.
    :param model: PPNet model
    :param img: a batch of images to attack [B, C, W, H]
    :param activations: a numpy array with prototype activations over image patches of shape [B, P, Wp, Hp]
    :param attack_type: type of attack, in terms of the attacked prototypes
    :param cls: a vector ground truth classes of the images
    :param epsilon: maximum perturbation of the adversarial attack
    :param epsilon_iter: maximum perturbation of the adversarial attack within one iteration
    :param nb_iter: number of iterations of the adversarial attack
    :return: a dictionary contained the modified images and the mask of the region of the attack
    """
    proto_cls_identity = model.prototype_class_identity.cpu().detach().numpy()
    cls_proto_nums = [np.argwhere(proto_cls_identity[:, c] == 1).flatten() for c in cls]
    if attack_type == 'gt_protos':
        proto_nums = cls_proto_nums
    elif attack_type == 'top_proto':
        proto_nums = []
        for sample_act in activations:
            proto_max_act = np.max(sample_act.reshape(sample_act.shape[0], -1), axis=-1)
            proto_max_act = np.argmax(proto_max_act)
            proto_nums.append(np.asarray([int(proto_max_act)]))
    else:
        raise ValueError(attack_type)
    
    mask = get_all_class_proto_low_activation_bbox_mask(
        proto_nums=proto_nums,
        bbox = bb_box,
        activations=activations
    )
    #print(mask.shape)
    mask = torch.tensor(mask, device=img.device)
    ##############

    # need mask in wrapper 
    img_modified, activations_before, activations_after = [], [], []
    for sample_i in range(img.shape[0]):
        sample_img = img[sample_i].unsqueeze(0)
        sample_proto_nums = proto_nums[sample_i]
        sample_mask = mask[sample_i].unsqueeze(0)
        wrapper = PPNetAdversarialWrapper(model=model, img=sample_img, proto_nums=sample_proto_nums, mask=sample_mask)
        #break
    #return None
        sample_modified = projected_gradient_descent(
            model_fn=wrapper,
            x=sample_img,
            eps=epsilon,
            eps_iter=epsilon_iter,
            nb_iter=nb_iter,
            norm=np.inf,
        )
        img_modified.append(sample_modified)
        activations_before.append(np.clip(wrapper.initial_activation, a_min=0.0, a_max=None))
        activations_after.append(np.clip(wrapper.final_activation, a_min=0.0, a_max=None))

    img_modified = torch.cat(img_modified, dim=0)
    img_modified = img_modified * mask + img * (1 - mask)
    activations_before = np.concatenate(activations_before, axis=0)
    activations_after = np.concatenate(activations_after, axis=0)
    img_modified_numpy = img_modified.clone().cpu().detach().numpy()
    for d in range(3):
        img_modified_numpy[:, d] = (img_modified_numpy[:, d] * std[d] + mean[d])
    img_modified_numpy = img_modified_numpy.clip(0, 1)
    return {
        'img_modified_numpy': img_modified_numpy,
        'img_modified_tensor': img_modified.detach(),
        'mask': mask.cpu().detach().numpy(),
        'proto_nums': proto_nums,
        'activations_before': activations_before,
        'activations_after': activations_after,
        'cls_proto_nums': cls_proto_nums,
    }



class PPNetAdversarialWrapper(nn.Module):
    """
    Wrapper over the PPNet model that allows for adversarially attack activations of selected prototypes,
    over a selected image, and with a selected mask.
    The attack aims to minimize the activation of the selected prototypes, while modifying only the masked pixels.
    """

    def __init__(
            self,
            model: nn.Module,
            img: torch.Tensor,
            proto_nums: np.ndarray,
            mask: torch.Tensor,):
        """
        :param model: PPNet model
        :param img: an image to attack
        :param proto_nums: vector of prototype numbers to attack
        :param mask: binary mask, 1 for pixels that can be modified, 0 for pixels that cannot be modified
        """
        super(PPNetAdversarialWrapper, self).__init__()
        self.model = model
        self.proto_nums = proto_nums
        self.mask = mask

        # ensure that we do not propagate gradients through the image and the mask
        self.img = img.clone()
        self.img.requires_grad = False
        # self.mask = torch.tensor(mask, device=self.img.device)
        self.mask.requires_grad = False

        self.initial_activation, self.final_activation = None, None

    def forward(self, x):
        # 'x' can be modified by cleverhans
        # 'x2' is the actual output image. We use masking to ensure that cleverhans can affect only the masked pixels.
        x2 = x * self.mask + self.img * (1 - self.mask)
        max_activation, _, _ = self.model.greedy_distance(x2)
        #print(max_activation.shape)
        #max_activation_np= max_activation.clone().cpu().detach().numpy()
        self.final_activation = max_activation[0].clone().cpu().detach().numpy()
        if self.initial_activation is None:
            self.initial_activation = max_activation[0].clone().cpu().detach().numpy()
        return torch.mean(max_activation[0]).unsqueeze(0).unsqueeze(0)
    
def run_model_on_batch(
        model: torch.nn.Module,
        batch: torch.Tensor):
    _, _, bb_box_indices = model.push_forward(batch)
    #output, _, _ = model(batch)
    max_activation, min_distances, _ = model.greedy_distance(batch)
    output = model.last_layer(max_activation)
    # convert cosine dist to cosine sim
    #n_p = model.prototype_shape[-1]
    patch_activations = max_activation
    _, predicted = torch.max(output.data, 1)

    return predicted.cpu().detach().numpy(), np.clip(patch_activations.cpu().detach().numpy(), a_min=0, a_max=None), bb_box_indices


def run_model_on_dataset(
        model: nn.Module,
        dataset: Dataset,
        num_workers: int,
        batch_size: int
):
    """
    Runs the model on all images in the given directory and saves the results.
    :param model: the model to run
    :param dataset: pytorch dataset
    :param num_workers: number of parallel workers for the DataLoader
    :param batch_size: batch size for the DataLoader
    :param proto_pool: whether the model is ProtoPool
    :return a generator of model outputs for each of the images, together with batch data
    """
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )

    current_idx = 0

    for img_tensor, target in test_loader:
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        batch_samples = dataset.samples[current_idx:current_idx + batch_size]
        batch_filenames = [os.path.basename(s[0]) for s in batch_samples]
        with torch.no_grad():
            predicted_cls, patch_activations, bbox_indices = run_model_on_batch(
                model=model, batch=img_tensor)
            current_idx += img_tensor.shape[0]

        img_numpy = img_tensor.clone().cpu().detach().numpy()
        bbox_indices_numpy = bbox_indices.clone().cpu().detach().numpy()
        for d in range(3):
            img_numpy[:, d] = (img_numpy[:, d] * std[d] + mean[d])
        heat_map = retrieve_heatmap(model,img_tensor)
        yield {
            'filenames': batch_filenames,
            'target': target.cpu().detach().numpy(),
            'img_tensor': img_tensor,
            'img_original_numpy': img_numpy,
            'patch_activations': patch_activations,
            'predicted_cls': predicted_cls,
            'bbox_indx': bbox_indices_numpy,
            'heat_map':heat_map,
        }
    ########## activation metric change 
def get_activation_change_metrics(act_before, act_after, proto_nums, cls_proto_nums, proto_indx_af,proto_indx_bf, heat_map_bf,
                                 heat_map_af, log):
    metrics = {}
    # PAC calculation
    log("##############")
    max_activations_before = act_before[proto_nums] # jth prototype activation bf 
    #log(max_activations_before)
    #log('max activations before', max_activations_before)
    max_activations_after = act_after[proto_nums] # jth prototype activation af 
    #log(max_activations_after)
    
    # as a metric, calculate activation change of the top activated prototype
    argmax_act = np.argmax(max_activations_before) # most actiivated prototypes 
    top_proto_act_before, top_proto_act_after = float(max_activations_before[argmax_act]), \
        float(max_activations_after[argmax_act])
    metrics['PAC'] = (1 - top_proto_act_after / top_proto_act_before) * 100
    log(f'PAC:')
    log(str(metrics['PAC']))
    # PRC calculation 
    cls_proto_nums = set(cls_proto_nums)
    non_cls_proto_nums = np.asarray([i for i in range(act_before.shape[0]) if i not in cls_proto_nums])
    argmax_place_before = float(np.sum(act_before[non_cls_proto_nums] >
                                       act_before[proto_nums[argmax_act]]))
    argmax_place_after = float(np.sum(act_after[non_cls_proto_nums] >
                                      act_after[proto_nums[argmax_act]]))
    metrics['PRC'] = argmax_place_after - argmax_place_before
    #log('PRC:', metrics['PRC'])
    log(f'PRC:')
    log(str(metrics['PRC']))
    # PLC calculation
    total_loc = proto_indx_bf[proto_nums]
    max_proto_idx_bf = proto_indx_bf[proto_nums]
    max_proto_idx_af = proto_indx_af[proto_nums]
    max_proto_idx_bf_srt = np.sort(max_proto_idx_bf)
    max_proto_idx_af_srt = np.sort(max_proto_idx_af)
    print(proto_nums)
    
    LC_count_i =  0 
    for idx in max_proto_idx_bf_srt[0]:
        if idx not in max_proto_idx_af_srt:
            LC_count_i +=1 
    heat_map_af_proto = heat_map_af[proto_nums]
    heat_map_bf_proto = heat_map_bf[proto_nums]
    if LC_count_i != 0:
        iou = torch.sum(heat_map_af_proto & heat_map_bf_proto) / torch.sum(heat_map_af_proto | heat_map_bf_proto)
        iou = float(iou.item())
        metrics['PLC'] = 100*(1-iou)
    else:
        metrics['PLC'] = 0.0
    
    log(f'PLC:')
    log(str(metrics['PLC']))
    log("##############")
    return metrics

def adv_analysis(opt: Optional[List[str]])-> None:
    from adv_setting import load_model_path, test_dir, model_output_dir
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs=1, type=str, default='0')
    if opt is None:
        args, unknown = parser.parse_known_args()
    else:
        args, unknown = parser.parse_known_args(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\033[0;1;31m{device=}\033[0m')
    log, logclose = create_logger(log_filename=os.path.join(model_output_dir, 'local_analysis.log'))
    # load model 
    load_model_path = load_model_path
    ppnet = torch.load(load_model_path)
    ppnet = ppnet.cuda()
    img_size = ppnet.img_size
    normalize = transforms.Normalize(mean=mean,
                                 std=std)
    test_dataset_full = datasets.ImageFolder(
            test_dir,
            transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]))
    # random idx on test data 
    # for default, we use the full test data,  
    #we left this here for some possible further analysis on the subset of the test data
    random_idx = np.random.choice(np.arange(len(test_dataset_full)), replace=False, size=len(test_dataset_full))
    subset_test_dataset = Subset(test_dataset_full, random_idx)
    setattr(subset_test_dataset, 'samples', [test_dataset_full.samples[i] for i in random_idx])
    test_dataset = subset_test_dataset
    metrics_mean, metrics_all = {}, {}
    output_top_k_dir = os.path.join(model_output_dir, 'adversarial_images_summaries_cherrypicked')
    os.makedirs(output_top_k_dir, exist_ok=True)
    n_samples, n_correct_before, n_correct_after = 0, 0, 0
    metrics = defaultdict(list)
    top_k_examples, top_k_examples_diffs = [], []   
    ### attack parameter setting 
    top_k_save = 50
    attack_type = 'top_proto'
    epsilon = 0.4
    epsilon_iter = 0.01
    nb_iter = 50
    model_key = 'deit_small'
    metrics_df = defaultdict(list)
    model_keys = [model_key]
    n_samples_count, n_correct_before, n_correct_after = 0, 0, 0
    metrics = defaultdict(list)

    ## run experiments 
    pbar = tqdm(total=len(test_dataset)//8)
    for batch_result in run_model_on_dataset(
                    model=ppnet,
                    dataset=test_dataset,
                    num_workers=2,
                    batch_size=8):
        #print(batch_result)
        adversarial_result = attack_images_target_class_prototypes(
                model=ppnet,
                img=batch_result['img_tensor'],
                activations=batch_result['patch_activations'],
                attack_type=attack_type,
                cls=batch_result['target'],
                epsilon=epsilon,
                epsilon_iter=epsilon_iter,
                nb_iter=nb_iter,
                bb_box = batch_result['bbox_indx']
            )
            ## follow after break in for loop 
        n_samples_count += len(batch_result['filenames'])
        n_correct_before += np.sum(batch_result['predicted_cls'] == batch_result['target'])
        with torch.no_grad():
            predicted_cls_adv, patch_activations_adv,bbox_indices_after = run_model_on_batch(
                model=ppnet, batch=adversarial_result['img_modified_tensor']
            )
            heat_map_adv = retrieve_heatmap(ppnet, adversarial_result['img_modified_tensor'])
        bbox_indices_after = bbox_indices_after.cpu().detach().numpy()
        n_correct_after += np.sum(predicted_cls_adv == batch_result['target'])
        for sample_i in range(len(batch_result['filenames'])):
            filename = batch_result['filenames'][sample_i]
            img_original = batch_result['img_original_numpy'][sample_i]
            img_modified = adversarial_result['img_modified_numpy'][sample_i]
            sample_mask = adversarial_result['mask'][sample_i]
            proto_nums = adversarial_result['proto_nums'][sample_i]
            cls_proto_nums = adversarial_result['cls_proto_nums'][sample_i]
            proto_loc_adv = bbox_indices_after[sample_i]
            proto_loc_bf = batch_result['bbox_indx'][sample_i]
            img_original = img_original.transpose(1, 2, 0)
            img_modified = img_modified.transpose(1, 2, 0)
            sample_mask = sample_mask.transpose(1, 2, 0)
            alpha = 0.7
            modified_masked = img_modified * sample_mask + \
                                        (1 - sample_mask) * (alpha * sample_mask + (1 - alpha) * img_modified)
            activation_before = batch_result['patch_activations'][sample_i, proto_nums]
            activation_after = patch_activations_adv[sample_i, proto_nums]
            total_activation_before = np.sum(activation_before, axis=0)
            total_activation_after = np.sum(activation_after, axis=0)
            for metric_key, val in get_activation_change_metrics(batch_result['patch_activations'][sample_i],
                                                                        patch_activations_adv[sample_i],
                                                                        proto_nums, cls_proto_nums,
                                                                        proto_indx_af = proto_loc_adv,
                                                                        proto_indx_bf = proto_loc_bf,
                                                                        heat_map_bf = batch_result['heat_map'][sample_i],
                                                                        heat_map_af = heat_map_adv[sample_i],
                                                                        log = log ).items():
                if isinstance(val, list):
                    metrics[metric_key].extend(val)
                else:
                    metrics[metric_key].append(val)
                    
        acc1 = n_correct_before / n_samples_count * 100
        acc2 = n_correct_after / n_samples_count * 100
        pbar.set_description('{:s}. Acc before: {:.2f}%, after: {:.2f}%)'.format(model_key, acc1, acc2))  
        pbar.update()   
    pbar.close()

    with open(os.path.join(model_output_dir, 'metrics_all.json'), 'w') as f:
        json.dump(metrics, f)
    mean_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
    mean_metrics['Acc_before'] = float(n_correct_before / n_samples_count * 100)
    mean_metrics['Acc_after'] = float(n_correct_after / n_samples_count * 100)
    mean_metrics['AC'] = mean_metrics['Acc_before'] - mean_metrics['Acc_after']
    with open(os.path.join(model_output_dir, 'metrics_mean.json'), 'w') as f:
        json.dump(mean_metrics, f, indent=2)

    metrics_mean[model_key] = mean_metrics
    metrics_all[model_key] = metrics
    for model_key, metrics in metrics_mean.items():
        metrics_df['model'].append(model_key)
        for metric_key, val in metrics.items():
            metrics_df[metric_key].append(float(np.round(val, 2)))
    pd.DataFrame(metrics_df).to_csv(os.path.join(model_output_dir, 'metrics.csv'), index=False)
    return None 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prototype_local_analysis')
    parser.add_argument('--evaluate', '-e', action='store_true',
                        help='The run evaluation training model')
    args, unknown = parser.parse_known_args()
    
    adv_analysis(unknown)