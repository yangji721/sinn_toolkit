import torch
import numpy as np

import heapq

import matplotlib.pyplot as plt
import os
import time

import cv2
from helpers import makedir

def save_prototype_original_img_with_bbox(save_dir, img_rgb,
                                          sub_patches,
                                          bound_box_j, color=(0, 255, 255)):
    """
    a modified bbox function that takes the bound_box_j that contains k patches 
    and return the deformed boudning boxes 

    color for first selected (from top to bottom):
    Yellow, red, green, blue 
    """
    #p_img_bgr = cv2.imread(img_dir)
    p_img_bgr = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    # cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
    #               color, thickness=2)
    colors = [(0, 255, 255), (255, 0, 0), (0, 255, 0), (0,0, 255)]
    for k in range(sub_patches):
        if bound_box_j[1,k] != -1:
            # draw k 16x16 bounding boxes 
            bbox_height_start_k = bound_box_j[1,k]
            bbox_height_end_k = bound_box_j[2,k]
            bbox_width_start_k = bound_box_j[3,k]
            bbox_width_end_k = bound_box_j[4,k]
            color = colors[k]
            cv2.rectangle(p_img_bgr, (bbox_width_start_k, bbox_height_start_k), (bbox_width_end_k-1, bbox_height_end_k-1),
                       color, thickness=2)
    p_img_rgb = p_img_bgr[...,::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    plt.imsave(save_dir, p_img_rgb,vmin=0.0,vmax=1.0)

class ProtoImage:
    def __init__(self, bb_box_info, 
                 label, activation,
                 original_img=None):
        self.bb_box_info = bb_box_info
        self.label = label
        self.activation = activation

        self.original_img = original_img
    def __lt__(self, other):
        return self.activation < other.activation

    def __str__(self):
        return str(self.label) + str(self.activation)


class ImagePatch:

    def __init__(self, patch, label, activation,
                 original_img=None, act_pattern=None, patch_indices=None):
        self.patch = patch
        self.label = label
        self.activation = activation
        self.original_img = original_img
        self.patch_indices = patch_indices

    def __lt__(self, other):
        return self.activation < other.activation
    
class ImagePatchInfo:

    def __init__(self, label, activation):
        self.label = label
        self.activation = activation

    def __lt__(self, other):
        return self.activation < other.activations

# find the nearest patches in the dataset to each prototype
def find_k_nearest_patches_to_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                                         ppnet, # pytorch network with prototype_vectors
                                         num_nearest_neighbors=5,
                                         preprocess_input_function=None, # normalize if needed
                                         root_dir_for_saving_images='./nearest',
                                         log=print,
                                         prototype_layer_stride=1):
    ppnet.eval()
    log('find nearest patches')
    n_prototypes = ppnet.num_prototypes
    prototype_shape = ppnet.prototype_shape
    # allocate an array of n_prototypes number of heaps
    heaps = []
    for _ in range(n_prototypes):
        # a heap in python is just a maintained list
        heaps.append([])
    for index, (search_batch_input, search_y) in enumerate(dataloader):
        print('batch {}'.format(index))
        if preprocess_input_function is not None:
            search_batch = preprocess_input_function(search_batch_input)
            search_batch = search_batch.cuda()
        else:
            search_batch = search_batch_input.cuda()
        # calculate the necessary values 
        n_p = ppnet.prototype_shape[2]
        slots_torch_raw = torch.sigmoid(ppnet.patch_select*ppnet.temp)
        proto_slots = np.copy(slots_torch_raw.detach().cpu().numpy())
        factor = ((slots_torch_raw.sum(-1))).unsqueeze(-1)+1e-10
        # after push, we don't need to round anymore 
        protoL_input_torch, proto_dist_torch, proto_indices_torch = ppnet.push_forward(search_batch)
        _, _, values= ppnet(search_batch)
        values_slot = (values.clone())*(slots_torch_raw*n_p/factor)
        # actual activation used for calculation
        cosine_act = values_slot.sum(-1)
        protoL_input_torch = protoL_input_torch.detach()
        proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())
        proto_indice_ = np.copy(proto_indices_torch.detach().cpu().numpy())
        del protoL_input_torch, proto_dist_torch, proto_indices_torch,slots_torch_raw
        # since we already know the index of the selected region, we don't need activation map 
        for b_idx, indices_b in enumerate(proto_indice_):
            # img --> bsz, indices_b --> 2000, 4 (indexes for each subpatch)
            cosine_act_b = cosine_act[b_idx,]
            for j in range(n_prototypes):
                # find the corresponding activation 
                cosine_act_j = cosine_act_b[j]
                indices_j = indices_b[j]
                original_img = search_batch_input[b_idx].detach().cpu().numpy()
                original_img = np.transpose(original_img, (1, 2, 0))
                original_img_size = search_batch_input[b_idx].shape[-1]
                proto_slots_j = (proto_slots.squeeze())[j]
                proto_bound_boxes = np.full(shape=[5, n_p],
                                                fill_value=-1)
                #min_j_indice = indices_j#.cpu().numpy() # n_p
                min_j_indice = np.unravel_index(indices_j.astype(int), (14,14))
                grid_width = 16
                for k in range(n_p):
                    if proto_slots_j[k]!=0:
                        fmap_height_start_index_k = min_j_indice[0][k]* prototype_layer_stride
                        fmap_height_end_index_k = fmap_height_start_index_k + 1
                        fmap_width_start_index_k = min_j_indice[1][k] * prototype_layer_stride
                        fmap_width_end_index_k = fmap_width_start_index_k + 1
                        bound_idx_k = np.array([[fmap_height_start_index_k, fmap_height_end_index_k],
                        [fmap_width_start_index_k, fmap_width_end_index_k]])
                        pix_bound_k= bound_idx_k*grid_width
                        proto_bound_boxes[0] = j
                        proto_bound_boxes[1,k] = pix_bound_k[0][0]
                        proto_bound_boxes[2,k] = pix_bound_k[0][1]
                        proto_bound_boxes[3,k] = pix_bound_k[1][0]
                        proto_bound_boxes[4,k] = pix_bound_k[1][1]

                highest_patch = ProtoImage(bb_box_info =proto_bound_boxes, 
                    label = search_y[b_idx], 
                    activation = cosine_act_j ,
                    original_img=original_img)
                
                # add to the j-th heap
                if len(heaps[j]) < num_nearest_neighbors:
                    heapq.heappush(heaps[j], highest_patch)
                else:
                    # heappushpop runs more efficiently than heappush
                    # followed by heappop
                    heapq.heappushpop(heaps[j], highest_patch)
    # after looping through the dataset every heap will
    # have the num_nearest_neighbors closest prototypes
    for j in range(n_prototypes):
        # finally sort the heap; the heap only contains the num_nearest_neighbors closest
        # but they are not ranked yet
        heaps[j].sort()
        heaps[j] = heaps[j][::-1]
        dir_for_saving_images = os.path.join(root_dir_for_saving_images,
                                                 str(j))
        makedir(dir_for_saving_images)
        labels = []
        for i, patch in enumerate(heaps[j]):
            # save the original image where the patch comes from
            plt.imsave(fname=os.path.join(dir_for_saving_images,
                                              'nearest-' + str(i+1) + '_original.png'),
                           arr=patch.original_img,
                           vmin=0.0,
                           vmax=1.0)
            bb_dir = os.path.join(dir_for_saving_images, 'nearest-' + str(i) + '_patch_with_box.png')
            save_prototype_original_img_with_bbox(bb_dir, patch.original_img,
                                                  sub_patches = n_p,
                                                  bound_box_j = patch.bb_box_info, color=(0, 255, 255))
        labels = np.array([patch.label for patch in heaps[j]])
        np.save(os.path.join(dir_for_saving_images, 'class_id.npy'),
                    labels)
    labels_all_prototype = np.array([[patch.label for patch in heaps[j]] for j in range(n_prototypes)])
    np.save(os.path.join(root_dir_for_saving_images, 'full_class_id.npy'),
                labels_all_prototype)
            
    return labels_all_prototype