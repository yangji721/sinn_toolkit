import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import time
import torch.nn.functional as F

from helpers import makedir, find_high_activation_crop

def save_prototype_original_img_with_bbox(dir_for_saving_prototypes, img_dir,prototype_img_filename_prefix,j,
                                          sub_patches,
                                          indices,
                                          bound_box_j, color=(0, 255, 255)):
    """
    a modified bbox function that takes the bound_box_j that contains k patches 
    and return the deformed boudning boxes 

    color for first selected (from top to bottom):
    Yellow, red, green, blue 
    """
    save_dir = os.path.join(dir_for_saving_prototypes,
                 prototype_img_filename_prefix + 'bbox-original' + str(j) +'.png')
    p_img_bgr = cv2.imread(img_dir)
    img_bbox = p_img_bgr.copy()
    # cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
    #               color, thickness=2)
    colors = [(0, 255, 255), (255, 0, 0), (0, 255, 0), (0,0, 255)]
    mask_val = np.ones((14,14))*0.4 # set everything else to be 0.2
    for k in range(sub_patches):
        if bound_box_j[1,k] != -1:
            # draw k 16x16 bounding boxes if the patches are included 
            x,y = indices[0][k], indices[1][k]
            mask_val[x,y] = 1
            bbox_height_start_k = bound_box_j[1,k]
            bbox_height_end_k = bound_box_j[2,k]
            bbox_width_start_k = bound_box_j[3,k]
            bbox_width_end_k = bound_box_j[4,k]
            color = colors[k]
            #cv2.rectangle(img_bbox, (bbox_width_start_k, bbox_height_start_k), (bbox_width_end_k-1, bbox_height_end_k-1),
                    #color, thickness=1)
            cv2.rectangle(p_img_bgr, (bbox_width_start_k, bbox_height_start_k), (bbox_width_end_k-1, bbox_height_end_k-1),
                    color, thickness=2)
    p_img_rgb = p_img_bgr[...,::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    plt.imsave(save_dir, p_img_rgb,vmin=0.0,vmax=1.0)
    size = p_img_rgb.shape[1]
    
    img_bbox_rgb = np.clip(img_bbox + 150, 0, 255)# increase the brightness
    img_bbox_rgb = img_bbox[...,::-1]
    img_bbox_rgb = np.float32(img_bbox_rgb) / 255
    width = size//14
    
    #bb_og = p_img_rgb.copy()
    for i in range(0, 196):
        x = i %14
        y = i//14
        img_bbox_rgb[y*width:(y+1)*width, x*width:(x+1)*width]*=mask_val[y,x]
        #bb_og[y*width:(y+1)*width, x*width:(x+1)*width]*=mask_val[y,x]

    save_dir2 = os.path.join(dir_for_saving_prototypes,
                 prototype_img_filename_prefix + '_vis_' + str(j) +'.png')
    plt.imsave(save_dir2, img_bbox_rgb,vmin=0.0,vmax=1.0)
    
    #save_dir3 = os.path.join(dir_for_saving_prototypes,
                 #prototype_img_filename_prefix + '_vis_bb_' + str(j) +'.png')
    
    #plt.imsave(save_dir3, img_bbox_rgb,vmin=0.0,vmax=1.0)
    #for k in range()
    


def update_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               pnet,
                               global_min_proto_dist, # this will be updated
                               global_min_fmap_patches, # this will be updated
                               proto_bound_boxes, # this will be updated
                               class_specific=True,
                               search_y=None, # required if class_specific == True
                               num_classes=None, # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None):
    pnet.eval()
    if preprocess_input_function is not None:
        search_batch = preprocess_input_function(search_batch_input)
    with torch.no_grad():
        search_batch = search_batch.cuda()
    # pruned values 
    protoL_input_torch, proto_dist_torch, proto_indices_torch = pnet.push_forward(search_batch) 
    slots_torch_raw = torch.sigmoid(pnet.patch_select*pnet.temp)
    slots_torch = torch.round(slots_torch_raw, decimals=1)# remove the case such as 1e-4, 1e-5 thats approximately 0
    proto_slots = np.copy(slots_torch.detach().cpu().numpy())
    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())
    proto_indice_ = np.copy(proto_indices_torch.detach().cpu().numpy())
    del protoL_input_torch, proto_dist_torch, proto_indices_torch,slots_torch,slots_torch_raw
    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        # img_y is the image's integer label
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)
    prototype_shape = pnet.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    #proto_w = prototype_shape[3]
    # number of prototypical patches for each prototype 
    n_p = proto_h#*proto_w
    for j in range(n_prototypes):
        if class_specific:
            # target_class is the class of the class_specific prototype
            target_class = torch.argmax(pnet.prototype_class_identity[j]).item()
            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(class_to_img_index_dict[target_class]) == 0:
                continue
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:,j]
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_dist_j = proto_dist_[:,j]
        # find the min of the min_distances 
        batch_min_proto_dist_j = np.amin(proto_dist_j)
        #batch_min_dist_j_indices = np.argmin(proto_dist_j,keepdims=True)
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = \
                list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                      proto_dist_j.shape))
            if class_specific:
                '''
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                batch_argmin_proto_dist_j, the index of closest img to p_j
                min_j_indice, the indices of the sub-part of p_j on the closet img
                '''
                batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]
            # retrieve the corresponding feature map
            batch_argmin_j_patch_indices = proto_indice_[batch_argmin_proto_dist_j, j][0] # n_p
            #batch_argmin_j_patch_subvalues = protot_subvalues[batch_argmin_proto_dist_j, j][0]
            proto_slots_j = (proto_slots.squeeze())[j]
            min_j_indice = np.unravel_index(batch_argmin_j_patch_indices.astype(int), (14,14))
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            global_min_proto_dist[j] = batch_min_proto_dist_j
            # get the whole image 
            original_img_j = search_batch_input[batch_argmin_proto_dist_j[0]]
            original_img_j = original_img_j.numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            grid_width = 16
            for k in range(n_p):
                if proto_slots_j[k]!= 0:
                    # each patch is 1x1 containing 16 x 16 pixels 
                    fmap_height_start_index_k = min_j_indice[0][k]* prototype_layer_stride
                    fmap_height_end_index_k = fmap_height_start_index_k + 1
                    fmap_width_start_index_k = min_j_indice[1][k] * prototype_layer_stride
                    fmap_width_end_index_k = fmap_width_start_index_k + 1
                    #print(fmap_height_start_index_k)
                    batch_min_fmap_patch_j_k = protoL_input_[img_index_in_batch,
                                                        :,
                                                        fmap_height_start_index_k:fmap_height_end_index_k,
                                                        fmap_width_start_index_k:fmap_width_end_index_k]
                    #print(batch_min_fmap_patch_j_k.shape)
                    #print(global_min_fmap_patches[j,:,k].shape)
                    global_min_fmap_patches[j,:,k] = batch_min_fmap_patch_j_k.squeeze(-1).squeeze(-1)
                    bound_idx_k = np.array([[fmap_height_start_index_k, fmap_height_end_index_k],
                    [fmap_width_start_index_k, fmap_width_end_index_k]])
                    pix_bound_k= bound_idx_k*grid_width
                    # not saving prototype img for now, prototypes are shown in the bbox
                    proto_img_j_k = original_img_j[bound_idx_k[0][0]:bound_idx_k[0][1],
                                            bound_idx_k[1][0]:bound_idx_k[1][1], :]
                    proto_bound_boxes[j, 0, k] = batch_argmin_proto_dist_j[0] + start_index_of_search_batch
                    proto_bound_boxes[j, 1, k] = pix_bound_k[0][0]
                    proto_bound_boxes[j, 2, k] = pix_bound_k[0][1]
                    proto_bound_boxes[j, 3, k] = pix_bound_k[1][0]
                    proto_bound_boxes[j, 4, k] = pix_bound_k[1][1]
                    if proto_bound_boxes.shape[1] == 6 and search_y is not None:
                        proto_bound_boxes[j, 5, k] = search_y[batch_argmin_proto_dist_j[0]].item()
            # start saving images 
            if dir_for_saving_prototypes is not None:
                if prototype_img_filename_prefix is not None:
                    original_img_path = os.path.join(dir_for_saving_prototypes,
                            prototype_img_filename_prefix + '-original' + str(j) + '.png')
                    plt.imsave(original_img_path,
                    original_img_j,
                    vmin=0.0,
                    vmax=1.0)
            # rt = os.path.join(dir_for_saving_prototypes,
            #                 prototype_img_filename_prefix + 'bbox-original' + str(j) +'.png')
            save_prototype_original_img_with_bbox(dir_for_saving_prototypes, original_img_path,prototype_img_filename_prefix,j = j,
                                                  sub_patches = n_p,
                                                  indices = min_j_indice,
                                                  bound_box_j = proto_bound_boxes[j], color=(0, 255, 255))
            # rt_newvis = os.path.join(dir_for_saving_prototypes,
            #                 prototype_img_filename_prefix + '_newvis_' + str(j) +'.png')
            # proto_new_vis(rt_newvis, original_img_path,sub_patches= n_p,
            #                             slots = proto_slots_j,
            #                             indices = min_j_indice,
            #                             bound_box_j = proto_bound_boxes[j], color=(0, 255, 255))
            
    return None 



# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    pnet, # pytorch network with prototype_vectors
                    class_specific=True,
                    preprocess_input_function=None, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True, # which class the prototype image comes from
                    log=print,
                    prototype_activation_function_in_numpy=None):
    pnet.eval()
    log('\tpush')
    start = time.time()
    prototype_shape = pnet.prototype_shape # 10*nclass 128(feature depth) 1 1 
    n_prototypes = pnet.num_prototypes
    global_min_proto_dist = np.full(n_prototypes, np.inf)
    # saves the patch representation that gives the current smallest distance
    n_p = prototype_shape[2]#*prototype_shape[3]
    global_min_fmap_patches = np.zeros(
        [n_prototypes,
         prototype_shape[1],
         n_p])
    # update the discrete slots approximated by sigmoid 
    slots = torch.sigmoid(pnet.patch_select*pnet.temp).clone()
    slots_rounded = slots.round()
    result_tensor = torch.where(slots_rounded == 0, torch.tensor(-1), slots_rounded)*200 # ensure extreme values for eval
    pnet.patch_select.data.copy_(torch.tensor(result_tensor.detach().cpu().numpy(), dtype=torch.float32).cuda())
    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    ex_dim:sub_patch component index
    '''
    if save_prototype_class_identity:
        proto_bound_boxes = np.full(shape=[n_prototypes, 5, n_p],
                                            fill_value=-1)
    else:
        proto_bound_boxes = np.full(shape=[n_prototypes, 4, n_p],
                                            fill_value=-1)
    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-'+str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size
    num_classes = pnet.num_classes
    for push_iter, (search_batch_input, search_y) in enumerate(dataloader):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        start_index_of_search_batch = push_iter * search_batch_size
        update_prototypes_on_batch(search_batch_input,
                                   start_index_of_search_batch,
                                   pnet,
                                   global_min_proto_dist,
                                   global_min_fmap_patches,
                                   proto_bound_boxes,
                                   class_specific=class_specific,
                                   search_y=search_y,
                                   num_classes=num_classes,
                                   preprocess_input_function=preprocess_input_function,
                                   prototype_layer_stride=prototype_layer_stride,
                                   dir_for_saving_prototypes=proto_epoch_dir,
                                   prototype_img_filename_prefix=prototype_img_filename_prefix,
                                   prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                   prototype_activation_function_in_numpy=prototype_activation_function_in_numpy)
    
    # print out the stats for slots after push 
    slots_pushed= torch.sigmoid(pnet.patch_select*pnet.temp).squeeze(1).sum(-1)
    unique_elements, counts = torch.unique(slots_pushed, return_counts=True)
    counter= dict(zip(unique_elements.tolist(), counts.tolist()))
    log(str(counter))

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
                proto_bound_boxes)

    log('\tExecuting push ...')
    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(prototype_shape))
    # push prototype to latent feature 
    pnet.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
    # prototype_network_parallel.cuda()
    end = time.time()
    log('\tpush time: \t{0}'.format(end -  start))
    return None 