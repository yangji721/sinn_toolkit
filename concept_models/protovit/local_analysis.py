##### MODEL AND DATA LOADING
import torch
import torch.utils.data
import os
import shutil
import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as datasets
import argparse
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
import numpy as np
from PIL import Image
from typing import List, Optional
import copy
import pickle

##### HELPER FUNCTIONS FOR PLOTTING
def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def save_preprocessed_img(fname, preprocessed_imgs, index=0):
    img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
    undo_preprocessed_img = undo_preprocess_input_function(img_copy)
    print('image index {0} in batch'.format(index))
    undo_preprocessed_img = undo_preprocessed_img[0]
    undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0])
    plt.imsave(fname, undo_preprocessed_img)
    return undo_preprocessed_img

def save_prototype(fname,img_dir):
    p_img = plt.imread(img_dir)
    plt.imsave(fname, p_img)

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
    colors = [(0, 255, 255), (255, 0, 0), (0, 255, 0), (0,0, 255), (255,255,0),(255, 0, 255), (255, 255, 255)]
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


def local_analysis(imgs, ppnet, save_analysis_path,test_image_dir,
                        start_epoch_number,
                        load_img_dir, log, prototype_layer_stride = 1):
    # only run top1 class 
    ppnet.eval()
    imgs_sep = imgs.split('/') # eg. 083.White_breasted_Kingfisher\White_Breasted_Kingfisher_0012_73367.jpg
    img_file_name = imgs_sep[0] # eg. 083.White_breasted_Kingfisher
    analysis_rt = os.path.join(save_analysis_path, imgs_sep[0], imgs_sep[1])# dir to save the analysis class 
    makedir(analysis_rt)
    img_size = ppnet.img_size
    normalize = transforms.Normalize(mean=mean,
                                    std=std)
    preprocess = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(),
    normalize
    ])
    img_rt = os.path.join(test_image_dir, imgs)
    img_pil = Image.open(img_rt)
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    images_test = img_variable.cuda()
    test_image_label = 0#int(img_file_name[0:3])-1
    labels_test = torch.tensor([test_image_label])
    slots = torch.sigmoid(ppnet.patch_select*ppnet.temp) 
    factor = ((slots.sum(-1))).unsqueeze(-1) + 1e-10# 1, 2000, 1
    logits, min_distances, values= ppnet(images_test)
    #cosine_act = -min_distances
    proto_h = ppnet.prototype_shape[2]
    n_p = proto_h # number of prototype subpatches
    _, _, indices  = ppnet.push_forward(images_test)
    values_slot = (values.clone())*(slots*n_p/factor)
    cosine_act = values_slot.sum(-1) # return the actual cosine sim as activation
    _, predicted = torch.max(logits.data, 1)
    log(f'The predicted label is {predicted}')
    print(f'The actual lable is {labels_test.item()}')
    # save the original image
    original_img = save_preprocessed_img(os.path.join(analysis_rt, 'original_img.png'),
                                     img_variable, index = 0 )
    prototype_img_filename_prefix='prototype-img'

    ##### PROTOTYPES FROM TOP-k predicted CLASSES
    k = 5
    
    #proto_w = ppnet.prototype_shape[-1]
    log('Prototypes from top-%d classes:' % k)
    topk_logits, topk_classes = torch.topk(logits[0], k=k)
    prototype_img_filename_prefix='prototype-img'
    for idx,c in enumerate(topk_classes.detach().cpu().numpy()):
        topk_dir = os.path.join(analysis_rt, 'top-%d_class_prototypes_class%d' % ((idx+1),c+1))
        makedir(topk_dir)
        log('top %d predicted class: %d' % (idx+1, c+1))
        log('logit of the class: %f' % topk_logits[idx])
        # return the prototype indices from correponding class
        class_prototype_indices = np.nonzero(ppnet.prototype_class_identity.detach().cpu().numpy()[:, c])[0]
        #return the corresponding activation
        class_prototype_activations = cosine_act[0][class_prototype_indices]
        # from the highest act to lowest for given class c 
        _, sorted_indices_cls_act = torch.sort(class_prototype_activations)
        iterat = 0 
        for s in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
            proto_bound_boxes = np.full(shape=[5, n_p],
                                            fill_value=-1)
            prototype_index = class_prototype_indices[s]
            proto_slots_j = (slots.squeeze())[prototype_index]
            log('prototype index: {0}'.format(prototype_index))
            log('activation value (similarity score): {0}'.format(class_prototype_activations[s]))
            log('proto_slots_j: {0}'.format(proto_slots_j))
            log('last layer connection with predicted class: {0}'.format(ppnet.last_layer.weight[c][prototype_index]))
            min_j_indice = indices[0][prototype_index].cpu().numpy() # n_p
            min_j_indice = np.unravel_index(min_j_indice.astype(int), (14,14))
            #print(min_j_indice)
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
                    proto_bound_boxes[0] = s
                    proto_bound_boxes[1,k] = pix_bound_k[0][0]
                    proto_bound_boxes[2,k] = pix_bound_k[0][1]
                    proto_bound_boxes[3,k] = pix_bound_k[1][0]
                    proto_bound_boxes[4,k] = pix_bound_k[1][1]

            rt = os.path.join(topk_dir,
                        'most_highly_activated_patch_in_original_img_by_top-%d_class.png' %(iterat+1))
            save_prototype_original_img_with_bbox(rt, original_img,
                                                  sub_patches = n_p,
                                                  bound_box_j = proto_bound_boxes, color=(0, 255, 255))
             # save the prototype img 
            bb_dir = os.path.join(load_img_dir, prototype_img_filename_prefix + 'bbox-original' + str(prototype_index) +'.png')
            saved_bb_dir = os.path.join(topk_dir, 'top-%d_activated_prototype_in_original_pimg_%d.png'%(iterat+1,prototype_index))
            save_prototype(saved_bb_dir,bb_dir)
            iterat+=1    


    ##### MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE############################
    most_activated_proto_dir = os.path.join(analysis_rt,'most_activated_prototypes')
    makedir(most_activated_proto_dir)
    log('Most activated 10 prototypes of this image:')
    sorted_act, sorted_indices_act = torch.sort(cosine_act[0])
    for i in range(0,10):
        proto_bound_boxes = np.full(shape=[5, n_p],
                                            fill_value=-1)
        log('top {0} activated prototype for this image:'.format(i+1))
        log('top {0} activation for this image:'.format(sorted_act[-(i+1)]))
        #print(predicted.shape)
        #print(sorted_indices_act[-(i+1)].item())
        log('last layer connection with predicted class: {0}'.format(ppnet.last_layer.weight[predicted[0]][sorted_indices_act[-(i+1)].item()]))
        proto_indx = sorted_indices_act[-(i+1)].detach().cpu().numpy()
        slots_j = (slots.squeeze())[proto_indx]
        bb_dir = os.path.join(load_img_dir, prototype_img_filename_prefix + 'bbox-original' + str(proto_indx) +'.png')
        saved_bb_dir = os.path.join(most_activated_proto_dir, 
                                    'top-%d_activated_prototype_in_original_pimg_%d.png'%(i+1,proto_indx))
        save_prototype(saved_bb_dir, bb_dir)
        ###############################################
        min_j_indice = indices[0][proto_indx].cpu().numpy() # n_p
        min_j_indice = np.unravel_index(min_j_indice.astype(int), (14,14))
        grid_width = 16
        for k in range(n_p):
            if slots_j[k] != 0:
                fmap_height_start_index_k = min_j_indice[0][k]* prototype_layer_stride
                fmap_height_end_index_k = fmap_height_start_index_k + 1
                fmap_width_start_index_k = min_j_indice[1][k] * prototype_layer_stride
                fmap_width_end_index_k = fmap_width_start_index_k + 1
                bound_idx_k = np.array([[fmap_height_start_index_k, fmap_height_end_index_k],
                [fmap_width_start_index_k, fmap_width_end_index_k]])
                pix_bound_k= bound_idx_k*grid_width
                proto_bound_boxes[0] = 0
                proto_bound_boxes[1,k] = pix_bound_k[0][0]
                proto_bound_boxes[2,k] = pix_bound_k[0][1]
                proto_bound_boxes[3,k] = pix_bound_k[1][0]
                proto_bound_boxes[4,k] = pix_bound_k[1][1]

        rt = os.path.join(most_activated_proto_dir,
                    'top-%d_most_highly_activated_patch_in_original_img.png' %(i+1))
        save_prototype_original_img_with_bbox(rt, original_img,
                                                sub_patches = n_p,
                                                bound_box_j = proto_bound_boxes, 
                                                color=(0, 255, 255))


    return None


def analyze(opt: Optional[List[str]]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs=1, type=str, default='0')
    #parser.add_argument('--modeldir', nargs=1, type=str)
    #parser.add_argument('--model', nargs=1, type=str)
    #parser.add_argument('--save_analysis_dir',type = str, help = 'Path for saving analysis result') 
    #parser.add_argument('--test_dir',type = str)
    #parser.add_argument('--check_test_acc', type = bool, default=False)
    if opt is None:
        args, unknown = parser.parse_known_args()
    else:
        args, unknown = parser.parse_known_args(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\033[0;1;31m{device=}\033[0m')
    kwargs = {}
    from analysis_settings import load_model_dir, load_model_name,save_analysis_path,img_name, test_data,check_test_acc

    model_base_architecture = 'cait'#load_model_dir.split('/')[2]
    experiment_run = '/'.join(load_model_dir.split('/')[3:])
    test_image_dir = test_data
    makedir(save_analysis_path)
    log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'local_analysis.log'))
    load_model_path = os.path.join(load_model_dir, load_model_name)
    epoch_number_str = re.search(r'\d+', load_model_name).group(0)
    start_epoch_number = int(epoch_number_str)
    log('load model from ' + load_model_path)
    log('model base architecture: ' + model_base_architecture)
    log('experiment run: ' + experiment_run)
    ppnet = torch.load(load_model_path)
    ppnet = ppnet.cuda()
    normalize = transforms.Normalize(mean=mean,
                                 std=std)
    img_size = ppnet.img_size
    prototype_shape = ppnet.prototype_shape
    # place of saved prototyeps
    load_img_dir = os.path.join(load_model_dir, img_name)
    prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0)
    prototype_max_connection = prototype_max_connection.cpu().numpy()
    if np.sum(np.sort(prototype_max_connection) == prototype_max_connection) == ppnet.num_prototypes:
        log('All prototypes connect most strongly to their respective classes.')
    else:
        log('WARNING: Not all prototypes connect most strongly to their respective classes.')
    from settings import train_dir, test_dir, train_push_dir, \
                     train_batch_size, test_batch_size, train_push_batch_size, coefs
    #heck_test_acc = False
    if check_test_acc:
        test_dataset = datasets.ImageFolder(
            test_dir,
            transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]))
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=False,
            num_workers=2, pin_memory=False)
        log('test set size: {0}'.format(len(test_loader.dataset)))
        accu, tst_loss = tnt.test(model=ppnet, dataloader=test_loader,
                            class_specific=True, log=log)
        log(f'the accuracy of the model is: {accu}')
    from analysis_settings import check_list
    for name in check_list:
        local_analysis(name, ppnet,
                        save_analysis_path,test_image_dir,
                        start_epoch_number,
                        load_img_dir, log=log, prototype_layer_stride=1)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prototype_local_analysis')
    parser.add_argument('--evaluate', '-e', action='store_true',
                        help='The run evaluation training model')
    args, unknown = parser.parse_known_args()

    analyze(unknown)