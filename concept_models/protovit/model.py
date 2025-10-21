from xml.etree.ElementPath import xpath_tokenizer_re
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from ProtoViT.tools.deit_features import deit_tiny_patch_features, deit_small_patch_features
from ProtoViT.tools.cait_features import cait_xxs24_224_features

base_architecture_to_features = {'deit_small_patch16_224': deit_small_patch_features,
                                 'deit_tiny_patch16_224': deit_tiny_patch_features,
                                 #'deit_base_patch16_224':deit_base_patch16_224,
                                 'cait_xxs24_224': cait_xxs24_224_features,}

class PPNet(nn.Module):
    def __init__(self, features, img_size, prototype_shape,
                 num_classes, init_weights=True,
                 prototype_activation_function='log',
                 sig_temp = 1.0,
                 radius = 3,
                 add_on_layers_type='bottleneck'):

        super(PPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape # p, d, n_p
        self.num_prototypes = prototype_shape[0] #10 *num_class
        self.num_classes = num_classes
        self.num_prototypes_per_class = self.num_prototypes // self.num_classes
        self.epsilon = 1e-4
        self.normalizer = nn.Softmax(dim=1)
        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function
        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        assert(self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)
 
        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        #self.proto_layer_rf_info = proto_layer_rf_info

        # this has to be named features to allow the precise loading
        self.features = features

        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True) # same shape as 200, 128, 1, 1
        self.radius = radius # unit of patches to be considered as neighbor 
        # initializations for adaptive subpatch
        #self.patch_select_init = torch.zeros(prototype_shape[0],1, prototype_shape[-1]) # 2000,1, 4 
        self.patch_select = nn.Parameter(torch.ones(1, prototype_shape[0], prototype_shape[-1])*0.1, 
                                         requires_grad=True) # 2000 4 
        self.temp = sig_temp
        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False) # do not use bias
        features_name = str(features).upper()
        #print(features_name)
        if features_name.startswith('VISION'):
            self.arc = 'deit'
        elif features_name.startswith('CAIT'):
            self.arc = 'cait'
        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        this version is to reuse the cls-token 
        in computation of feature representation 

        patch_emb_new = patch_emb - cls_token_emb (a focal similarity style)
        
        '''
        x = self.features.patch_embed(x)
        cls_token = self.features.cls_token.expand(x.shape[0], -1, -1)# stole cls_tokens impl from Phil Wang, thanks
        if self.arc == 'deit':
            '''
            forward feature from Deit backbone 
            '''
            x = torch.cat((cls_token, x), dim=1) 
            x = self.features.pos_drop(x + self.features.pos_embed)
            x = self.features.blocks(x)
            x = self.features.norm(x) # bsz, 197, dim

        elif self.arc == 'cait':
            """
            forward feature from cait backbone 
            """
            x = x + self.features.pos_embed
            x = self.features.pos_drop(x)
            for i , blk in enumerate(self.features.blocks):
                x = blk(x)
            for i , blk in enumerate(self.features.blocks_token_only):
                cls_token = blk(x, cls_token)
            x = torch.cat((cls_token, x), dim=1)
            x = self.features.norm(x) # bsz, 197, dim

        # patch_emb that adds global info 
        x_2 = x[:, 1:] - x[:, 0].unsqueeze(1) # bsz, 196, dim
        #x = x[:,1:] # bsz, 196, dim
        fea_len =x_2.shape[1] 
        B, fea_width, fea_height = x_2.shape[0],int(fea_len ** (1/2)), int(fea_len ** (1/2))
        feature_emb = x_2.permute(0,2,1).reshape(B, -1, fea_width, fea_height)
        #print(feature_emb.shape)
        return feature_emb
    
    def _cosine_convolution(self, x):

        x = F.normalize(x,p=2,dim=1)
        now_prototype_vectors = F.normalize(self.prototype_vectors,p=2,dim=1)
        # Reshape prototype vectors to match conv2d dimensions
        now_prototype_vectors = now_prototype_vectors.view(self.num_prototypes, x.size(1), 1, 1)
        distances = F.conv2d(input=x, weight=now_prototype_vectors)
        distances = -distances
        
        return distances
    
    def _project2basis(self,x):
        # essentially the same 
        x = F.normalize(x,p=2,dim=1)
        now_prototype_vectors = F.normalize(self.prototype_vectors, p=2, dim=1)
        # Reshape prototype vectors to match conv2d dimensions 
        now_prototype_vectors = now_prototype_vectors.view(self.num_prototypes, x.size(1), 1, 1)
        distances = F.conv2d(input=x, weight=now_prototype_vectors)
        return distances
    
    def prototype_distances(self, x):

        conv_features = self.conv_features(x)
        cosine_distances = self._cosine_convolution(conv_features)
        project_distances = self._project2basis(conv_features)
        return project_distances,cosine_distances
    
    def global_min_pooling(self,distances):

        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        min_distances = min_distances.view(-1, self.num_prototypes)
        return min_distances

    def global_max_pooling(self,distances):

        max_distances = F.max_pool2d(distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        max_distances = max_distances.view(-1, self.num_prototypes)

        return max_distances
    
    def subpatch_dist(self, x):
        """
        Input: data x 
        output: conv_features, activation map for each subpatch concat into one tensor 
        dist_all: bsz, num_proto, 14*14, 4 
        (14*14): flatten number of activation map (vary by prototype size)
        """
        #slots = torch.sigmoid(self.patch_select*self.temp) # temp set large enough to approximate step functions 
        #factor = ((slots.sum(-1))).unsqueeze(-1)# 1, 2000, 1, 1
        dist_all = torch.FloatTensor().cuda()
        conv_feature = self.conv_features(x)
        conv_features_normed = F.normalize(conv_feature,p=2,dim=1)#/factor 
        now_prototype_vectors= F.normalize(self.prototype_vectors,p=2,dim=1)#/factor
        now_prototype_vectors = now_prototype_vectors#*slots/factor
        n_p = self.prototype_shape[-1]
        for i in range(n_p):
            proto_i = now_prototype_vectors[:,:, i].unsqueeze(-1).unsqueeze(-1)
            # Reshape proto_i to match expected dimensions (out_channels, in_channels, height, width)
            proto_i = proto_i.view(-1, conv_features_normed.size(1), 1, 1)
            dist_i = F.conv2d(input=conv_features_normed, weight=proto_i).flatten(2).unsqueeze(-1) # bsz, n_p, 196,1 
            dist_all = torch.cat([dist_all, dist_i], dim=-1)
        #dist_all = dist_all # bsz, 2000, 196, 4
        return conv_feature, dist_all
    
    def neigboring_mask(self, center_indices):
        """
        This function add a radius by radius matrix center on the 
        selected top patches to encourage adjacency of the prototypes 
        Input center_indices: max_patch_id shape bsz, 2000, 1
        Some hardcoding is here (lazy)
        return a neighboring mask: bsz 2000 196 
        0: means non-adjacent (not included)
        1: adjacent (included)
        """
        # add padding to the original target size 
        large_padded = (14+self.radius*2)**2 
        large_matrix = torch.zeros(center_indices.shape[0], self.num_prototypes, large_padded).cuda()
        small_total = (2*self.radius + 1)**2
        small_matrix = torch.ones(center_indices.shape[0], self.num_prototypes, small_total).cuda()
        batch_size, num_points, _ = center_indices.shape
        small_size = int(small_matrix.shape[-1]**0.5)
        large_size = int(large_matrix.shape[-1]**0.5)
        # Reshape center_indices for broadcasting, and convert to 2D indices
        # Unfortunately divmod doesn't work on torch 
        #center_row, center_col = divmod(center_indices.squeeze(-1).cpu().numpy(), 14)
        center_row, center_col = center_indices.squeeze(-1)//14, center_indices.squeeze(-1)%14
        # Calculate the top-left corner for the rxr addition
        # relative location in the padded matrix to the original shape 
        start_row = torch.tensor(center_row+self.radius - small_size // 2)
        start_col = torch.tensor(center_col+self.radius - small_size // 2)
        # Handle boundaries (padding might be required if indices go negative)
        start_row = torch.clamp(start_row, 0, large_size - small_size)
        start_col = torch.clamp(start_col, 0, large_size - small_size)
        # Iterate through each possible position in the rxr matrix
        for i in range(small_size):
            for j in range(small_size):
                # Determine the corresponding position in the 14x14 matrix
                large_row = start_row + i
                large_col = start_col + j
                # Convert 2D indices back to 1D indices for both matrices
                large_idx = large_row * large_size + large_col
                small_idx = i * small_size + j
                # Add the small matrix values to the large matrix
                large_matrix.view(batch_size, num_points, -1)[torch.arange(batch_size)[:, None], 
                                                                torch.arange(num_points), large_idx] += small_matrix[..., small_idx]
        large_matrix_reshape = large_matrix.view(batch_size, num_points,large_size,large_size )
        large_matrix_unpad = large_matrix_reshape[:,:, self.radius: -self.radius,  self.radius:-self.radius] # bsz, 2000, 14,14 
        large_matrix_unpad = large_matrix_unpad.reshape(batch_size,num_points,-1) # bdz, 2000, 196
        return large_matrix_unpad
    

    def greedy_distance(self, x, get_f = False):
        """
        This function implements greedy matching algorithm 
        takes input image and returns the similarity scores 
        by greedy match, the designed mindistances, 
        and corresponding patch index. 
        mask identity: 1 kept, 0 removed 

        Similarity score is caculated as a sum of scores for all
        sub-component of prototypes 

        X: input from sample batches 
        get_f: bool indicate if we want to return conv_features
        """
        conv_features, dist_all = self.subpatch_dist(x)
        slots = torch.sigmoid(self.patch_select*self.temp) # temp set large enough to approximate step functions
        factor = ((slots.sum(-1))).unsqueeze(-1) + 1e-10# 1, 2000, 1, avoid 0 division
        #slots = self.soft_round(slots)
        # distance calculation 
        n_p = self.prototype_shape[-1]#*self.prototype_shape[-2]
        # hard-code for now, always reinitialize for each of the calculation 
        # 196 hard code for now 
        mask_act = torch.ones((x.shape[0], self.num_prototypes, dist_all.shape[2])).cuda() # bsz, num proto, 196
        mask_subpatch = torch.ones((x.shape[0], self.num_prototypes, n_p)).cuda()
        mask_all = torch.ones((x.shape[0], self.num_prototypes, dist_all.shape[2], n_p)).cuda() # bsz, num_proto, 196, 4
        # initialize adj mask ==> everything is considered adjacent at begining 
        adjacent_mask = torch.ones((x.shape[0], self.num_prototypes, dist_all.shape[2])).cuda()# bsz, num proto, 196
        indices =  torch.FloatTensor().cuda()
        values =  torch.FloatTensor().cuda()
        # to record the sequence of subpatches being selected for later reordering 
        subpatch_ids = torch.LongTensor().cuda()
        for _ in range(n_p):
            # update activation map with masks 
            dist_all_masked = dist_all + (1-mask_all*adjacent_mask.unsqueeze(-1))*(-1e5)
            # for each subpatch, find the closest latent patch 
            max_subs, max_subs_id = dist_all_masked.max(2) # bsz, num_proto, num_subpatches
            # find the closest subpatch 
            max_sub_act, max_sub_act_id = max_subs.max(-1) # bsz, num_proto
            # get the index of actual selected img patch
            max_patch_id = max_subs_id.gather(-1,max_sub_act_id.unsqueeze(-1)) # bsz, 2000, 1
            # by the actual index, find the adjacent entries by radius 
            adjacent_mask = self.neigboring_mask(max_patch_id)
            # update adjacent mask for next iteration 
            #adjacent_mask = adjacent_mask.cuda()
            # remove the selected img patch for each subpatch from mask
            mask_act = mask_act.scatter(index = max_patch_id, dim=2, value =0)
            # remove the selected subpatch
            mask_subpatch = mask_subpatch.scatter(index=max_sub_act_id.unsqueeze(-1), dim=2, value=0)
            # remove the selected subpatch from the overall patch 
            mask_all = mask_all*mask_act.unsqueeze(-1)
            # remove the non-adjacent patches from all the subpatches pool
            mask_all = mask_all.permute(0,1,3,2)
            mask_all = mask_all*mask_subpatch.unsqueeze(-1)
            mask_all = mask_all.permute(0,1,3,2)# bsz, 2000, 196, 4
            #update selected activations and indices 
            max_sub_act = max_sub_act.unsqueeze(-1)
            subpatch_ids = torch.cat([subpatch_ids, max_sub_act_id.unsqueeze(-1)], dim = -1)
            indices = torch.cat([indices, max_patch_id], dim =-1)
            # use the default values to compute for loss
            values = torch.cat([values, max_sub_act], dim =-1)
        subpatch_ids = subpatch_ids.to(torch.int64)
        _,sub_indexes = subpatch_ids.sort(-1) 
        values_reordered = torch.gather(values, -1,sub_indexes)
        indices_reordered = torch.gather(indices, -1,sub_indexes)
        # standardized values by slots --> used for prediction 
        values_slot = (values_reordered.clone())*(slots*n_p/factor)
        #assert((mask.sum(2) == (196-n_p)).sum() == (mask.shape[0]*mask.shape[1]))
        #max_activations = values.sum(-1) # bsz, 2000/ n_p
        max_activation_slots = values_slot.sum(-1)
        min_distances = n_p -max_activation_slots # cosine distance
        if get_f:
            return conv_features, min_distances, indices_reordered
        return max_activation_slots, min_distances, values_reordered

    def push_forward_old(self, x):
        conv_output = self.conv_features(x) #[batchsize,128,14,14]
        distances = self._project2basis(conv_output)
        distances = - distances
        return conv_output, distances
    
    def push_forward(self, x):
        """
        This function does not return distance measure for 
        each patch. Instead, it returns the max overall similarity score 
        by the nature of greedy matching
        """
        #conv_output = self.conv_features(x)
        conv_output, min_distances,indices = self.greedy_distance(x, get_f=True)
        return conv_output, min_distances, indices 
        
    def forward(self, x):
        max_activation, min_distances, values = self.greedy_distance(x)
        logits = self.last_layer(max_activation)
        return logits, min_distances, values
    
    def __repr__(self):
        # PPNet(self, features, img_size, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            #'\tproto_layer_rf_info: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.img_size,
                          self.prototype_shape,
                          #self.proto_layer_rf_info,
                          self.num_classes,
                          self.epsilon)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)



def construct_PPNet(base_architecture, pretrained=True, img_size=224,
                    prototype_shape=(2000, 192, 1, 1), num_classes=200,
                    prototype_activation_function='log',
                    sig_temp = 1.0,
                    radius = 1,
                    add_on_layers_type='bottleneck'):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)

    return PPNet(features=features,
                 img_size=img_size,
                 prototype_shape=prototype_shape,
                 num_classes=num_classes,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 radius = radius,
                 sig_temp = sig_temp,
                 add_on_layers_type=add_on_layers_type)

