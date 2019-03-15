import torch
import torch.nn as nn
import torch.nn.functional as F

class HardTripletLoss(nn.Module):
    """Hard/Hardest Triplet Loss
    (pytorch implementation of https://omoindrot.github.io/triplet-loss)

    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    """
    def __init__(self, margin = 1,cos_sim = False):
        """
        Args:
            margin: margin for triplet loss
        """
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.cos_sim = cos_sim
        
    def forward(self, attributes, embeddings, labels):
        """
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        if self.cos_sim:
            relations = _pairwise_distance_cos(attributes, embeddings ,labels)
        else:
            relations = _pairwise_distance(attributes, embeddings ,labels)
            
        #print(relations.size())
        # Get the hardest positive pairs
        mask_pos=_get_anchor_positive_triplet_mask(relations, labels).float()
        #print(mask_pos)
        valid_positive_dist = relations * mask_pos
        hardest_positive_dist, hardest_positive_dist_index  = torch.max(valid_positive_dist, 0 ,keepdim=True)
        hardest_positive_dist_img, _ = torch.max(valid_positive_dist, 1 ,keepdim=True)
        #print(hardest_positive_dist_img.size())

        
        
        # Get the hardest negative pairs
        mask_neg=_get_anchor_negative_triplet_mask(relations, labels).float()
        #print(mask_neg)
        max_anchor_negative_dist, _ = torch.max(relations, 0 , keepdim=True)
        max_anchor_negative_dist_img, _ = torch.max(relations, 1 , keepdim=True)
        
        anchor_negative_dist = relations + max_anchor_negative_dist * (1.0 - mask_neg)
        anchor_negative_dist_img = relations + max_anchor_negative_dist_img * (1.0 - mask_neg)
        
        hardest_negative_dist, _ = torch.min(anchor_negative_dist, 0 , keepdim=True)
        hardest_negative_dist_img = torch.zeros_like(hardest_negative_dist)
        #print(hardest_negative_dist_img.size())
        for i in range(hardest_positive_dist_index.size()[1]):
            hardest_negative_dist_img[0][i] = torch.min(anchor_negative_dist_img[hardest_positive_dist_index[0][i]])
        
        triplet_loss_all = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        triplet_loss_all_img = F.relu(hardest_positive_dist - hardest_negative_dist_img + self.margin)
        num_hard_triplets = torch.sum(torch.gt(triplet_loss_all, 1e-16).float())
        num_hard_triplets_img = torch.sum(torch.gt(triplet_loss_all_img, 1e-16).float())
        
        triplet_loss = (torch.sum(triplet_loss_all_img))/(num_hard_triplets_img+1e-16)
        return triplet_loss
class HardTripletLoss2(nn.Module):
    """Hard/Hardest Triplet Loss
    (pytorch implementation of https://omoindrot.github.io/triplet-loss)

    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    """
    def __init__(self, margin = 1,cos_sim = False):
        """
        Args:
            margin: margin for triplet loss
        """
        super(HardTripletLoss2, self).__init__()
        self.margin = margin
        self.cos_sim = cos_sim
        
    def forward(self, attributes, embeddings, labels):
        """
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        if self.cos_sim:
            relations = _pairwise_distance_cos(attributes, embeddings ,labels)
        else:
            relations = _pairwise_distance(attributes, embeddings ,labels)
            relations2 = euclid_pairwise_distance(attributes, embeddings)
 
        
        # Get the hardest positive pairs
        mask_pos=_get_anchor_positive_triplet_mask(relations, labels).float()
        
        valid_positive_dist = relations * mask_pos
        hardest_positive_dist, _ = torch.max(valid_positive_dist, 0 ,keepdim=True)
        


        # Get the hardest negative pairs
        mask_neg=_get_anchor_negative_triplet_mask(relations, labels).float()
        
        max_anchor_negative_dist, _ = torch.max(relations, 0 , keepdim=True)
        anchor_negative_dist = relations + max_anchor_negative_dist * (1.0 - mask_neg)
        

        hardest_negative_dist, _ = torch.min(anchor_negative_dist, 0 , keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
       
        triplet_loss_all = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        num_hard_triplets = torch.sum(torch.gt(triplet_loss_all, 1e-16).float())
        
        triplet_loss = torch.sum(triplet_loss_all) / (num_hard_triplets+1e-16)
        
        
        
        return triplet_loss
    
class HardTripletLoss_D(nn.Module):
    """Hard/Hardest Triplet Loss
    (pytorch implementation of https://omoindrot.github.io/triplet-loss)

    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    """
    def __init__(self, margin = 1,cos_sim = False):
        """
        Args:
            margin: margin for triplet loss
        """
        super(HardTripletLoss_D, self).__init__()
        self.margin = margin
        self.cos_sim = cos_sim
        
    def forward(self, attributes, embeddings, labels):
        """
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        if self.cos_sim:
            relations = _pairwise_distance_cos(attributes, embeddings ,labels)
        else:
            relations = _pairwise_distance(attributes, embeddings ,labels)
            
        #print(relations.size())
        # Get the hardest positive pairs
        mask_pos=_get_anchor_positive_triplet_mask(relations, labels).float()
        #print(mask_pos)
        valid_positive_dist = relations * mask_pos
        hardest_positive_dist, hardest_positive_dist_index  = torch.max(valid_positive_dist, 0 ,keepdim=True)
        hardest_positive_dist_img, _ = torch.max(valid_positive_dist, 1 ,keepdim=True)
        #print(hardest_positive_dist_img.size())

        
        
        # Get the hardest negative pairs
        mask_neg=_get_anchor_negative_triplet_mask(relations, labels).float()
        #print(mask_neg)
        max_anchor_negative_dist, _ = torch.max(relations, 0 , keepdim=True)
        max_anchor_negative_dist_img, _ = torch.max(relations, 1 , keepdim=True)
        
        anchor_negative_dist = relations + max_anchor_negative_dist * (1.0 - mask_neg)
        anchor_negative_dist_img = relations + max_anchor_negative_dist_img * (1.0 - mask_neg)
        
        hardest_negative_dist, _ = torch.min(anchor_negative_dist, 0 , keepdim=True)
        hardest_negative_dist_img = torch.zeros_like(hardest_negative_dist)
        #print(hardest_negative_dist_img.size())
        for i in range(hardest_positive_dist_index.size()[1]):
            hardest_negative_dist_img[0][i] = torch.min(anchor_negative_dist_img[hardest_positive_dist_index[0][i]])
        
        triplet_loss_all = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        triplet_loss_all_img = F.relu(hardest_positive_dist - hardest_negative_dist_img + self.margin)
        num_hard_triplets = torch.sum(torch.gt(triplet_loss_all, 1e-16).float())
        num_hard_triplets_img = torch.sum(torch.gt(triplet_loss_all_img, 1e-16).float())
        
        triplet_loss = (torch.sum(triplet_loss_all) + torch.sum(triplet_loss_all_img))/(num_hard_triplets+num_hard_triplets_img+1e-16)
        
        return triplet_loss
def _pairwise_distance(bat_attributes, bat_features, bat_lables):
    # Compute the 2D matrix of distances between all the embeddings.
 
    distances = F.pairwise_distance(bat_features,bat_attributes,2)

    distances = (distances.view(bat_lables.size()[0],-1))
    return distances
def _pairwise_distance_cos(bat_attributes, bat_features, bat_lables):
    # Compute the 2D matrix of distances between all the embeddings.
   
    distances = F.cosine_similarity(bat_features,bat_attributes)
    
    distances = (distances.view(bat_lables.size()[0],-1))
    return distances
def _get_anchor_positive_triplet_mask(relations,labels):
    # Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mask_pos = torch.zeros_like(relations).to(device).byte()
    for i in range(relations.size()[0]):
        mask_pos[i][labels[i]]=1

    return mask_pos
def _get_anchor_negative_triplet_mask(relations,labels):
    # Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Check if labels[i] != labels[k]
    mask_neg = torch.ones_like(relations).to(device).byte()
    #print('-'*100)
    for i in range(relations.size()[0]):
        mask_neg[i][labels[i]]=0
    return mask_neg


def euclid_pairwise_distance(x, y ,squared=True, eps=1e-16):
    # Compute the 2D matrix of distances between all the embeddings.

    cor_mat = torch.matmul(x, y.t())
    norm_mat = cor_mat.diag()
    distances = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)
    distances = distances
    
    if not squared:
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * eps
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances
