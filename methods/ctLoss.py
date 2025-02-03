import torch
import torch.nn.functional as F

def sim(a, b):
    """
    Tính cosine similarity giữa 2 tensors a và b
    - a, b: tensor (N, D)
    """
    return F.cosine_similarity(a, b, dim=-1)

def contrastive_loss(reps, pos_reps, neg_reps):
    """
    Hàm loss cho mô hình Contrastive Learning
    - reps: tensor (N, D) - Biểu diễn của các điểm dữ liệu
    - pos_reps: tensor (N, P) - Biểu diễn của các điểm dữ liệu dương
    - neg_reps: tensor (N, Q) - Biểu diễn của các điểm dữ liệu âm
    """
    # Tính cosine similarity giữa reps và pos_reps
    pos_sim = sim(reps, pos_reps)
    # Tính cosine similarity giữa reps và neg_reps
    neg_sim = sim(reps, neg_reps)
    
    # Tính loss
    loss = torch.mean(-torch.log(torch.sigmoid(pos_sim)) - torch.log(1 - torch.sigmoid(neg_sim)))
    return loss