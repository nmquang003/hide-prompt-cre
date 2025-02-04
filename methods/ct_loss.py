import torch
import torch.nn.functional as F

def sim(a, b):
    """
    Tính cosine similarity giữa 2 tensors a và b
    - a, b: tensor (N, D)
    """
    return F.cosine_similarity(a, b, dim=-1)

def CT_loss(reps, pos_reps, neg_reps, temperature=0.5):
    """
    Hàm loss cho mô hình Contrastive Learning
    - reps: tensor (N, D) - Biểu diễn của các điểm dữ liệu
    - pos_reps: tensor (N, P) - Biểu diễn của các điểm dữ liệu dương
    - neg_reps: tensor (N, Q) - Biểu diễn của các điểm dữ liệu âm
    """
    # Tính cosine similarity giữa reps và pos_reps
    pos_sims = torch.exp(sim(reps, pos_reps) / temperature)
    # Tính cosine similarity giữa reps và neg_reps
    neg_sims = torch.exp(sim(reps, neg_reps) / temperature)
    # Tính tỷ lệ softmax giữa pos_sims và neg_sims
    softmax = pos_sims / (pos_sims + neg_sims)
    # Tính loss
    loss = -torch.log(softmax).mean()
    return loss