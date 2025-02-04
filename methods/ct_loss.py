import torch
import torch.nn.functional as F

def sim(A, B):
    """
    Tính cosine similarity giữa 2 tensors a và b
    - a, b: tensor (N, D)
    """
    # Chuẩn hóa vector A và B
    A_norm = A / A.norm(dim=1, keepdim=True)  # Shape (m, d)
    B_norm = B / B.norm(dim=1, keepdim=True)  # Shape (n, d)
    
    # Tích vô hướng giữa tất cả các cặp vector
    similarity_matrix = torch.mm(A_norm, B_norm.T)  # Shape (m, n)
    
    return similarity_matrix

def CT_loss(reps, pos_reps, neg_reps, temperature=5):
    """
    Hàm loss cho mô hình Contrastive Learning
    - reps: tensor (N, D) - Biểu diễn của các điểm dữ liệu
    - pos_reps: tensor (N, P) - Biểu diễn của các điểm dữ liệu dương
    - neg_reps: tensor (N, Q) - Biểu diễn của các điểm dữ liệu âm
    """
    # Tính cosine similarity giữa reps và pos_reps
    pos_sims = torch.exp(sim(reps, pos_reps).sum() / temperature)
    # Tính cosine similarity giữa reps và neg_reps
    neg_sims = torch.exp(sim(reps, neg_reps).sum() / temperature)
    # Tính tỷ lệ softmax giữa pos_sims và neg_sims
    softmax = pos_sims / (pos_sims + neg_sims)
    # Tính loss
    loss = -torch.log(softmax).mean()
    return loss