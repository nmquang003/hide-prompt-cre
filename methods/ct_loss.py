import torch
import torch.nn.functional as F

def sim(A, B):
    """
    Tính cosine similarity giữa 2 tensors a và b
    - a: tensor (N, D)
    - b: tensor (N, M, D)
    """
    # Chuẩn hóa vector A và B
    
    A = F.normalize(A, p=2, dim=-1) # (N, D)
    B = F.normalize(B, p=2, dim=-1) # (N, M, D)
    
    # Tính cosine similarity giữa A và B
    return torch.einsum("nd,nmd->nm", A, B) # (N, M)


def constractive_loss(reps, pos_reps, neg_reps, temperature=5):
    """
    Hàm loss cho mô hình Contrastive Learning
    - reps: tensor (N, D) - Biểu diễn của các điểm dữ liệu
    - pos_reps: tensor (N, P) - Biểu diễn của các điểm dữ liệu dương
    - neg_reps: tensor (N, Q) - Biểu diễn của các điểm dữ liệu âm
    """
    # Tính cosine similarity giữa reps và pos_reps
    pos_sims = torch.exp(sim(reps, pos_reps).mean(dim=1) / temperature)
    # Tính cosine similarity giữa reps và neg_reps
    neg_sims = torch.exp(sim(reps, neg_reps).mean(dim=1) / temperature)
    # Tính tỷ lệ softmax giữa pos_sims và neg_sims
    softmax = pos_sims / (pos_sims + neg_sims)
    # Tính loss
    loss = -torch.log(softmax).mean()
    return loss