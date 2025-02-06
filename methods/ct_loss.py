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

def contrastive_loss(reps, targets, descriptions, temperature=5):
    """
    Tính loss kiểu -log(sim(x, des(x)) / sim(x, des))
    
    - reps: Tensor (N, D), biểu diễn đặc trưng của các mẫu
    - targets: Tensor (N,), nhãn tương ứng của reps
    - descriptions: Dict[int, Tensor], ánh xạ nhãn đến mô tả (M, D)
    - temperature: Hệ số nhiệt độ để điều chỉnh độ sắc nét của phân phối
    
    Trả về:
    - loss: Giá trị tổn thất trung bình
    """
    device = reps.device
    
    # Tạo tensor description tương ứng với từng mẫu trong reps
    desc_tensors = torch.stack([descriptions[int(label)] for label in targets]).to(device)  # (N, M, D)
    
    # Tính cosine similarity
    similarities = sim(reps, desc_tensors) / temperature  # (N, M)
    
    # Lấy similarity giữa reps và descriptions tương ứng
    pos_sim = similarities[:, 0]  # (N,)
    
    # Tính loss theo công thức: -log(sim(x, des(x)) / sum(sim(x, des)))
    loss = -torch.log(pos_sim / similarities.sum(dim=1))
    
    return loss.mean()


