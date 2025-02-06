import torch
import torch.nn.functional as F

def sim(x, y):
    """
    Tính độ tương đồng giữa hai vectơ x, y
    
    - x: Tensor (N, D), batch của N vectơ đầu vào
    - y: Tensor (M, D), batch của M vectơ so sánh
    
    Trả về:
    - sim: Tensor (N, M), ma trận độ tương đồng giữa x và y
    """
    return F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=-1)

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
    
    # Tạo batch descriptions tương ứng với từng mẫu trong reps
    desc_list = torch.stack([descriptions[int(label)][0] for label in targets]).to(device)  # (N, D)
    
    # Tạo batch tất cả descriptions
    all_descriptions = torch.stack([des[0] for des in descriptions.values()]).to(device)  # (M, D)
    
    # Tính cosine similarity giữa reps và descriptions
    similarities = sim(reps, all_descriptions) / temperature  # (N, M)
    
    # Lấy similarity giữa reps và mô tả tương ứng
    pos_sim = sim(reps, desc_list).diag()  # (N,)
    
    # Tính loss theo công thức -log(sim(x, des(x)) / sum(sim(x, des)))
    loss = -torch.log(torch.exp(pos_sim) / torch.sum(torch.exp(similarities), dim=1))
    
    return loss.mean()
