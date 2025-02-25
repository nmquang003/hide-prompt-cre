import torch
import torch.nn.functional as F

def sim(x, y):
    """
    Tính độ tương đồng cosine giữa hai batch vector x và y.
    """
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    return torch.mm(x, y.T)

def contrastive_loss(reps, targets, descriptions, num_negs=4, temperature=5):
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
    
    # Tính cosine similarity một lần duy nhất
    similarities = sim(reps, all_descriptions) / temperature  # (N, M)

    # Lấy similarity giữa reps và mô tả tương ứng (positive similarity)
    pos_sim = similarities.gather(1, targets.view(-1, 1)).squeeze(1)  # (N,)
    
    # Lấy top-k mô tả gần nhất (negative similarities)
    num_negs = min(num_negs, similarities.size(1) - 1)
    mask = torch.arange(similarities.size(1)).to(device).unsqueeze(0) != targets.unsqueeze(1)
    filtered_similarities = similarities.masked_select(mask).view(similarities.size(0), -1)
    neg_sims, _ = filtered_similarities.topk(num_negs, dim=1)  # (N, num_negs)
    
    # Tính loss theo công thức
    loss = -torch.log(torch.sigmoid(pos_sim.unsqueeze(1) - neg_sims).mean(dim=1) + 1e-8).mean()

    return loss
