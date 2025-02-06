import torch
import torch.nn.functional as F

def sim(x, y):
    """
    Tính độ tương đồng giữa hai vectơ x, y
    
    - x: Tensor (D,), vectơ thứ nhất
    - y: Tensor (D,), vectơ thứ hai
    
    Trả về:
    - sim: Độ tương đồng giữa x, y
    """
    return F.cosine_similarity(x, y)

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
    
    loss = []
    
    for i in range(reps.size(0)):
        pos = sim(reps[i], descriptions[targets[i]])
        negs = [sim(reps[i], des) for des in descriptions.values() if des is not descriptions[targets[i]]]
        
        loss.append(-torch.log(torch.exp(pos / temperature) / torch.sum(torch.exp(torch.tensor(negs) / temperature))))
        
    return torch.mean(torch.tensor(loss))
    
    

    
    


