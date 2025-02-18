import torch
import torch.nn.functional as F

# Giả sử có 1 tensor prompt_key ban đầu với 5 vectors (5, 3)
self_prompt_key = torch.tensor([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
    [1.0, 1.1, 1.2],
    [1.3, 1.4, 1.5]
], dtype=torch.float32)

# Giả sử x_key có 2 vectors (2, 3)
x_key = torch.tensor([
    [0.2, 0.3, 0.4],
    [0.6, 0.7, 0.8]
], dtype=torch.float32)

# Chuẩn hóa vectors
prompt_key_norm = F.normalize(self_prompt_key[1:,], dim=1)  # Bỏ phần tử đầu tiên
x_key_norm = F.normalize(x_key, dim=1)

# Tính similarity giữa x_key_norm và prompt_key_norm
similarity = torch.matmul(x_key_norm, prompt_key_norm.T)
softmax_sim = F.softmax(similarity, dim=1) * 0.5

# Chọn top-k-1 giá trị lớn nhất từ similarity (giả sử self.top_k = 3 => k=2)
_, _id = torch.topk(similarity, k=2, dim=1)  # Lấy chỉ số của 2 giá trị lớn nhất

# Giả sử x_embed.shape[0] = x_key.shape[0]
x_embed_shape_0 = x_key.shape[0]

# Vì prompt_key_norm đã bị dịch đi 1 đơn vị, ta cần cộng thêm 1 vào _id
reduce_sim = torch.sum(self_prompt_key[_id + 1] * x_key_norm.unsqueeze(1), dim=[1, 2]) / x_embed_shape_0

# Hiển thị kết quả
print(softmax_sim)
print(similarity)
print(prompt_key_norm)
print(_id)
print(reduce_sim)