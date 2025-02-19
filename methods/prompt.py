import torch
import torch.nn as nn
import torch.nn.functional as F

class Prompt(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.length = args.prompt_length
        self.embed_dim = args.prompt_embed_dim
        self.pool_size = args.prompt_pool_size
        self.top_k = args.prompt_top_k
        # ensure that the top_k is less than the pool_size
        if self.top_k > self.pool_size:
            self.top_k = self.pool_size
        self.prompt_type = args.prompt_type
        
        # total prompt length
        self.total_prompt_length = self.top_k * self.length

        # prompt
        prompt_pool_shape = (self.pool_size, self.length, self.embed_dim)
        if args.prompt_init == "zero":
            self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
        elif args.prompt_init == "uniform":
            self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
            nn.init.uniform_(self.prompt, -1, 1)
        else:
            raise Exception("Not support type of prompt initialization")

        # prompt_key
        key_shape = (self.pool_size, self.embed_dim * 2)
        if args.prompt_key_init == "zero":
            self.prompt_key = nn.Parameter(torch.zeros(key_shape))
        elif args.prompt_key_init == "uniform":
            self.prompt_key = nn.Parameter(torch.randn(key_shape))
            nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            raise Exception("Not support type of prompt key initialization")                                        


    # def forward(self, x_embed, x_key=None):
    #     out = dict()
        
    #     # Chuẩn hóa vector khóa
    #     prompt_key_norm = nn.functional.normalize(self.prompt_key, dim=1)  # (pool_size, embed_dim)
    #     x_key_norm = nn.functional.normalize(x_key, dim=1)  # (batch_size, embed_dim)

    #     # Tính độ tương đồng cosine giữa x_key và prompt_key
    #     similarity = torch.matmul(x_key_norm, prompt_key_norm.t())  # (batch_size, pool_size)
    #     _, _id = torch.topk(similarity, k=self.top_k, dim=1)  # (batch_size, top_k)

    #     # Tính softmax theo pool_size
    #     softmax_sim = F.softmax(similarity, dim=1)  # (batch_size, pool_size)

    #     # Reshape prompt từ (pool_size, length, embed_dim) -> (pool_size, length * embed_dim)
    #     reshaped_prompt = self.prompt.view(self.pool_size, -1)  # (pool_size, length * embed_dim)

    #     # Mở rộng softmax_sim để phù hợp với reshaped_prompt
    #     softmax_sim = softmax_sim.unsqueeze(2).expand(
    #         x_embed.shape[0], self.pool_size, reshaped_prompt.shape[-1]
    #     )  # (batch_size, pool_size, length * embed_dim)

    #     # Nhân softmax_sim với prompt để tạo result
    #     result = softmax_sim * reshaped_prompt.unsqueeze(0)  # (batch_size, pool_size, length * embed_dim)

    #     # Lấy trung bình theo pool_size để thu gọn prompt
    #     mean_result = torch.mean(result, dim=1)  # (batch_size, length * embed_dim)

    #     # Tính độ tương đồng tổng hợp
    #     x_key_norm = x_key_norm.unsqueeze(1)  # (batch_size, 1, embed_dim)
    #     out["reduce_sim"] = torch.sum(prompt_key_norm[_id] * x_key_norm) / x_embed.shape[0]  # (scalar)

    #     # Reshape mean_result về dạng 3D
    #     mean_result_reshaped = mean_result.view(x_embed.shape[0], self.length, self.embed_dim)  
    #     # (batch_size, length, embed_dim)

    #     # Kết hợp embedding của prompt và x_embed
    #     out["prompted_embedding"] = torch.cat(
    #         [mean_result_reshaped, x_embed], dim=1
    #     )  # (batch_size, length + x_embed.shape[1], embed_dim)

    #     return out

    
    # def forward(self, x_embed, x_key=None):
    #     out = dict()

    #     prompt_key_norm = nn.functional.normalize(self.prompt_key, dim=1)
    #     x_key_norm = nn.functional.normalize(x_key, dim=1)

    #     similarity = torch.matmul(x_key_norm, prompt_key_norm.t())
    #     _, _id = torch.topk(similarity, k=self.top_k, dim=1)

    #     batched_prompt = self.prompt[_id].reshape(-1, self.total_prompt_length, self.embed_dim)

    #     # Put pull_constraint loss calculation inside
    #     x_key_norm = x_key_norm.unsqueeze(1)
    #     out["reduce_sim"] = torch.sum(prompt_key_norm[_id] * x_key_norm) / x_embed.shape[0]
    #     out["prompted_embedding"] = torch.cat([batched_prompt, x_embed], dim=1)
    #     return out
    
    # forward của quangnm
    def forward(self, x_embed, x_key=None):
        out = dict()
        
        # Chuẩn hóa vector khóa
        prompt_key_norm = nn.functional.normalize(self.prompt_key, dim=1)  # (pool_size, embed_dim)
        x_key_norm = nn.functional.normalize(x_key, dim=1)  # (batch_size, embed_dim)

        # Tính độ tương đồng và lấy top-k giá trị lớn nhất
        similarity = torch.matmul(x_key_norm, prompt_key_norm.t())  # (batch_size, pool_size)
        topk_values, topk_indices = torch.topk(similarity, k=self.top_k, dim=1)  # (batch_size, top_k)

        # Tính softmax chỉ trên top-k
        softmax_sim = F.softmax(topk_values, dim=1)  # (batch_size, top_k)

        # Lấy top-k prompt tương ứng (batch-wise indexing)
        selected_prompts = self.prompt[topk_indices]  # (batch_size, top_k, length, embed_dim)

        # Nhân trọng số softmax với các prompt đã chọn
        softmax_sim = softmax_sim.unsqueeze(-1).unsqueeze(-1)  # (batch_size, top_k, 1, 1)
        weighted_prompts = softmax_sim * selected_prompts  # (batch_size, top_k, length, embed_dim)

        # Tổng hợp top-k prompt bằng cách lấy trung bình
        mean_result = weighted_prompts.mean(dim=1)  # (batch_size, length, embed_dim)

        # Tính độ tương đồng tổng hợp
        x_key_norm = x_key_norm.unsqueeze(1)  # (batch_size, 1, embed_dim)
        out["reduce_sim"] = torch.sum(prompt_key_norm[topk_indices] * x_key_norm) / x_embed.shape[0]  # (scalar)

        # Kết hợp embedding của prompt và x_embed
        out["prompted_embedding"] = torch.cat([mean_result, x_embed], dim=1)  
        # (batch_size, length + x_embed.shape[1], embed_dim)

        return out
