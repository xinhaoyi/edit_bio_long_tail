import torch

def get_position_ids(attention_mask):
    
    assert len(attention_mask.shape) == 2 # attention mask 必须是二维的
    batch_size, seq_len = attention_mask.shape
    
    assert batch_size > 1
    
    sum_attention_mask = torch.sum(attention_mask, dim=0)
    if sum_attention_mask[0] == batch_size:
        # 右 padding
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    elif sum_attention_mask[-1] == batch_size:
        # 左 padding 
        position_ids_list = []
        for i in range(batch_size):
            num_tokens = attention_mask[i].sum()
            position_ids_list.append([0] * (seq_len-num_tokens) + list(range(num_tokens)))
        position_ids = torch.tensor(position_ids_list, dtype=torch.long)
    else:
        print("Could not detect left padding or right padding, setting the position_ids to None!")
        position_ids = None
    
    return position_ids