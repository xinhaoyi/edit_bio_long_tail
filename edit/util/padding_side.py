from transformers import AutoTokenizer
import torch

def detect_padding_direction(tokenizer):
    # 检查tokenizer是否有padding_side属性
    try:
        if hasattr(tokenizer, 'padding_side'):
            return tokenizer.padding_side
    except Exception as e:
        print(f"Error accessing tokenizer's padding_side attribute: {e}")

    # 如果没有padding_side属性，或访问失败，则尝试通过编码示例文本推断
    try:
        sample_text = "Sample Text"
        encoded = tokenizer.encode_plus(sample_text, return_tensors='pt', max_length=10, padding='max_length', truncation=True)
        attention_mask = encoded['attention_mask'][0]

        # 检查是否存在填充
        if torch.all(attention_mask == 1):
            return 'none'  # 所有位置都是有效的，没有填充

        # 根据attention_mask推断填充方向
        if attention_mask[0] == 0:
            return 'left'
        elif attention_mask[-1] == 0:
            return 'right'
    except Exception as e:
        print(f"Error during encoding or padding direction inference: {e}")

    # 如果以上方法都无法确定，返回'unknown'
    return 'unknown'

def demo():
    # 示例
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 调用函数
    padding_direction = detect_padding_direction(tokenizer)
    print(f"Detected padding direction: {padding_direction}")