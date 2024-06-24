import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from models import GPT_2


def tok_k_logits(logits, k):
    v, ix = torch.topk(logits,k)
    out = logits.clone()
    out[out < v[:,[-1]]] = -float('inf')
    return out


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')

    block_size = 512
    n_embed = 512
    n_heads = 8
    n_layers = 8
    dropout_ratio = 0.1
    topk = 10

    # load model
    model = GPT_2(tokenizer.vocab_size, block_size, n_embed, n_heads, n_layers, dropout_ratio)
    model.load_state_dict(torch.load('weights/epoch_35_loss_1.03.pt', map_location='cpu'))
    model.eval()
    model.to(device)

    # input sentence
    input_sentence = '张三：一位公司技术高管，项目经验丰富，对技术架构设计有独特的见解。 李四：一位公司底层技术职员，负责简单的架构设计和维护。 生成一段他们的对话内容。'
    print('Prompt:' + '\n' + input_sentence + '\n' + 'Dialogue:')

    prompt_token = tokenizer(input_sentence)['input_ids']
    indexes = prompt_token + [tokenizer.cls_token_id]
    # Start decoding
    for _ in range(n_embed):
        input_token = torch.tensor([indexes]).to(device)
        # Decoder forward pass
        logits = model(input_token)
        logits = logits[:, -1, :]
        logits = tok_k_logits(logits, topk)
        # Forward to linear classify token in vocab and Softmax
        probs = F.softmax(logits, dim=-1)

        index = torch.multinomial(probs, num_samples=1)
        if index.item() == tokenizer.sep_token_id:
            break

        indexes.append(index.item())
        print(tokenizer.decode(index.item()), end='')

if __name__ == "__main__":
    main()