import torch
from torch.utils.data import Dataset

from datasets import load_dataset
from transformers import AutoTokenizer

from tqdm import tqdm


class GPT2Dataset(Dataset):
    def __init__(self, corpus_path, block_size=1024):
        super(GPT2Dataset, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')
        self.inputs = []
        self.targets = []
        dialogues = load_dataset('parquet', data_files=corpus_path)
        for dialogue in tqdm(dialogues['train']):
            assert dialogue['input'][:6] == 'user: ', 'wrong prompt'
            assert dialogue['input'][-5:] == ' bot:', 'wrong dialogue'

            prompt_token = self.tokenizer(dialogue['input'][6:-5])['input_ids']
            dialogue_token = self.tokenizer(dialogue['output'])['input_ids']

            input = (prompt_token + dialogue_token[:-1])[:block_size]
            target = ([-1 for _ in range(len(prompt_token))] + dialogue_token[1:])[:block_size]

            padding = [self.tokenizer.pad_token_id for _ in range(block_size - len(input))]
            input.extend(padding)
            target.extend(padding)

            self.inputs.append(input)
            self.targets.append(target)

    def __getitem__(self, index):
        input_ids = torch.tensor(self.inputs[index], dtype=torch.long)
        target_ids = torch.tensor(self.targets[index], dtype=torch.long)
        return input_ids, target_ids

    def __len__(self):
        return len(self.inputs)


if __name__ == '__main__':
    corpus_path = 'data/chinese_dialogue_instruction.parquet'
    data = GPT2Dataset(corpus_path)
    print(data[0])