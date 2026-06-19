import torch
from torch.utils.data import Dataset
from spiraton.data.aba_parser import ABAParser

import sys
import os

TOKENIZER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Tokenizer/python"))
if TOKENIZER_DIR not in sys.path:
    sys.path.insert(0, TOKENIZER_DIR)

from spiraton_tokenizer.spiraton_v4 import SpiratonTokenizerV4

class ABADataset(Dataset):
    def __init__(self, corpus_path: str, max_tokens_per_segment: int = 16):
        self.corpus_path = corpus_path
        self.max_tokens = max_tokens_per_segment
        self.tokenizer = SpiratonTokenizerV4()
        
        self.samples = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                parsed = ABAParser.parse_line(line)
                if parsed:
                    self.samples.append(parsed)
                    
    def __len__(self):
        return len(self.samples)
        
    def _tokenize_segment(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer.tokenize(text, max_tokens=self.max_tokens)
        vecs = []
        for t in tokens:
            vecs.append(torch.tensor(t["vector33d"], dtype=torch.float32))
            
        if not vecs:
            return torch.zeros((self.max_tokens, 33), dtype=torch.float32)
            
        tensor_vecs = torch.stack(vecs)
        seq_len = tensor_vecs.shape[0]
        if seq_len < self.max_tokens:
            pad = torch.zeros((self.max_tokens - seq_len, 33), dtype=torch.float32)
            tensor_vecs = torch.cat([tensor_vecs, pad], dim=0)
        else:
            tensor_vecs = tensor_vecs[:self.max_tokens]
            
        return tensor_vecs
        
    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        
        vec_a = self._tokenize_segment(sample["segments"]["A"]["text"])
        vec_b = self._tokenize_segment(sample["segments"]["B"]["text"])
        vec_a_prime = self._tokenize_segment(sample["segments"]["A_PRIME"]["text"])
        
        return {
            "op": sample["op"],
            "A": vec_a,
            "B": vec_b,
            "A_PRIME": vec_a_prime,
            "chiralite_A": sample["segments"]["A"]["chiralite"],
            "chiralite_A_PRIME": sample["segments"]["A_PRIME"]["chiralite"]
        }
