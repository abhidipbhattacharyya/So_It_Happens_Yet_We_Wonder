import json
from transformers import AutoTokenizer, AutoModel
import random
import torch
from collections import defaultdict, namedtuple
import json
import glob
from typing import Union
import re
from tqdm import tqdm

random.seed(42)
Tokenized = namedtuple("Tokenized", ["base", "src", "alignment_base", "alignment_src", "base_label", "src_label"])


class Pair:
    """
    A pair of sentences where all features except one are held constant.

    Each pair has a base sentence and a source sentence. These two sentences
    have different "types" (the value of the differing feature) and different
    "labels" (expected continuation of the sentence).
    """

    def __init__(self, base: str, src: str, base_label: str, src_label: str):
        #print(base)
        if isinstance(base,str):
            self.base = base.split(" ")
            self.src = src.split(" ")
            self.base = [self.base[0]]+[f' {b}' for b in self.base[1:]]
            self.src = [self.src[0]]+[f' {s}' for s in self.src[1:]]
            length = min(len(self.base), len(self.src))
            self.base = self.base[:length]
            self.src = self.src[:length]
            
        else:#already list of words
            self.base = base
            self.src = src
        
        self.base_label = base_label
        self.src_label = src_label
        self.pos_tok_map = {}
    

    def tokenize(self, tokenizer: AutoTokenizer, device: str="cpu") -> Tokenized:
        """Tokenize the pair and produce alignments."""
        alignment_base, alignment_src = [], []
        pos_base, pos_src = 0, 0
        blank_pos=False
        for span_i in range(len(self.base)):

            # get span lengths in tokens
            tok_base = tokenizer.tokenize(self.base[span_i])
            tok_src = tokenizer.tokenize(self.src[span_i])
            alignment = [
                list(range(pos_base, pos_base + len(tok_base))),
                list(range(pos_src, pos_src + len(tok_src)))
            ]
            if ' _' in self.base[span_i]:
                blank_pos =True
                self.pos_tok_map['blank'] = span_i
                for npi in range(1,5):
                    self.pos_tok_map[f'blank+{npi}'] = span_i+npi
                
            
            # update positions
            alignment_base.append(alignment[0])
            alignment_src.append(alignment[1])
            pos_base += len(tok_base)
            pos_src += len(tok_src)

        # tokenize full pair and return
        base_tok = tokenizer(''.join(self.base), return_tensors="pt", padding=True).to(device)
        src_tok = tokenizer(''.join(self.src), return_tensors="pt", padding=True).to(device)

        #print(self.base_label)
        #print('---------')
        bl_tok = tokenizer.encode(self.base_label[0])[0]
        #tokenizer(self.base_label,return_tensors="pt",add_special_tokens=False)["input_ids"].squeeze(-1)[0].item()
        sl_tok = tokenizer.encode(self.src_label[0])[0]
        #tokenizer(self.src_label,return_tensors="pt",add_special_tokens=False)["input_ids"].squeeze(-1)[0].item()
        assert blank_pos
        return Tokenized(base=base_tok, src=src_tok, alignment_base=alignment_base, alignment_src=alignment_src, base_label=bl_tok, src_label= sl_tok)
    

    def swap(self) -> "Pair":
        """Swap the base and src sentences."""
        return Pair(self.src, self.base, self.src_label, self.base_label)

    
    def __repr__(self):
        return f"Pair('base:{self.base}' > '{self.base_label}',\n'Src:{self.src}' > '{self.src_label}\n Baselen:{len(self.base)}\n srclen:{len(self.src)}\n-----------')"


class Batch:
    """
    A Batch is a collection of Pairs that have been tokenized and padded.
    The messy part is figuring out where to do interventions, so a Batch
    encapsulates the functions for computing pos_i for the intervention
    at inference time, using the tokenized pair and alignments.
    """

    def __init__(self, pairs: list[Pair], tokenizer: AutoTokenizer, device: str="cpu"):
        self.pairs = pairs

        # tokenize base and src
        #for pair in pairs:
            #print(pair)
        tokenized = [pair.tokenize(tokenizer, device) for pair in pairs]
        max_len = max([max(x.base.input_ids.shape[-1], x.src.input_ids.shape[-1]) for x in tokenized])
        self.base = self._stack_and_pad([x.base for x in tokenized], max_len=max_len)
        self.src = self._stack_and_pad([x.src for x in tokenized], max_len=max_len)
        self.alignment_base = [x.alignment_base for x in tokenized]
        self.alignment_src = [x.alignment_src for x in tokenized]
        
        # labels
        self.base_labels = torch.LongTensor([x.base_label for x in tokenized]).to(device)
        self.src_labels = torch.LongTensor([x.src_label for x in tokenized]).to(device)
        #self.base_types = [pair.base_type for pair in pairs]
        #self.src_types = [pair.src_type for pair in pairs]
        self.cached_pos = {}
    
    
    def _pos_bounds(self, span1: list[int], span2: list[int]) -> list[int]:
        """Compute the bounds of a span."""
        if self.pos_strategy == "first":
            return span1[:1], span2[:1]
        elif self.pos_strategy == "last":
            return span1[-1:], span2[-1:]
        elif self.pos_strategy == "all":
            max_len = max(len(span1), len(span2))
            return span1 + [span1[-1]] * (max_len - len(span1)), span2 + [span2[-1]] * (max_len - len(span2))
    

    def compute_pos(self, strategy: str) -> torch.LongTensor:
        """Compute pos alignments as tensors."""
        # shape of alignment: [batch_size, 2, num_spans, tokens_in_span]
        # not a proper tensor though! tokens_in_span is variable, rest is constant
        if strategy in self.cached_pos:
            return self.cached_pos[strategy]
        self.pos_strategy = strategy
        assert self.pos_strategy in ["first", "last", "all"]
        rets_base, rets_src = [], []
        for batch_i in range(len(self.pairs)):
            ret_base, ret_src = [], []
            st_span = self.pairs[batch_i].pos_tok_map['blank']
            for span_i in range(st_span, st_span+5):#range(len(self.alignment_src[batch_i])):
                # skip null alignments
                if len(self.alignment_base[batch_i][span_i]) == 0 or len(self.alignment_src[batch_i][span_i]) == 0:
                    ret_base.append([-1])
                    ret_src.append([-1])
                else:
                    bounds = self._pos_bounds(self.alignment_base[batch_i][span_i], self.alignment_src[batch_i][span_i])
                    ret_base.append(bounds[0])
                    ret_src.append(bounds[1])
            rets_base.append(ret_base)
            rets_src.append(ret_src)
        
        # shape: [2, batch_size, length, 1]
        # dim 0 -> src, base (the intervention code wants src first)
        ret = [rets_src, rets_base]
        self.cached_pos[strategy] = ret
        return ret


    def _stack_and_pad(self, input_list: dict, pad_token: int=0, max_len: int=100) -> dict:
        """Stack and pad a list of tensors outputs from a tokenizer."""
        input_ids = torch.stack([torch.nn.functional.pad(x.input_ids[0], (0, max_len - x.input_ids.shape[-1]), mode='constant', value=pad_token) for x in input_list])
        attention_mask = torch.stack([torch.nn.functional.pad(x.attention_mask[0], (0, max_len - x.attention_mask.shape[-1]), mode='constant', value=0) for x in input_list])
        return {"input_ids": input_ids, "attention_mask": attention_mask}


    def __repr__(self):
        return f"Batch({len(self.pairs)} pairs)\n" + "\n".join([f"  {pair}" for pair in self.pairs])


class Dataset:
    """
    A Dataset is a template for generating minimal pairs that is loaded
    from a JSON specification.

    We importantly want examples generated from a dataset to include token-
    level alignments.
    """

    templates: list[str]
    label_vars: list[str]
    labels: dict[str, list[str]]
    variables: dict[str, Union[list[str], dict[str, list[str]]]]
    result_prepend_space: bool


    def __init__(self, data: dict, seed: int=42, shuffle:bool=False, inverse:bool=False):
        # load basics
        random.seed(seed)
        random.shuffle(data["prompts"]) if shuffle else None
       
        if not inverse:
            self.clean_prompts = [d["clean"] for d in data["prompts"]]
            self.corrupt_prompts = [d["corrupt"] for d in data["prompts"]]
            self.answer_strs = [d["answers"] for d in data["prompts"]]
            self.wrong_answer_strs = [d["wrong_answers"] for d in data["prompts"]]
        else:
            self.clean_prompts = [d["corrupt"] for d in data["prompts"]]
            self.corrupt_prompts = [d["clean"] for d in data["prompts"]]
            self.answer_strs = [d["wrong_answers"] for d in data["prompts"]]
            self.wrong_answer_strs = [d["answers"] for d in data["prompts"]]
        

    @classmethod
    def load_from(self, file_path, inverse=False) -> "Dataset":
        """Load a Dataset from a json template."""
        
        data = json.load(open(file_path, "r"))#['prompts']
        return Dataset(data, inverse)
    

    @property
    def length(self) -> int:
        return len(self.clean_prompts)
    

    def sample_pair(self) -> Pair:
        """Sample a minimal pair from the dataset."""
        # pick types (should differ)
        random_integer = random.randint(0, self.length-1)
        base = self.clean_prompts[random_integer]
        src = self.corrupt_prompts[random_integer]
        base_label =  self.answer_strs[random_integer]
        src_label = self.wrong_answer_strs[random_integer]

        return Pair(base, src,base_label, src_label)
    

    @torch.no_grad()
    def _sample_doable_pair(self, model: AutoModel, tokenizer: AutoTokenizer, device: str="cpu", discard: set[str]=set()) -> Pair:
        """Sample a minimal pair from the dataset that is correctly labelled by a model."""

        # keep resampling until we get a pair that is correctly labelled
        correct, ct = False, 0
        while not correct:
            pair = self.sample_pair(discard)
            if ''.join(pair.base) in discard:
                continue
            base = tokenizer(''.join(pair.base), return_tensors="pt").to(device)
            src = tokenizer(''.join(pair.src), return_tensors="pt").to(device)
            base_logits = model(**base).logits[0, -1]
            src_logits = model(**src).logits[0, -1]
            base_label = tokenizer.encode(pair.base_label)[0]
            src_label = tokenizer.encode(pair.src_label)[0]
            if base_logits[base_label] > base_logits[src_label] and src_logits[src_label] > src_logits[base_label]:
                correct = True
            ct += 1
            if ct == 20 and not correct:
                print("WARNING: could not find a doable pair after 20 iterations")
                print("Using a random pair instead")
                break
            
        return pair


    def sample_batch(
            self, tokenizer: AutoTokenizer, batch_size: int, device: str="cpu",
            model: Union[AutoModel, None]=None, discard: set[str]=set(),
            manipulate: Union[str, None]=None) -> Batch:
        """Sample a batch of minimal pairs from the dataset."""
        pairs = []

        # get the pairs
        if model is None:
            while len(pairs) < batch_size // 2:
                pair = self.sample_pair()
                ct = 0
                while ''.join(pair.base) in discard:
                    pair = self.sample_pair()
                    ct += 1
                    if ct == 20:
                        print("WARNING: could not find a pair not in the discard set after 20 iterations")
                        print("Using a random pair instead")
                        break
                pairs.append(pair)
        else:
            pairs = [
                self._sample_doable_pair(model, tokenizer, device, discard)
                for _ in range(batch_size // 2)
            ]

        
        
        # add flipped pairs
        for i in range(batch_size // 2):
            pairs.append(pairs[i].swap())
        
        # make batch
        return Batch(pairs, tokenizer, device)


    def sample_batches(
            self, tokenizer: AutoTokenizer, batch_size: int, num_batches: int,
            device: str="cpu", seed: int=42, model: Union[AutoModel, None]=None,
            discard: set[str]=set(), manipulate: Union[str, None]=None) -> list[Batch]:
        """Sample a list of batches of minimal pairs from the dataset."""
        random.seed(seed)
        return [self.sample_batch(tokenizer, batch_size, device, model, discard, manipulate) for _ in range(num_batches)]






# ===== MODIFIED ======
def get_combined_batch(datasources, tokenizer, batch_size, steps, device, seed, manipulate, discard = []):
    """Get a combined batch from multiple datasets."""
    out_batches = {}
    temp_batches = [data.sample_batches(tokenizer, batch_size, steps, device, seed=seed, manipulate=manipulate, discard = discard) for data in datasources]
    
    for pos_i in range(datasources[0].first_var_pos, datasources[0].length):
        
        out_batches[pos_i] = []
        for idx in range(steps):
            pairs = []
            for batch_list in temp_batches:
                for pair in batch_list[idx].pairs:
                    if len(pair.base[pos_i]) > 0 and len(pair.src[pos_i]) > 0:
                        pairs.append(pair)
            
            subset = random.sample(pairs, k=batch_size)

            new_batch = Batch(subset, tokenizer, device)
            out_batches[pos_i].append(new_batch)
        
    return out_batches
# ===== END MODIFIED ======

if __name__ == "__main__":
    convert_to_huggingface_format()