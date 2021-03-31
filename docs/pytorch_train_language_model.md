# PyTorch での言語モデル学習 - パイプライン

自分の環境の CUDA のバージョンを確認して、公式ドキュメント に従ってコマンドを実行する。対応した PyTorch をインストールします。
今回は CUDA 11.1 対応の Pytorch をインストールします。


```python
!pip3 install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Hugging Face の Transformers と、SentencePiece トークナイザを利用するためにそのパッケージをインストールします。


```python
!pip3 install transformers==4.4.2 sentencepiece==0.1.95
```

## トークナイザーの準備


```python
import transformers
```


```python
tokenizer = transformers.AutoTokenizer.from_pretrained("colorfulscoop/gpt2-small-ja")
```

## Dataset の作成

PyTorch でモデル学習のためにはじめに行うことは DataLoader の作成です。
DatLoader はデータをロードする役目を追っています。


```python
import torch


class BlockDataset(torch.utils.data.IterableDataset):
    def __init__(self, tokenizer, generator, block_size, drop_last=True):
        super().__init__()
        self._block_size = block_size
        self._tokenizer = tokenizer
        self._generator = generator
        self._drop_last = drop_last

    @classmethod
    def from_texts(cls, tokenizer, texts, block_size):
        return cls(tokenizer=tokenizer, generator=lambda: texts, block_size=block_size)

    @classmethod
    def from_file(cls, tokenizer, filepath, block_size):
        return cls(tokenizer=tokenizer,
                   generator=lambda: (line.strip("\n") for line in open(filepath)),
                   block_size=block_size
                  )
    
    def __iter__(self):
        """
            Yields (List[int])
        """
        ids = []
        for text in self._generator():
            ids.extend(self._tokenizer.encode(text))
            while len(ids) >= self._block_size+1:
                yield {"input_ids": ids[:self._block_size], "labels": ids[1:self._block_size+1]}
                ids = ids[self._block_size:]
        #if not self._drop_last:
        #    yield ids
```


```python
train_dataset = BlockDataset.from_file(block_size=1024, tokenizer=tokenizer, filepath="data/train.txt")
valid_dataset = BlockDataset.from_file(block_size=1024, tokenizer=tokenizer, filepath="data/valid.txt")
```


```python
shuffled_train_dataset = torch.utils.data.BufferedShuffleDataset(train_dataset, buffer_size=100)
```

## DataLoader の作成


```python
def collate_fn(item):
    """
    Args:
        item (List[dict[str, List[int]]]): BlockDataset のイテレータが返す辞書のリスト
    Returns:
        (dict[str, torch.Tensor]):
    """
    keys = item[0].keys()
    dic = {
        key: torch.tensor([x[key] for x in item])
        for key in keys
    }
    return dic
```


```python
shuffled_train_loader = torch.utils.data.DataLoader(
    dataset=shuffled_train_dataset,
    batch_size=3,
    collate_fn=collate_fn,
    prefetch_factor=10,
    num_workers=1,
)
valid_loader = torch.utils.data.DataLoader(
    dataset=valid_dataset,
    batch_size=3,
    collate_fn=collate_fn,
    prefetch_factor=10,
    num_workers=1,
)
```


```python
import itertools

for item in itertools.islice(shuffled_train_loader, 1):
    print("Shape", {key: val.shape for key, val in item.items()})
```

    Shape {'input_ids': torch.Size([3, 1024]), 'labels': torch.Size([3, 1024])}



```python

```
