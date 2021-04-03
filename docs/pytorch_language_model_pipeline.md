# PyTorch での言語モデル学習 - 学習パイプライン

このドキュメントでは、PyTorch で言語モデルを学習する際の学習パイプラインの組み方を説明します。

モデルを学習する際には、モデル自体の実装に比べて、データのロードや学習ループの作成といった箇所が大部分を占めます。
そこで、このドキュメントでは、言語モデル自体の詳細に踏み込むことはせず、
それ以外の Dataset, DataLoader, 学習ループの作成に注目することにします。
言語モデルは Hugging Face transformers の [OpenAI GPT2](https://huggingface.co/transformers/model_doc/gpt2.html)
モデルを使うことにし、詳細は触れません。

## 環境のセットアップ

自分の環境の CUDA のバージョンを確認して、[公式ドキュメント](https://pytorch.org/get-started/locally/)に従って
対応するPyTorchをインストールします。
ここではCUDA 11.1対応の Pytorch をインストールします。


```python
!pip3 install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Hugging Face のモデルを利用するために、 Transformers と、トークナイザを利用するために SentencePiece のパッケージをインストールします。


```python
!pip3 install transformers==4.4.2 sentencepiece==0.1.95
```

**Note:** このドキュメントの実行環境は、次のように Docker コンテナ内で実行しています。

```sh
$ docker container run --gpus all --rm -it -v $(pwd):/work -w /work -p 8888:8888 nvidia/cuda:11.2.2-devel-ubuntu20.04 bash
(container)$ apt update && apt install -y python3 python3-pip
(container)$ pip3 install jupyter==1.0.0
(container)$ jupyter notebook --ip 0.0.0.0 --allow-root
```

## トークナイザーの準備

言語モデルの学習には、テキストを ID のリストに変換する必要があります。
今回は、 Hugging Face の model hub で公開している SentencePiece モデルを利用することにします。


```python
import transformers
```


```python
tokenizer = transformers.AutoTokenizer.from_pretrained("colorfulscoop/gpt2-small-ja")
```

一度 tokenizer をインスタンス化すれば、 `encode` と `decode` を通して文字列を ID のリストに、またその逆を行うことができます。


```python
tokenizer.encode("テキストのID化のテスト")
```




    [9069, 8, 6640, 191, 8, 2505]



`encode` してから `decode` を行うと元の文字列に戻ることがわかります。


```python
tokenizer.decode(tokenizer.encode("テキストのID化のテスト"))
```




    'テキストのID化のテスト'



## 学習パイプライン

PyTorch でモデルを学習する際の流れ（ここではこれを **学習パイプライン** と呼ぶことにします）は次のようになります。

1. Dataset の作成
1. DataLoader の作成
1. 学習ループ

Datasetの作成では、モデルへ入力するデータを提供するDatasetを作成します。

DataLoaderの作成では、Datasetが提供するデータを、モデルが効率的に扱えるバッチの形に変換するDataLoaderを作成します。

学習ループでは、DataLoaderから受け取ったバッチに対して、損失の計算グラフの作成、損失の計算グラフから勾配の計算、そしてモデルのパラメータのアップデートのサイクルを回します。

以下では順を追って各ステップを見ていきます。

## Dataset の作成

PyTorch でモデル学習のためにはじめに行うことは DataLoader の作成です。
DatLoader はデータをロードする役目を追っており、言語モデルの場合にはテキストデータを特定の長さの ID のリストに変換する処理を行います。

PyTorch の Dataset は [大きく二つの種類](https://pytorch.org/docs/stable/data.html#dataset-types) があります。

1. Map-style datasets
2. Iterable-style datasets

Map-style datasets は、 `__getitem__()` と `__len__()` を実装した任意のクラスがなり得ます。
データサイズがメモリに余裕を持って乗る場合には扱いやすいクラスです。

一方で Iterable-style dataset は、 `IterableDataset` クラスを継承し、その上で `__iter__()` メソッドを実装する必要があります。
データサイズがメモリに乗らないような場合にはこちらを選択する必要があります。

一般的に近年の言語モデルは数 GB 〜 数十 GB のデータを扱う必要があるため、ここでは Iterable-style dataset を実装することにします。


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
        if not self._drop_last:
            yield ids
```

`BlockDataset` という Iterable-style dataset を実装してみました。
`from_texts` というクラスメソッドでテキストから、`from_file` というクラスメソッドでファイルからテキストを読み込み、tokenizerでID化したのちに、 `block_size` に区切って出力を行います。

簡単な例で動作を確認してみましょう。


```python
sample = "こんにちは。ここでは BlockDataset という Iterable-style datasets を実装してみました。"
sample_dataset = BlockDataset.from_texts(tokenizer=tokenizer, texts=[sample], block_size=5)
[x for x in sample_dataset]
```




    [{'input_ids': [10272, 15, 679, 9, 7], 'labels': [15, 679, 9, 7, 5234]},
     {'input_ids': [5234, 745, 27920, 228, 9723],
      'labels': [745, 27920, 228, 9723, 120]},
     {'input_ids': [120, 1622, 14738, 3291, 2832],
      'labels': [1622, 14738, 3291, 2832, 13081]},
     {'input_ids': [13081, 64, 1199, 531, 1621],
      'labels': [64, 1199, 531, 1621, 4954]},
     {'input_ids': [4954, 2020, 6112, 8341, 19],
      'labels': [2020, 6112, 8341, 19, 16]},
     {'input_ids': [16, 5658, 58, 220, 3914], 'labels': [5658, 58, 220, 3914, 7]}]



言語モデルを学習するので、BlockDataset はモデルへの入力 ID となる `input_ids` の他に、出力側に与える ID を一つずらして `labels` として辞書の形で出力します。
出力された ID を `decode` してどのようになっているか確認してみます。


```python
[{key: tokenizer.decode(val) for key, val in x.items()} for x in sample_dataset]
```




    [{'input_ids': 'こんにちは。', 'labels': 'にちは。ここでは'},
     {'input_ids': 'ここでは BlockData', 'labels': 'BlockDatas'},
     {'input_ids': 'set という Iter', 'labels': 'et という Iterable'},
     {'input_ids': 'able-style', 'labels': '-style d'},
     {'input_ids': 'datasets ', 'labels': 'atasets を'},
     {'input_ids': 'を実装してみました', 'labels': '実装してみました。'}]



この例では見やすいように `blocksize=5` としていますが、実際にはモデルが許容可能な文長を指定します。
[`OpenAI GPT2` の場合には `n_ctx=1024` と指定されている](https://huggingface.co/transformers/model_doc/gpt2.html#transformers.GPT2Config)
ため 1024 の文長を扱うことができます。ですので、1024に設定して学習に使うデータセットを準備します。


```python
train_dataset = BlockDataset.from_file(block_size=1024, tokenizer=tokenizer, filepath="data/train.txt")
valid_dataset = BlockDataset.from_file(block_size=1024, tokenizer=tokenizer, filepath="data/valid.txt")
```

さて、ニューラルネットワークのパラメータ最適化を行う勾配降下法では、学習サンプルをランダムに並べておくことが重要です。
Map-style datasets では、`DataLoader` を作成する際に `shuffle=True` と指定すればいいのですが、
Iterable-style datasets の場合にはサンプルがどの程度あるかわからないため DataLoader でシャッフルはできません。
（Iterable-style datasets で `DataLoader` の `shuffle=True` を指定すると例外が発生します。）

その代わりに、PyTorch は `BufferedShuffleDataset` というデータセットを用意しており、先頭から `buffer_size` 分を
シャッフルして順次データを返すという挙動を行うことができます。
`train_dataset` に対してはこの `BufferedShuffleDataset` を適用してデータをシャッフルします。


```python
shuffled_train_dataset = torch.utils.data.BufferedShuffleDataset(train_dataset, buffer_size=100)
```

さて、これで Dataset の準備は完了です。次は DataLoader を作成にすすみましょう。

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
