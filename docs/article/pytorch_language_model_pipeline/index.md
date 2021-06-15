---
created_on: 2021/04/03
updated_on: 2021/04/05
---

# PyTorch での言語モデル学習

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

**Note:** このドキュメントの実行環境は、次のように Docker コンテナにより環境をセットアップしています。

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
1. モデルの準備
1. 学習ループ

Datasetの作成では、モデルへ入力するデータを提供するDatasetを作成します。

DataLoaderの作成では、Datasetが提供するデータを、モデルが効率的に扱えるバッチの形に変換するDataLoaderを作成します。

モデルの準備では、学習するモデルを作成します。今回は Hugging Face transformers のモデルを利用しますが、必要に応じて自分でモデルを実装することになります。

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

Note: 予め言語モデルの学習に使うデータを `data/train.txt`, `data/valid.txt` として保存しておいてください。


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

さて、これで Dataset の準備は完了です。次は DataLoader を作成に進みましょう。

## DataLoader の作成

DataLoader は、Dataset の値を受け取り、それをバッチに変形します。
DataLoader はバッチへの変形を自動で行ってくれますが、今回の言語モデルの場合のように、自動の変換方法ではうまくいかない場合もあります。
そのような場合には自身で `collate_fn` という関数を自分で実装し、DataLoader へ渡すことでバッチへの変形方法を指定できます。


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

colalte_fn には、BlockDataset が返すオブジェクトの `batch_size` (`DataLoader` のイニシャライザで指定します) のリストが渡されます。
そのリストを適切な形に変形する処理をかき、その結果を返却します。
今回は `input_ids` と `label` をバッチ化して PyTorch のテンソルの形で返します。

さて、どうなるか小さなサンプルで確かめて見ましょう。


```python
sample_dataloader = torch.utils.data.DataLoader(dataset=sample_dataset, batch_size=2, collate_fn=collate_fn)
[x for x in sample_dataloader]
```




    [{'input_ids': tensor([[10272,    15,   679,     9,     7],
              [ 5234,   745, 27920,   228,  9723]]),
      'labels': tensor([[   15,   679,     9,     7,  5234],
              [  745, 27920,   228,  9723,   120]])},
     {'input_ids': tensor([[  120,  1622, 14738,  3291,  2832],
              [13081,    64,  1199,   531,  1621]]),
      'labels': tensor([[ 1622, 14738,  3291,  2832, 13081],
              [   64,  1199,   531,  1621,  4954]])},
     {'input_ids': tensor([[4954, 2020, 6112, 8341,   19],
              [  16, 5658,   58,  220, 3914]]),
      'labels': tensor([[2020, 6112, 8341,   19,   16],
              [5658,   58,  220, 3914,    7]])}]



辞書の値になっている PyTorch テンソルのサイズを見ることでよりはっきりとバッチ化されていることがわかります。


```python
[{key: val.size() for key, val in x.items()} for x in sample_dataloader]
```




    [{'input_ids': torch.Size([2, 5]), 'labels': torch.Size([2, 5])},
     {'input_ids': torch.Size([2, 5]), 'labels': torch.Size([2, 5])},
     {'input_ids': torch.Size([2, 5]), 'labels': torch.Size([2, 5])}]



基本的にはこれで完了なのですが、効率的なバッチ化のために `prefetch_factor` と `num_workers` を導入しましょう。
学習時に時間がかかる部分はモデルでの計算時間 (e.g. forward, backward, パラメータ更新) のほかに、データロードにかかる時間があります。
データのロードをモデルでの計算と直列に行うと効率が悪いため、データのロードはモデルでの計算とは別に裏で進めておくと効率よく学習が行えます。
そのためのオプションが `prefetch_factor` と `num_workers` です。

`prefetch_factor` でいくつのバッチを事前に作成しておくかを指定でき、 `num_workers` でそのための裏で動かしておくプロセス数を指定します。
これらを合わせると、実際に学習にしようする DataLoader は次のように作成できます。

**Note:** 公式ドキュメントでは [Single- and Multi-process Data Loading](https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading) の箇所で説明されています。


```python
train_loader = torch.utils.data.DataLoader(
    dataset=shuffled_train_dataset,
    batch_size=2,
    collate_fn=collate_fn,
    prefetch_factor=10,
    num_workers=1,
)
valid_loader = torch.utils.data.DataLoader(
    dataset=valid_dataset,
    batch_size=2,
    collate_fn=collate_fn,
    prefetch_factor=10,
    num_workers=1,
)
```


```python
import itertools
[{key: val.size() for key, val in x.items()} for x in itertools.islice(train_loader, 3)]
```




    [{'input_ids': torch.Size([2, 1024]), 'labels': torch.Size([2, 1024])},
     {'input_ids': torch.Size([2, 1024]), 'labels': torch.Size([2, 1024])},
     {'input_ids': torch.Size([2, 1024]), 'labels': torch.Size([2, 1024])}]



## モデルの準備

Hugging Face transformers のモデルの初期化には、まず Config クラスでモデルのレイヤー数といった値を設定したのちに、モデルのクラスに渡してインスタンス化します。
今回は、OpenAI GPT2 の Config クラス `transformers.GPT2` を設定し、その言語モデルである `transformers.GPT2LMHeadModel` をインスタンス化します。


```python
config = transformers.GPT2Config(
    vocab_size=len(tokenizer),
    tokenizer_class="BertGenerationTokenizer",
    bos_token_id=tokenizer.bos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    sep_token_id=tokenizer.sep_token_id,
    cls_token_id=tokenizer.cls_token_id,
    unk_token_id=tokenizer.unk_token_id,
)     
```


```python
config
```




    GPT2Config {
      "activation_function": "gelu_new",
      "attn_pdrop": 0.1,
      "bos_token_id": 2,
      "cls_token_id": 4,
      "embd_pdrop": 0.1,
      "eos_token_id": 3,
      "gradient_checkpointing": false,
      "initializer_range": 0.02,
      "layer_norm_epsilon": 1e-05,
      "model_type": "gpt2",
      "n_ctx": 1024,
      "n_embd": 768,
      "n_head": 12,
      "n_inner": null,
      "n_layer": 12,
      "n_positions": 1024,
      "pad_token_id": 0,
      "resid_pdrop": 0.1,
      "sep_token_id": 5,
      "summary_activation": null,
      "summary_first_dropout": 0.1,
      "summary_proj_to_labels": true,
      "summary_type": "cls_index",
      "summary_use_proj": true,
      "tokenizer_class": "BertGenerationTokenizer",
      "transformers_version": "4.4.2",
      "unk_token_id": 1,
      "use_cache": true,
      "vocab_size": 32000
    }




```python
model = transformers.GPT2LMHeadModel(config)
```

モデルの準備ができたので、学習ループの実装にうつりましょう。

## 学習ループ

学習ループを実装するにあたり、まずは PyTorch でのモデルの学習の流れを一度まとめます。

モデルの学習にあたり大きくは次のステップが必要になります。

1. 損失の計算グラフの構築
1. 勾配の計算
1. 勾配に従ったパラメータのアップデート

ニューラルネットワークでは勾配に基づいてモデルのパラメータを更新していきます。
勾配を計算するには、現在のパラメータの値での微分を計算する必要があり、ニューラルネットワークでは損失の計算グラフを通して自動微分を行うことで求めます。
したがって、まずは損失の計算グラフを求める必要があるわけです。

計算グラフの構築と自動微分は、まさに PyTorch の **テンソル** の大きな目的であり、テンソルを使うことによって実現できます。
簡単な例で見て見ましょう。


```python
import torch

x = torch.tensor(10.0, requires_grad=True)
y = torch.tensor(5.0)
z = x + 2 * y
w = z ** 2
```


```python
x, y, z, w
```




    (tensor(10., requires_grad=True),
     tensor(5.),
     tensor(20., grad_fn=<AddBackward0>),
     tensor(400., grad_fn=<PowBackward0>))



`x` のように `requires_grad` を設定したテンソルは、そのテンソルが計算に使われた計算グラフに対して `.backward()` メソッドを呼ぶことで自動微分を実行した際に、微分の結果が `.grad` に保存されます。
（このように、 `requires_grad=True` と設定されたテンソルを今後 **パラメータ** とよぶことにします。）

自動微分をする前の `.grad` の値を見てみましょう


```python
x.grad, y.grad
```




    (None, None)



自動微分を行う前は、このように `.grad` には `None` が設定されています。
それではテンソルの `.backward()` メソッドを呼び出して自動微分を行ってみましょう。


```python
w.backward()
```

微分した結果がテンソルに入っているか `.grad` にアクセスして確かめてみます。


```python
x.grad, y.grad
```




    (tensor(40.), None)



確かに `requires_grad=True` を設定したテンソルにのみ微分の値が計算されて保存されているのがわかります。

さて、計算グラフの構築と自動微分による勾配の計算は PyTorch のテンソルを使うことで行うということがわかりました。
しかし、実際の PyTorch でのモデルは `nn.Module` というモジュールのサブクラスとして実装を行います。
それではモジュールとテンソルの関係はどうなっているでしょうか？

実はモジュールは、自身が最適化が必要なパラメータを内部で保持しています。
パラメータには、モジュールの `.parameters` アトリビュートを通してアクセスできます。


```python
one_param = next(model.parameters())
one_param
```




    Parameter containing:
    tensor([[ 0.0067,  0.0060, -0.0201,  ..., -0.0153, -0.0092,  0.0038],
            [-0.0231,  0.0080, -0.0129,  ..., -0.0116, -0.0267,  0.0094],
            [ 0.0095,  0.0595, -0.0047,  ..., -0.0033,  0.0188, -0.0065],
            ...,
            [-0.0005,  0.0166, -0.0038,  ..., -0.0315, -0.0261,  0.0179],
            [ 0.0221,  0.0020, -0.0198,  ..., -0.0117,  0.0096, -0.0135],
            [ 0.0208,  0.0151,  0.0328,  ...,  0.0188, -0.0229,  0.0227]],
           requires_grad=True)



結果を見ると分かる通り、テンソルに `requires_grad=True` が設定されていることがわかります。
このように、モジュールは一見すると何を行っているかわかりにくいかもしれませんが、モジュールの大きな役割の一つとしてパラメータを管理しているわけです。

ここまでで計算グラフの構築と勾配の計算の方法がわかりました。
最後に勾配に従ったパラメータのアップデートについて見ていきましょう。

先ほどの `x` を SGD を使って更新することを考えると、

```md
lr = 0.001
x = x - lr * x.grad
```

のように、パラメータをその勾配に従って更新を行えばよいことがわかります。

より一般に、パラメータをアップデートする機構（これを **オプティマイザ** といいます）には、更新対象のパラメータ、およびその勾配をわたすことでパラメータのアップデートが行えます。
これらの情報は、今までの説明から分かる通り PyTorch のテンソルがその役目を担っています。パラメータは `requires_grad=True` が設定されたテンソル、そしてその勾配は `.grad` からアクセスできるのでした。
したがって、オプティマイザには更新対象のテンソルを渡しておけばよいわけです

PyTorch では、オプティマイザは [`torch.optim` 以下で定義](https://pytorch.org/docs/stable/optim.html) されており、そのイニシャライザには今確認したように、更新対象となるテンソルを渡して初期化を行います。


```python
x_optim = torch.optim.SGD([x], lr=0.001)
```

生成したオプティマイザの `step()` メソッドを呼ぶことで、パラメータに設定された勾配に従って、パラメータが更新されます。

まず更新前の `x` とその勾配を確認しておきましょう。


```python
x, x.grad
```




    (tensor(10., requires_grad=True), tensor(40.))



`step()` メソッドでパラメータを更新してみます。


```python
x_optim.step()
```

予想では、 `10 - 40 * 0.001 = 9.96` となるはずです。実際に表示して確認してみます。


```python
x
```




    tensor(9.9600, requires_grad=True)



予想通りの値になっていますね。

ではモジュールに対してオプティマイザを作成してみましょう。モジュールのパラメータは `parameters()` で取得できるでした。
したがって、オプティマイザのイニシャライザにその値を直接渡せばよいことになります。
また、オプティマイザは SGD の代わりに Adam を使ってみることにします。


```python
optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
```

**Note:** オプティマイザは、イニシャライザに渡されたパラメータの更新のみを行います。したがって、更新を行いたくないパラメータがある場合には、オプティマイザのイニシャライザに渡さなければよいことになります。
モジュールの `.named_parameters()` メソッドでモジュール名つきのパラメータを取得できるため、そのモジュール名でフィルタをかける方法が有効です。

```py
torch.optim.Adam(
    [param for name, param in model.named_parameters()
     if name in ["module", "names", "to", "be", "updated"]
     ],
    lr=1.0e-4
)
```

必要な準備は整いましたので、ついに学習ループの実装に移ります。
重要な点はすでに説明済みなので、コメントとともに実装をしてみます。


```python
def train(
    model,
    optimizer,
    train_dataloader,
    valid_dataloader,
    n_epochs,
    loss_fn,
    device,
):
    for epoch in range(1, n_epochs+1):
        # [*1] 学習モード
        model.train()

        train_loss = 0

        for train_batch_idx, item in enumerate(train_dataloader, start=1):
            # ロスの計算グラフを構築する
            # forward 関数は、検証時にも利用するため別の関数で後で定義する
            loss = forward(model, item, loss_fn, device)
            # [*2] 勾配の初期化
            optimizer.zero_grad()
            # 勾配を計算し、その結果をテンソルの.gradに保存する
            loss.backward()
            # 勾配に従ってオプティマイザに登録したパラメータ (required_grad=Trueのテンソル) を更新
            optimizer.step()
            
            # エポックのロス計算は、勾配計算を行わないため計算グラフを構築する必要はない。
            # 計算グラフを構築しないために item を使ってテンソルの中身を取り出して計算している。
            # item を使わないと計算グラフをバッチのループ毎に作り続けそれを train_loss にキープし続けるため、
            # メモリを大量に消費してしまう
            train_loss += loss.item()
            
            # ログの出力
            if train_batch_idx % 100 == 0:
                batch_log = dict(epoch=epoch, batch=train_batch_idx, train_loss=train_loss/train_batch_idx)
                print(batch_log)

            
        # [*1] 検証モード
        model.eval()
        # [*3] 推論モードでは勾配計算しないので計算グラフを作成する必要がない。
        #      `torch.no_grad()` コンテキスト内のテンソルの計算では計算グラフは構築されない。
        with torch.no_grad():
            val_loss = 0
            for val_batch_idx, item in enumerate(valid_dataloader, start=1):
                loss = forward(model, item, loss_fn, device)
                val_loss += loss.item()
                
                # 次の行の assert で計算グラフが構築されていないことが確認できる。
                # assert loss.grad is None
                
        epoch_log = dict(
            epoch=epoch,
            train_loss=train_loss/train_batch_idx,
            valid_loss=val_loss/val_batch_idx,
        )
        print(epoch_log)
```


```python
def forward(model, item, loss_fn, device):
    """1バッチ毎のロスの計算を行う。
    
    item は DataLoader が返す辞書オブジェクトで `input_ids` と `labels` キーからなる。
    各々さずは (batch_size, input_len) となる。
    """
    # テンソルの to はインプレースではないので代入しないといけないということであっている？
    src, tgt = item["input_ids"], item["labels"]
    
    # [*4] テンソルを対象デバイスに移す。
    # テンソルの `to` はモジュールの `to` と異なりインプレースでデバイスに移らず、
    # 移動した先の新しいテンソルを返すので、必ず代入を行うこと
    src = src.to(device=device)
    tgt = tgt.to(device=device)

    # ロスを計算する
    output = model(input_ids=src)
    logits = output.logits  # shape: (batch_size, input_len, vocab_size)
    loss = loss_fn(
        input=logits.view(-1, logits.shape[-1]),
        target=tgt.view(-1)
    )
    return loss
```

いくつか [*N] の形式でコメントを付けた箇所について説明を加えます。

**[*1]** PyTorch のモジュールには学習モードと検証モードがあり、それぞれ `.train()`, `.eval()` メソッドで切り替えることができます。これらは、例えば Dropout のような学習時と検証時の挙動を変更する必要があるモジュールに対して、設定を変更することを意味しています。
基本的には、パラメータ更新を加える学習時には `.train()` で学習モードに設定し、学習後の評価時には `.eval()` で評価モードに設定します。

**[*2]** 勾配は初期化しないとずっと値が加算されつづけられます。これは、例えば一つのパラメータに対して二つの計算グラフが存在している状況で便利です。例えば、 `x` の例でもう一つ `v` という計算グラフが `x` から計算されているとします。


```python
v = 2 * x
```

`x` の勾配は、先ほど `w` から微分を計算したため現在はこうなっています。


```python
x.grad
```




    tensor(40.)



ここでもう一度 `.backward()` を呼び出して勾配を計算するとどうなるでしょうか。


```python
v.backward()
```


```python
x.grad
```




    tensor(42.)



以前の勾配 `40` に、新しい勾配 `2` が加算された結果になっていることがわかります。
このように複数のヘッドに対して勾配を計算する必要がある状況（つまり、複数の損失関数がある状況）では、加算機能は便利です。

一方で、バッチ毎に勾配を計算する場合はパラメータの勾配を初期化する必要があります。
それには、オプティマイザの `.zero_grad()` メソッドで実行できます。
`.zero_grad()` メソッドは、呼び出されるとそのオプティマイザに設定されているパラメータの勾配を `0` に初期化します。
バッチ毎の勾配計算時にはここでの学習ループの実装のように、
ロス計算グラフに対して `.backward()` を呼び出す直前に `.zero_grad()` を呼び出すと、
呼び出し忘れもなく、よいと考えられます。

**[*3]** `requires_grad=True` が設定されたテンソルから構築した計算グラフは自動微分が可能で便利なのですが、勾配計算が不要な場合にはオーバーヘッドとなります。
そこで、勾配計算が不要な際には `torch.no_grad()` コンテキスト内でテンソルの計算を行うことで、計算グラフが構築されなくなりオーバーヘッドが解消されます。

実際に `x` に対して行って見ましょう。
通常通り計算すると `grad_fn` がついていることから計算グラフが構築されることがわかります。


```python
x + x
```




    tensor(19.9200, grad_fn=<AddBackward0>)



一方で、`torch.no_grad()` コンテキスト内で起算すると `grad_fn` がついていないことから計算グラフが構築されていないことがわかります。


```python
with torch.no_grad():
    print(x + x)
```

    tensor(19.9200)


**[*4]** ニューラルネットワークの学習に GPU は欠かせません。GPU で学習する際には、PyTorch のテンソルも GPU に移動させる必要があります。
学習時には、すでに見た通り、モジュールに登録されているテンソル、および DataLoader がイテレーションし、モジュールでフォワードするテンソルの二つが大きくあり、各々を GPU のメモリ上で扱う必要があります。

テンソルおよびモジュールの `.to()` メソッドは、引数に指定したデバイスにテンソルを移動させます。
デバイスは文字列で指定でき、次のコードで GPU が利用できるときには GPU を、そうでない場合には CPU をデバイスとして設定します。


```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```


```python
device
```




    device(type='cuda')



今回は GPU が利用可能な環境なので `device` には `cuda` が設定されています。

あとはこの `device` を使ってモジュールおよびテンソルを GPU 上に移動すれば良いことになります。
まずはモジュールを GPU に移動させてみます。モジュールの `.to()` メソッドで引数に `device` を指定するだけです。
モジュールの `.to()` メソッドはインプレースなので、結果を再度変数に代入する必要はないことに注意してください。


```python
model.to(device=device)
```

次に DataLoader がイテレーションするテンソルを GPU に移動してみます。
モジュールの `.to()` メソッドと異なり、テンソルの `.to()` メソッドはインプレースで移動せずに、GPU のメモリ上にコピーした
新しいテンソルのインスタンスを返します。
従って [*4] のように結果を変数に代入する必要があることに注意してください。

```md
src = src.to(device=device)
tgt = tgt.to(device=device)
```

<hr />

さて、ここまででコメントの説明は終わりです。

最後にロス関数を定義して学習ループを実行してみましょう。


```python
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
```


```python
train(
    model=model,
    optimizer=optimizer,
    train_dataloader=train_loader,
    valid_dataloader=valid_loader,
    n_epochs=2,
    loss_fn=loss_fn,
    device=device,
)
```

これで学習ループの実装は完了です。

## 次のステップ

次のステップとして学習ループをよりよくするアイディアは、例えば次のようなものがあります。

* [Tensorboard による学習の可視化](https://pytorch.org/docs/stable/tensorboard.html)
* モデルの保存と復元（学習途中からの再開）
* [Learning rate のスケジュール](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
* [Automatic mixed precision](https://pytorch.org/docs/stable/notes/amp_examples.html)
* Early stopping

各々実装してももちろんよいのですが、実験毎に実装し直すのも大変です。
そこで、このようなオプションを提供する [PyTorch Lightning](https://www.pytorchlightning.ai/) という
素晴らしいパッケージがありますので、利用を検討してもよいでしょう。
