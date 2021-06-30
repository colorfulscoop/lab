---
created_on: 2021/04/03
updated_on: 2021/04/05
---

# PyTorch 1.9.0 で BufferedShuffleDataset を使う

## BufferedShuffleDataset とは

機械学習、とりわけ深層学習でで大きなデータを扱うときに、はじめにメモリにすべてロードすることができない場合は少なくありません。

そんな大きなデータセットを扱う上で便利なのが PyTorch の [iterable-style dataset](https://pytorch.org/docs/stable/data.html#dataset-types) です。
Iterable-style dataset を使うことで、サンプルをはじめに全てロードすることなく、学習に必要になったときにサンプルを準備して返すことができます。
                                                          
その際に問題になるのが学習データのシャッフルです。
Map-style dataset では、[DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) の `shuffle` パラメータを `True` に設定することで学習データをシャッフルします。
一方で iterable-style dataset でははじめに全データをロードするわけではなく、この方法が使えません。実際、iterable-style dataset で `shuffle` パラメータを `True` に設定すると例外が発生します。

この問題に対処するために、PyTorch 1.8.1 では [BufferedShuffleDataset](https://pytorch.org/docs/1.8.1/data.html#torch.utils.data.BufferedShuffleDataset) が提供されていました。

```py
torch.utils.data.BufferedShuffleDataset(dataset, buffer_size)
```

BufferedShuffleDatasetは `buffer_size` で指定したサイズのバッファを内部で作成し、サンプルはまずバッファに格納されます。
そして、バッファが満たされたらそのうちの一つをランダムサンプルして返します。そうするとバッファに一つ空きができますので、次のサンプルをバッファに格納します。
これを続けることで、バッファサイズ分のシャッフルを行いながらサンプルを返していくのです。


## PyTorch を 1.9.0 に更新してみたら

今回 PyTorch を 1.8.1 から 1.9.0 に上げてみたところ、

```py
>>> import torch
>>> torch.utils.data.BufferedShuffleDataset
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: module 'torch.utils.data' has no attribute 'BufferedShuffleDataset'
```

のようなエラーが出てしまいました。

ぶーちゃんもショックを受けているようです。

> ぶーちゃん：ぶ〜〜〜〜。。。。（うーなんでエラーが出ちゃうんだろう、しょんぼり。。。）

<img src="20210628-01.png">

## 原因調査

PyTorchのコミットを調べてみると次のようなものを見つけました。

https://github.com/pytorch/pytorch/commit/89b1053413dab77c9a6c67da5a54ab9bbad1fbdd#diff-425b66e1ff01d191679c386258a7156dfb5aacd64a8e0947b24fbdebcbee8529

コミットを読んでいくと、どうやら`torch.utils.data` 以下から DataPipe という機能群に移されたようです。

実際にコードを確認すると `torch.utils.data.datapipes.iter.combinatorics` で `ShuffleIterDataPipe` というクラスで定義され、torch.utils.data.datapipes.iter の名前空間で Shuffle という名前でimport されていることがわかります。

* https://github.com/pytorch/pytorch/blob/v1.9.0/torch/utils/data/datapipes/iter/combining.py#L43
* https://github.com/pytorch/pytorch/blob/v1.9.0/torch/utils/data/datapipes/iter/__init__.py

ということは、次のようにすればうまく動くのでは...




> ぶーちゃん：ターンッ！（うごけ！）

<img src="20210628-02.png">

```py
>>> torch.utils.data.datapipes.iter.Shuffle
<class 'torch.utils.data.datapipes.iter.combinatorics.ShuffleIterDataPipe'>
```

インポートできているようです。
実際に引数にジェネレータを渡して動くか確かめてみましょう。

```py
>>> shuffle_dataset = torch.utils.data.datapipes.iter.Shuffle(range(10), buffer_size=3)
>>> list(shuffle_dataset)
[2, 0, 3, 5, 1, 6, 7, 8, 4, 9]
```

期待通り動いているようですね！

> ぶーちゃん：ぶおぉぉぉおおおっ！！！！（うまくいった！喜んだときの得意技、耳倒立）


<img src="20210628-03.png">

## まとめ

今回は PyTorch 1.8.1 で提供されていた ShuffledBufferDataset が PyTorch 1.9.0 でインポートできてなくなっている原因を調べました。
コミットを調査すると `torch.utils.data.datapipes.iter.Shuffle` に移動したようです。

[PyTorch 1.9.0 のリリースノート](https://github.com/pytorch/pytorch/releases/tag/v1.9.0) にも書かれていなかったのであまり使われていない機能なのでしょうか。

もともと TensorFlow を使っていて tf.data.Dataset の [shuffle](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle)
でバッファ付きのシャッフルという機能を知り、その後 PyTorch 1.8.1 で PyTorch に移行してきてから同等の機能を求めてドキュメントを読んでいたら見つけたクラスでした。
PyTorch 1.9.0 のリリースで削除されてしまったかと思いましたが、少なくとも今のところは `torch.utils.data.datapipes.iter.Shuffle` を利用すればよさそうです。

