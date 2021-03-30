# Hugging Face Transformers ではじめる PyTorch での言語モデル学習

自分の環境の CUDA のバージョンを確認して、公式ドキュメント に従ってコマンドを実行する。対応した PyTorch をインストールします。
今回は CUDA 11.1 対応の Pytorch をインストールします。


```python
!pip3 install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

    Looking in links: https://download.pytorch.org/whl/torch_stable.html
    Collecting torch==1.8.1+cu111
      Downloading https://download.pytorch.org/whl/cu111/torch-1.8.1%2Bcu111-cp38-cp38-linux_x86_64.whl (1982.2 MB)
    [K     |████████████████████████████████| 1982.2 MB 11 kB/s  eta 0:00:017
    [?25hCollecting numpy
      Downloading numpy-1.20.2-cp38-cp38-manylinux2010_x86_64.whl (15.4 MB)
    [K     |████████████████████████████████| 15.4 MB 8.0 MB/s eta 0:00:01
    [?25hCollecting typing-extensions
      Downloading typing_extensions-3.7.4.3-py3-none-any.whl (22 kB)
    Installing collected packages: numpy, typing-extensions, torch
    Successfully installed numpy-1.20.2 torch-1.8.1+cu111 typing-extensions-3.7.4.3


Hugging Face の Transformers と、SentencePiece トークナイザを利用するためにそのパッケージをインストールします。


```python
!pip3 install transformers==4.4.2 sentencepiece==0.1.95
```

    Collecting transformers==4.4.2
      Downloading transformers-4.4.2-py3-none-any.whl (2.0 MB)
    [K     |████████████████████████████████| 2.0 MB 9.0 MB/s eta 0:00:01
    [?25hCollecting sentencepiece==0.1.95
      Downloading sentencepiece-0.1.95-cp38-cp38-manylinux2014_x86_64.whl (1.2 MB)
    [K     |████████████████████████████████| 1.2 MB 8.4 MB/s eta 0:00:01
    [?25hCollecting requests
      Downloading requests-2.25.1-py2.py3-none-any.whl (61 kB)
    [K     |████████████████████████████████| 61 kB 6.1 MB/s eta 0:00:01
    [?25hCollecting regex!=2019.12.17
      Downloading regex-2021.3.17-cp38-cp38-manylinux2014_x86_64.whl (737 kB)
    [K     |████████████████████████████████| 737 kB 8.4 MB/s eta 0:00:01
    [?25hCollecting tokenizers<0.11,>=0.10.1
      Downloading tokenizers-0.10.1-cp38-cp38-manylinux2010_x86_64.whl (3.2 MB)
    [K     |████████████████████████████████| 3.2 MB 8.5 MB/s eta 0:00:01
    [?25hRequirement already satisfied: packaging in /usr/local/lib/python3.8/dist-packages (from transformers==4.4.2) (20.9)
    Collecting filelock
      Downloading filelock-3.0.12-py3-none-any.whl (7.6 kB)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.8/dist-packages (from transformers==4.4.2) (1.20.2)
    Collecting sacremoses
      Downloading sacremoses-0.0.43.tar.gz (883 kB)
    [K     |████████████████████████████████| 883 kB 8.5 MB/s eta 0:00:01
    [?25hCollecting tqdm>=4.27
      Downloading tqdm-4.59.0-py2.py3-none-any.whl (74 kB)
    [K     |████████████████████████████████| 74 kB 3.1 MB/s eta 0:00:01
    [?25hCollecting urllib3<1.27,>=1.21.1
      Downloading urllib3-1.26.4-py2.py3-none-any.whl (153 kB)
    [K     |████████████████████████████████| 153 kB 8.5 MB/s eta 0:00:01
    [?25hCollecting idna<3,>=2.5
      Downloading idna-2.10-py2.py3-none-any.whl (58 kB)
    [K     |████████████████████████████████| 58 kB 7.5 MB/s eta 0:00:01
    [?25hCollecting certifi>=2017.4.17
      Downloading certifi-2020.12.5-py2.py3-none-any.whl (147 kB)
    [K     |████████████████████████████████| 147 kB 8.4 MB/s eta 0:00:01
    [?25hCollecting chardet<5,>=3.0.2
      Downloading chardet-4.0.0-py2.py3-none-any.whl (178 kB)
    [K     |████████████████████████████████| 178 kB 8.4 MB/s eta 0:00:01
    [?25hRequirement already satisfied: pyparsing>=2.0.2 in /usr/local/lib/python3.8/dist-packages (from packaging->transformers==4.4.2) (2.4.7)
    Collecting click
      Downloading click-7.1.2-py2.py3-none-any.whl (82 kB)
    [K     |████████████████████████████████| 82 kB 2.9 MB/s eta 0:00:01
    [?25hCollecting joblib
      Downloading joblib-1.0.1-py3-none-any.whl (303 kB)
    [K     |████████████████████████████████| 303 kB 8.5 MB/s eta 0:00:01
    [?25hRequirement already satisfied: six in /usr/local/lib/python3.8/dist-packages (from sacremoses->transformers==4.4.2) (1.15.0)
    Building wheels for collected packages: sacremoses
      Building wheel for sacremoses (setup.py) ... [?25ldone
    [?25h  Created wheel for sacremoses: filename=sacremoses-0.0.43-py3-none-any.whl size=893260 sha256=47968d3787e8e16031f78c4fd076343175a44480b363d9e3cab4bc3373675855
      Stored in directory: /root/.cache/pip/wheels/7b/78/f4/27d43a65043e1b75dbddaa421b573eddc67e712be4b1c80677
    Successfully built sacremoses
    Installing collected packages: urllib3, idna, certifi, chardet, requests, regex, tokenizers, filelock, click, joblib, tqdm, sacremoses, transformers, sentencepiece
    Successfully installed certifi-2020.12.5 chardet-4.0.0 click-7.1.2 filelock-3.0.12 idna-2.10 joblib-1.0.1 regex-2021.3.17 requests-2.25.1 sacremoses-0.0.43 sentencepiece-0.1.95 tokenizers-0.10.1 tqdm-4.59.0 transformers-4.4.2 urllib3-1.26.4



```python

```
