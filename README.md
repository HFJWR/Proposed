# Proposed
提案手法アーカイブ

# Directory Tree
Programs以下のディレクトリ構造
~~~
.
└── Programs
    ├── Dataloader
    │   ├── BLMconstructor（BLM元データからコミュニティを割り当てた二部グラフ作成）
    │   └── OAGconstructor（OAG元データからコミュニティを割り当てた二部グラフ作成）
    ├── RecSystems
    │   ├── DGRec（変更なしDGRec）
    │   ├── LightGCN（比較手法1: LightGCN）
    │   ├── Proposed（提案手法）
    │   └── SELFRec（比較手法2: XSimGCL）
    └── datasets（データセットの元データ）
~~~
・作成済みデータセットは推薦手法ごとに用意して各ディレクトリに配置済みです(BLMのみ)
・データセット作成時の元データ詳細は以下ディレクトリのmemo.txtに記載しています
~~~
Programs/datasets/memo.txt
~~~

# Dataloader
BLM，OAGのデータセットを作成します

## BLMconstructor
### ディレクトリ構造
~~~
── BLMconstrutor
    │   ├── conf（収集期間やtrain,test,valの変更）
    │   ├── output（作成したデータセット，作成時の中間ファイル（データベース））
    │   └── result（Homogeniety Levelや推薦結果のテキストファイル）
~~~
### Running
~~~
python main.py
~~~
・実行することでdatasets内の元データからコミュニティ情報付きの二部グラフを作成します
    - 出力先: ./Dataloader/BLMconstructor/datasets/
    - データセット名: 実行時の日付、時間
・手法ごとにデータセットの形式が異なるためフォルダ分けしています
・item_categoryは提案手法と共通しているのでコピペで追加してください

## OAGconstructor
基本的にはBLMconstructorと同様です．
confファイルの廃止，2012年のデータのみに絞る，等一部変更があります．


# RecSystems
提案手法，比較手法をまとめています
## Proposed
~~~
python main.py --lr 0.05 --weight_decay 1e-6 --beta_class 0.90 --epoch 300 --patience 10 --dataset BLM
~~~
・Step2のリランキングは(./utils/tester.py)の256~296行目のコメントアウトを解除して実行
・リランキングなしの場合は全部コメントアウト

・各ユーザの推薦結果は(./output/)に保存（実行日-時間でフォルダ分け）
・フォルダ内の(diversity_maxmin.txt)はcoverage,entropyそれぞれ最大・最小の値とその値をとったユーザを保存しており、詳細は以下のようになっています
    1行目
    - cov_max: coverage最大値
    - cov_min: coverage最小値
    - max_categories: coverage最大ユーザのtop-k
    - max_user: coverage最大をとったユーザID
    - min_categories: coverage最小ユーザのtop-k
    - min_user: coverage最小をとったユーザID
    2行目
    (1行目のentropy版)
    
### Environment  
DGL version 1.0.1  
Pytorch version 1.12.1  

### Dataset
Format of train.txt val.txt test.txt: UserID,ItemID  
Format of item_category.txt: ItemID,CategoryID  

### Base Method: DGRec  
>A PyTorch and DGL implementation for the WSDM 2023 paper below:  
>[DGRec: Graph Neural Network for Recommendation with Diversified Embedding Generation](https://arxiv.org/pdf/2211.10486.pdf) 

## LightGCN
### Running
~~~
python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="BLM" --topks="[100]" --recdim=64 --epoch 301
~~~

## XSimGCL
~~~
python main.py
~~~

パラメータ、データセット等は(./conf/XSimGCL.conf)で決定


# 実行環境作成の流れ
## イメージ作成
<https://ngc.nvidia.com/signin>へアクセスしログイン
右上名前->Setup->Get API Key->Generate API Key->Confirmを確認
~~~
$ docker login nvcr.io
~~~
でアカウント認証
- Username:$oauthtoken
- Password:サイトでコピペ
↑（詳細は<https://github.com/meruemon/Ubuntu-Setup/blob/main/docker_instructions.md>の動作確認の章にあります）
~~~
$ docker build -t rec_env:cuda11.3.1-py3 ./
~~~
## コンテナ立ち上げ
    $ docker-compose up -d
    $ docker exec -it fujiwara_env bash
成功していればProgramsディレクトリに移動しています