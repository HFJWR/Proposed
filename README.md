# Proposed
提案手法のアーカイブ

## 提案手法概要

あらまし
<details>
<summary>
ソーシャルメディアにおいて，ユーザ推薦システムは新しいコンテンツとの...
</summary>

ソーシャルメディアにおいて，ユーザ推薦システムは新しいコンテンツとの出会いや交友関係の拡大に重要な役割を果たす．しかし，推薦精度の向上を重視するあまり，類似したユーザのみを提示することでエコーチェンバーやフィルターバブルを形成するという問題が指摘されている．本研究では，この問題に対し，ユーザの相互作用とコミュニティ情報を活用した2段階の多様化手法を提案する．第1段階ではデータセット全体の多様性を向上させ，第2段階では個人の推薦リストをリランキングすることで，更なる多様化を実現する．代表的なソーシャルメディアであるTwitter (現X)のデータを用いた実験により，提案手法が従来手法と同等の推薦精度を保ちながら，多様性指標において最大2.5倍の向上を達成することを確認した．

</details>
概要図

![Image](https://github.com/user-attachments/assets/b68d47bf-5297-4bf5-ab88-922b78bfe61c)


## Directory Tree
Programs以下の大まかなディレクトリ構造
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

- Dataloader
    BLM，OAGのデータセット(二部グラフ)を作成
    - BLMconstructor: BLMdatasets内の元データからコミュニティ情報付きの二部グラフを作成
    - OAGconstructor: OAGdataset内の元データからコミュニティ情報付きの二部グラフを作成
    ~~~
    ── BLMconstrutor
        │   ├── conf（収集期間やtrain,test,valの変更）
        │   ├── output（作成したデータセット，作成時の中間ファイル（データベース））
        │   └── result（Homogeniety Levelや推薦結果のテキストファイル）
    ~~~


- RecSystems
    提案手法，比較手法をまとめています
    - Proposed
        実行方法
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