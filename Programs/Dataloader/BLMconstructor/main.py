import configparser
import pandas as pd
import sqlite3
from datetime import datetime
from community_detection import *
from dataloader import *
from tweet_extractor import *
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List

def pairwise_jaccard_similarity_real_vectors(X):
    """
    Compute the pairwise Jaccard similarity between real-valued feature vectors.

    For real-valued vectors, the Jaccard similarity can be defined as:
    J(x, y) = sum_i min(x_i, y_i) / sum_i max(x_i, y_i)

    Parameters:
    X (numpy.ndarray): A 2D numpy array of shape (n_samples, n_features),
                       where each row represents a sample with real-valued features.

    Returns:
    numpy.ndarray: A 2D numpy array of size n_samples x n_samples,
                   where element (i, j) is the Jaccard similarity between X[i] and X[j].
    """
    n_samples = X.shape[0]
    
    # Calculate the pairwise minimum and maximum between all vectors
    min_matrix = np.minimum(X[:, np.newaxis, :], X[np.newaxis, :, :]).sum(axis=2)
    max_matrix = np.maximum(X[:, np.newaxis, :], X[np.newaxis, :, :]).sum(axis=2)

    # Avoid division by zero by adding a small epsilon where max_matrix is zero
    max_matrix[max_matrix == 0] = 1e-9
    
    # Calculate Jaccard similarity matrix
    sim_matrix = min_matrix / max_matrix
    
    return sim_matrix

def calculate_homogeneity_level(item_features: List[List[int]]) -> float:
    """
    Calculate the Homogeneity Level (HL).
    
    :param item_features: A list of binary feature vectors for each item.
    :return: The Homogeneity Level, a float between 0 and 1.
    """
    # Convert the list of feature vectors to a numpy array
    features = np.array(item_features)
    
    # Get the number of items
    n_items = len(item_features)
    
    if n_items <= 1:
        return 0.0  # No homogeneity can be calculated for 0 or 1 item

    # Normalize the feature vectors
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized_features = features / np.maximum(norms, 1e-8)  # Avoid division by zero

    jaccard_sim = pairwise_jaccard_similarity_real_vectors(normalized_features)
    
    # Remove self-similarities from the diagonal
    np.fill_diagonal(jaccard_sim, 0)

    # Calculate HL
    hl = (2 / (n_items * (n_items - 1))) * np.triu(jaccard_sim).sum()

    return hl


pd.set_option('display.max_columns',100)
# pd.set_option('display.max_colwidth', None)

# Read configuration(./conf内に収集期間などの設定を保持しています)
config = configparser.ConfigParser()
config.read('conf/blmdata.ini', encoding='utf-8')
# Get settings
start_date = config['DEFAULT']['start_date']
end_date = config['DEFAULT']['end_date']
jsonfile = config['DEFAULT']['jsonfile']
db_dir = config['DEFAULT']['db_dir']
datasets_dir = config['DEFAULT']['datasets_dir']
train = config['DEFAULT']['train']
val = config['DEFAULT']['val']
test = config['DEFAULT']['test']
rnd_state = int(config['DEFAULT']['rnd_state'])
# DB file
dbfile = db_dir+start_date+end_date+'.db'

# Extract retweets
tweets, retweets = extract_retweets(jsonfile, start_date, end_date)
print(retweets.head)

# # Save to DB file "tweets"
# conn = sqlite3.connect(dbfile)
# tweets.to_sql("tweets", conn, if_exists="replace", index=None)
# conn.close
# # Save to DB file "retweets"
# conn = sqlite3.connect(dbfile)
# retweets.to_sql("retweets", conn, if_exists="replace", index=None)
# conn.close
# # Read DB file
# conn = sqlite3.connect(dbfile)
# retweets = pd.read_sql_query("SELECT * FROM retweets", conn)
# conn.close
# print(retweets.head())

# Drop duplications
retweets['text'] = retweets.groupby(['author_id', 'mention_id'])['text'].transform(lambda x: ' '.join(x))
retweets = retweets.drop_duplicates(subset=['author_id', 'mention_id'])

# Hold items 5 <= mention_id < 100
minimum = 5 
maximum = 100
counts = retweets['mention_id'].value_counts()
selected_mentions = counts[(counts>=minimum) & (counts<maximum)].index.tolist()
retweets = retweets[retweets['mention_id'].isin(selected_mentions)==True]

# Hold active users
duplicated_rows = retweets[retweets.duplicated(subset='mention_id', keep=False)]
retweets = duplicated_rows.groupby('author_id').filter(lambda x: len(x)>=5)
print("total: "+str(len(retweets)))


# Longtailグラフ
# count_df = retweets["mention_id"].value_counts().reset_index()
# count_df.columns=["mention_id","count"]
# count_freq = count_df["count"].value_counts().reset_index()
# count_freq.columns = ["count", "frequency"]
# count_freq = count_freq.sort_values(by="frequency", ascending=False)

# plt.figure(figsize=(6, 4))
# plt.bar(count_freq.index, count_freq["frequency"])
# plt.xlabel("Interacted User Index")
# plt.xlim(-5,100)
# plt.ylim(0,16000)
# plt.ylabel("Number of Occurance")
# plt.title("Long tail of BLM")
# plt.xticks(np.arange(0,101,30))
# plt.yticks(np.arange(0,15001,3000))
# plt.grid(True, which='both', axis='both', linestyle='-', linewidth=0.7)
# plt.savefig("./results/longtail.png")

## Read DB file
# conn = sqlite3.connect(dbfile)
# tweets = pd.read_sql_query("SELECT * FROM tweets", conn)
# conn.close


# authorごとにツイートをまとめる
grouped_tweets = tweets.groupby("author_id").agg({
    "tweet_id": list,
    "tags": list,
    "text": list
}).reset_index().rename(columns={
    "author_id":"author_id",
    "tweet_id":"authors_tweet_ids",
    "tags":"authors_tags",
    "text": "authors_texts"
})
# retweetsのuser1-user2のうちuser1を軸にマージ（推薦を行うユーザの過去の発言を取るため）
retweets = pd.merge(retweets, grouped_tweets, on="author_id")
# 過去の発言回数によって絞る（>=5）
retweets = retweets[retweets["authors_tweet_ids"].apply(len)>=5]
retweets = retweets.astype({"authors_tweet_ids":str, "authors_tags":str, "authors_texts":str})
print(retweets.head)


# データベースに分割保管
# リツイート関係，ツイートID，ツイート内容
retweets_save = retweets[["author_id", "mention_id", "created", "tweet_id", "text"]]
conn = sqlite3.connect(dbfile)
retweets_save.to_sql("retweets", conn, if_exists="replace", index=None)
conn.close()
# ユーザごとのツイートID，ツイート内容
authors_tweets = retweets[["author_id", "authors_tweet_ids", "authors_texts"]]
conn = sqlite3.connect(dbfile)
retweets.to_sql("authors_tweets", conn, if_exists="replace", index=None)
conn.close()

# DB読み込み
conn = sqlite3.connect(dbfile)
retweets = pd.read_sql_query("SELECT * FROM retweets", conn)
conn.close()

# Community detection
retweets = retweets[["author_id", "mention_id", "created", "tweet_id", "text"]]
comm_df, retweets_LCC = detect_community(retweets, rnd_state)

# Merge community data
retweets = pd.merge(retweets_LCC, retweets, on=["author_id","mention_id"])
# community追加
retweets = pd.merge(retweets, comm_df, left_on="mention_id", right_on="user").drop(columns='user')
print(retweets.sort_values("community", ascending=False).head())

# # Save to DB file "retweets_COM"
# conn = sqlite3.connect(dbfile)
# retweets.to_sql("retweets_COM", conn, if_exists="replace", index=None)
# conn.close
# # Read DB file
# conn = sqlite3.connect(dbfile)
# cur = conn.cursor()
# retweets = pd.read_sql_query("SELECT * FROM retweets_COM", conn)
# conn.close()


# Make consecutive data
mapped_retweets = consecutive_data(retweets)

# # Save to DB file "retweets_Consecutive"
# conn = sqlite3.connect(dbfile)
# mapped_retweets.to_sql("retweets_Consecutive", conn, if_exists="replace", index=None)
# conn.close
# # Read DBfile
# conn = sqlite3.connect(dbfile)
# mapped_retweets = pd.read_sql_query("SELECT * FROM retweets_Consecutive", conn)
# conn.close

# Make dataset
DGRec_LGCN_dataset(mapped_retweets, datasets_dir, train, val, test, rnd_state)


# 以下202412月 Homogeneity Level計測
# ===================================================================================================
# Homogeneity Level用BERTopic事前計算
# BLM呼び出し
conn = sqlite3.connect(db_dir+"Retweets.db")
retweets = pd.read_sql_query("SELECT * FROM retweets", conn)
conn.close

conn = sqlite3.connect(db_dir+"Text.db")
text = pd.read_sql_query("SELECT * FROM retweets", conn)
conn.close

retweets = pd.merge(retweets, text, on=["tweet_id"])
print(retweets)

# Bertopicによるトピック割当（割当済みデータは./output/database/Retweets.dbに）

# from bertopic import BERTopic
# # 同一user_idなら文章をuser_idごとにまとめる
# usertext = retweets[["user","text"]]
# usertext = usertext.groupby('user')['text'].agg(lambda x: ' '.join(x)).reset_index()
# usertext["user_id_renban"] = range(0,len(usertext))
# sentences = []
# for i in usertext["text"]:
#     sentences.append(i)
# print(f"num of text: ", str(len(sentences)))
# # BERTopicモデル
# os.environ["TOKENIZERS_PARALLELISM"]="false"
# model = BERTopic(embedding_model="all-MiniLM-L6-v2", calculate_probabilities=False, low_memory=True)
# topics, probs = model.fit_transform(sentences)
# # トピック削減
# model.reduce_topics(sentences, nr_topics=70)
# # 外れ値の移動
# new_topics = model.reduce_outliers(sentences, topics)

# res = model.get_document_info(sentences)
# res = res.sort_values('Topic')
# res["user_id_renban"] = range(0,len(res))
# usertext = pd.merge(usertext, res, on="user_id_renban")

# usertext = usertext[["user","Topic"]]
# usertext = usertext.sort_values("Topic")
# print(usertext.head)

# retweets = pd.merge(retweets, usertext, on="user")
# retweets = retweets[["user","inter","community","Topic","created","tweet_id","author_id","mention_id","text"]]
# print(retweets.head)

# # 分割保存
# text = retweets[["tweet_id","text"]]
# conn = sqlite3.connect(db_dir+"Text.db")
# text.to_sql("retweets", conn, if_exists="replace",index=None)
# conn.close
# retweets = retweets[["user","inter","community","Topic","created","tweet_id","author_id","mention_id"]]
# conn = sqlite3.connect(db_dir+"Retweets.db")
# retweets.to_sql("retweets", conn, if_exists="replace", index=None)
# conn.close


# BLM呼び出し
conn = sqlite3.connect(db_dir+"Retweets.db")
retweets = pd.read_sql_query("SELECT * FROM retweets", conn)
conn.close
retweets = retweets[["inter","community","Topic"]]
print(retweets)

# LightGCNの結果呼び出し
hl_com_list=[]
hl_topic_list=[]
with open("./results/results_lgcn.txt", "r") as f:
    for line in tqdm(f):
        data = json.loads(line)
        for user, inter_list in data.items():
            com=[]
            topic=[]
            # community, topicの推薦リスト
            for inter in inter_list:
                matching_row = retweets[retweets["inter"] == inter]
                if not matching_row.empty:
                    com.append(matching_row.iloc[0]["community"])
                    topic.append(matching_row.iloc[0]["Topic"])
            # communityの01行列
            com_features = np.zeros((len(com),len(com)),dtype=int)
            for i in range(len(com)):
                for j in range(len(com)):
                    if com[i] == com[j]:
                        com_features[i,j]=1
            # Topicの01行列
            topic_features = np.zeros((len(topic),len(topic)),dtype=int)
            for i in range(len(topic)):
                for j in range(len(topic)):
                    if topic[i] == topic[j]:
                        topic_features[i,j]=1          
            
            # ユーザごとのHL計算
            hl_com = calculate_homogeneity_level(com_features)
            hl_com_list.append(hl_com)
            hl_topic = calculate_homogeneity_level(topic_features)
            hl_topic_list.append(hl_topic)

# 平均、標準偏差
print(f"ComHL mean: {np.mean(hl_com_list)},ComHL std: {np.std(hl_com_list)} ")
print(f"TopicHL mean: {np.mean(hl_topic_list)},TopicHL std: {np.std(hl_topic_list)} ")

# import pickle
# f = open("./results/hl_com.txt","wb")
# pickle.dump(hl_com_list,f)
# f = open("./results/hl_topic.txt","wb")
# pickle.dump(hl_topic_list,f)
# with open("./results/hl_com.txt","rb") as f:
#     hl_com_list = pickle.load(f)
# with open("./results/hl_topic.txt","rb") as f:
#     hl_topic_list = pickle.load(f)

import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.hist([hl_com_list,hl_topic_list],bins=50,alpha=1.0,label=["HL community","HL topic"])

plt.title("Comparison of Community and Topic", fontsize=12)
plt.xlabel("Homogeneity Level")
plt.ylabel("Frequency", fontsize=12)
plt.legend()
plt.savefig("./results/HL.png")