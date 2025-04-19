import sqlite3
from tqdm import tqdm
from dataloader import *
from extractor import *
import matplotlib.pyplot as plt
import ast
from community_detection import *
import matplotlib.pyplot as plt
from dataloader import *


pd.set_option('display.max_columns',100)
#pd.set_option('display.max_colwidth', None)

jsondir= '../../datasets/AcademicGraph_datasets/OAG_ver2-1/unzipped/'
dbfile = "papers.db"
rnd_state = 2024


papers = extract_references(jsondir, 21, 51) # end_fileは含まれない
print(papers.head)

# Save to SQLite
conn = sqlite3.connect(dbfile)
papers.to_sql("papers", conn, if_exists="append", index=None)
conn.close

# Read DB file
conn = sqlite3.connect(dbfile)
papers = pd.read_sql_query("SELECT * FROM papers", conn)
conn.close
print(papers)


# データ確認
year_counts = papers["year"].value_counts().sort_index()
year_counts = year_counts.reset_index()
year_counts.columns=["year","count"]
year_counts.to_csv("output.csv", index=False)

df = pd.read_csv('output.csv')
plt.figure(figsize=(10, 6))
plt.bar(df['year'], df['count'])
plt.xticks(range(df['year'].min(), df['year'].max() + 1, 10))
plt.title('Yearly Count')
plt.xlabel('Year')
plt.ylabel('Count')
plt.savefig('yearly_count_bar_plot.png')

# 多すぎるので2012年に限定
papers_2012 = papers[papers["year"]==2012]
conn = sqlite3.connect("papers_2012.db")
papers_2012.to_sql("papers", conn, if_exists="replace", index=None)
conn.close


# Read DB file
conn = sqlite3.connect("papers_2012.db")
papers_2012 = pd.read_sql_query("SELECT * FROM papers", conn)
conn.close
print(papers_2012.head)

# SQLで保存しているのでリスト再構築 & 2部グラフ構築
label = ["paper_id", "author_id", "reference", "fos", "title", "citation"]
i = []
for _, row in tqdm(papers_2012[["paper_id","title","citation","fos","authors_id","references"]].iterrows()):
    references = ast.literal_eval(row["references"])
    author_id = list(ast.literal_eval(row["authors_id"]))[0]

    # 条件付け n_citation>=50, references>=50(この閾値はextractor.pyより厳しく)
    if len(references)>=50 and int(row["citation"]>=50): 
        paper_id = row["paper_id"]
        title = row["title"]
        fos = row["fos"]
        citation = row["citation"]

        for reference in references:
            i.append([paper_id, author_id, reference, fos, title, citation])
inter = pd.DataFrame(i, columns=label)
print(inter.head)
conn = sqlite3.connect("interaction.db")
inter.to_sql("interaction", conn, if_exists="replace", index=None)


# referenceに紐付ける用のpaper_id:fos保管 (2012以外含かつn_citation>=20, references>=10(この閾値はextractor.pyに揃えてます))
conn = sqlite3.connect("papers.db")
papers = pd.read_sql_query("SELECT * FROM papers", conn)
conn.close
print(papers.head)
label = ["paper_id", "title", "author_id", "fos", "citation","refs_num"]
i = []
for _, row in tqdm(papers[["paper_id", "title", "citation", "fos", "authors_id", "references"]].iterrows()):
    if len(list(ast.literal_eval(row["authors_id"])))>0:
        author_id = list(ast.literal_eval(row["authors_id"]))[0]
        paper_id = row["paper_id"]
        title = row["title"]
        fos = row["fos"]
        citation = row["citation"]
        refs_num = len(list(ast.literal_eval(row["references"])))
        i.append([paper_id, title, author_id, fos, citation, refs_num])
ref_info = pd.DataFrame(i, columns = label)
print(ref_info.head)
conn = sqlite3.connect("ref_info.db")
ref_info.to_sql("ref_info", conn, if_exists="replace", index=None)


conn = sqlite3.connect("interaction.db")
interaction = pd.read_sql_query("SELECT * FROM interaction", conn)
conn.close
conn = sqlite3.connect("ref_info.db")
ref_info = pd.read_sql_query("SELECT * FROM ref_info", conn)
conn.close
# 保存した列名
# interaction: ["paper_id", "author_id", "reference", "fos", "title", "citation"]
# ref_info: ["paper_id", "title", "author_id", "fos", "citation","refs_num"]
ref_info = ref_info.rename(columns = {"paper_id":"ref_paper", "title":"ref_title","author_id":"ref_author","fos":"ref_fos","citation":"ref_citation","refs_num":"ref_refs"})
# ref_info: ["ref_paper", "ref_title", "ref_author", "ref_fos", "ref_citation","ref_refs"]

# referenceにfosを割当て
interaction = pd.merge(interaction, ref_info, left_on = "reference", right_on = "ref_paper")
interaction = interaction[["author_id","reference","ref_fos","paper_id","title","fos","citation","ref_paper","ref_title","ref_author","ref_citation","ref_refs"]]
print(interaction.head)
conn=sqlite3.connect("inter_fos.db")
interaction.to_sql("interaction", conn, if_exists="replace", index=None)

# 読み込み
conn = sqlite3.connect("inter_fos.db")
interaction = pd.read_sql_query("SELECT * FROM interaction", conn)
conn.close

# 条件付け
interaction = interaction.query("citation>=20 & ref_citation>=20 & ref_refs>=20")
# アイテム側アクティブ(min<=item<max)
counts = interaction["ref_author"].value_counts()
selected_mentions = counts[(counts>=5) & (counts<200)].index.tolist()
interaction = interaction[interaction['ref_author'].isin(selected_mentions)==True]
# ユーザ側アクティブ(n回以上登場)
counts = interaction["author_id"].value_counts()
selected_mentions = counts[counts>=5].index.tolist()
interaction = interaction[interaction['author_id'].isin(selected_mentions)==True]

auth_inter = interaction[["author_id", "ref_author"]].drop_duplicates().sort_values(by=["author_id","ref_author"], ascending=[True, True])
print(auth_inter.head)


# # Largest connected components
# # Community Detection
comm_df, inter_LCC = detect_community(auth_inter, rnd_state)
auth_inter = pd.merge(inter_LCC, comm_df, left_on="ref_author", right_on="user").drop(columns="user")
print(auth_inter.head)
print(f"author: ", auth_inter['author_id'].nunique())
print(f"ref_author: ", auth_inter['ref_author'].nunique())
print(f"community: ", auth_inter['community'].nunique())

conn=sqlite3.connect("inter_com.db")
auth_inter.to_sql("interaction", conn, if_exists="replace", index=None)


# long tail graph
auth_counts = auth_inter["ref_author"].value_counts()
auth_counts = auth_counts.reset_index()
auth_counts.columns = ['ref_author', 'counts']
auth_counts = auth_counts.sort_values(["counts"])
print(auth_counts.head)
auth_counts_freq = auth_counts["counts"].value_counts()
auth_counts_freq.sort_index().plot(kind='bar')
plt.xlabel('value_counts')    # 横軸のラベル
plt.ylabel('Frequency')       # 縦軸のラベル
plt.title('Frequency of value_counts')  # グラフのタイトル
plt.savefig("counts.png", format="png", dpi=300)



conn = sqlite3.connect("inter_com.db")
auth_inter = pd.read_sql("SELECT * FROM interaction", conn)
conn.close()
print(auth_inter)

# Make consecutive data
mapped_retweets = consecutive_data(auth_inter)

# Save to DB file "retweets_Consecutive"
conn = sqlite3.connect(dbfile)
mapped_retweets.to_sql("auth_inter_Consecutive", conn, if_exists="replace", index=None)
conn.close

# Read DBfile
conn = sqlite3.connect(dbfile)
mapped_retweets = pd.read_sql_query("SELECT * FROM auth_inter_Consecutive", conn)
conn.close


datasets_dir = "./dataset/"
train = 0.7
val = 0.15
test = 0.15


# Make dataset
DGRec_LGCN_dataset(mapped_retweets, datasets_dir, train, val, test, rnd_state)
