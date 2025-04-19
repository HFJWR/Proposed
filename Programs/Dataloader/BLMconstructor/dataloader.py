import os
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

def create_mapping(column, start):
    unique_values = column.unique()
    mapping = {value: idx for idx, value in enumerate(unique_values, start=start)}
    return column.map(mapping)

def consecutive_data(retweets):
    """
    retweets: df['created']['tweet_id']['author_id']['mention_id']['text']['community']
    """
    retweets["author_id"] = create_mapping(retweets["author_id"],0)
    retweets["mention_id"] = create_mapping(retweets["mention_id"],0)
    retweets["community"] = create_mapping(retweets["community"],0)

    return retweets

def Full_dataset(retweets, datasets_dir, train, val, test, rnd_state):
    """
    retweets: df['created']['tweet_id']['author_id']['mention_id']['text']['community']
    train = 0.6
    val = 0.2
    test = 0.2
    """
    
    current_time = datetime.now().strftime("%m%d-%H%M")
    directory = datasets_dir+f"{current_time}/"
    os.makedirs(directory, exist_ok=True)

    # iniの内容保存
    source_file = "./conf/blmdata.ini"
    target_file = directory+"blmdata.txt"
    with open(source_file, "r") as src_file:
        content = src_file.read()
    with open(target_file, "w") as dst_file:
        dst_file.write(content)

    # interaction, item_category保存
    retweets = retweets.sort_values(["author_id", "mention_id"])
    # 重複削除
    retweets_inter = retweets[["author_id","mention_id"]].drop_duplicates()
    print(retweets_inter.nunique())
    retweets_inter[['author_id', 'mention_id']].to_csv(directory+'full.csv', index=False, sep=',')

    # item_category
    item_category = retweets[['mention_id', 'community']].drop_duplicates()
    item_category = item_category.sort_values(["mention_id", "community"])
    print(item_category.nunique())
    item_category.to_csv(directory+'item_category.txt', index=False, header=False, sep=',')

    train_data, temp_data = train_test_split(retweets_inter[['author_id', 'mention_id']], test_size = (1-float(train)), random_state=rnd_state)
    val_data, test_data = train_test_split(temp_data, test_size = float(test) / (float(val)+float(test)), random_state=rnd_state)

    return(directory, train_data.sort_values(["author_id", "mention_id"]), val_data.sort_values(["author_id", "mention_id"]), test_data.sort_values(["author_id", "mention_id"]))


def DGRec_LGCN_dataset(retweets, datasets_dir, train_data, val_data, test_data, rnd_state):

    directory, train_data, val_data, test_data = Full_dataset(retweets, datasets_dir, train_data, val_data, test_data, rnd_state)
    
    dgrec_dir = directory+'/DGRec/'
    lgcn_dir = directory+'/LightGCN/'
    xsim_dir = directory+'/XSimGCL/'
    
    os.makedirs(dgrec_dir, exist_ok=True)
    os.makedirs(lgcn_dir, exist_ok=True)
    os.makedirs(xsim_dir, exist_ok=True)
    
    # DGRec
    train_data.to_csv(dgrec_dir+'train.txt', index=False, header=False, sep=',')
    val_data.to_csv(dgrec_dir+'val.txt', index=False, header=False, sep=',')
    test_data.to_csv(dgrec_dir+'test.txt', index=False, header=False, sep=',')
    

    # XSimGCL
    train_xsim = train_data.copy()
    val_xsim = val_data.copy()
    test_xsim = test_data.copy()
    
    train_xsim["c"] =1
    val_xsim["c"] =1
    test_xsim["c"] =1

    train_xsim.to_csv(xsim_dir+'train.txt', index=False, header=False, sep=' ')
    val_xsim.to_csv(xsim_dir+'val.txt', index=False, header=False, sep=' ')
    test_xsim.to_csv(xsim_dir+'test.txt', index=False, header=False, sep=' ')

    
    #LightGCN
    # train
    train_new=[]
    grouped = train_data.groupby('author_id')['mention_id'].apply(list).reset_index()
    for index, row in grouped.iterrows():
        train_new.append(f"{row['author_id']} {' '.join(map(str, row['mention_id']))}")
    with open(lgcn_dir+'train.txt', 'w') as f:
        for line in train_new:
            f.write(line+'\n')

    # val
    val_new=[]
    grouped = val_data.groupby('author_id')['mention_id'].apply(list).reset_index()
    for index, row in grouped.iterrows():
        val_new.append(f"{row['author_id']} {' '.join(map(str, row['mention_id']))}")
    with open(lgcn_dir+'val.txt', 'w') as f:
        for line in val_new:
            f.write(line+'\n')

    # test
    test_new=[]
    grouped = test_data.groupby('author_id')['mention_id'].apply(list).reset_index()
    for index, row in grouped.iterrows():
        test_new.append(f"{row['author_id']} {' '.join(map(str, row['mention_id']))}")
    with open(lgcn_dir+'test.txt', 'w') as f:
        for line in test_new:
            f.write(line+'\n')

    return 0
