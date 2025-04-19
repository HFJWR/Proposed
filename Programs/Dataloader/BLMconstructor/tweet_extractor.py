import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import time

def get_tweet_status(tweet):
    if 'referenced_tweets' in tweet:
        refs = tweet.get('referenced_tweets')
        for ref in refs:
            if ref['type'] == 'retweeted':
                return 'retweet'
            elif ref['type'] == 'replied_to':
                return 'reply'
            elif ref['type'] == 'quoted':
                return 'quote'
    else:
        return 'tweet'

def extract_retweets(jsonfile, start_date, end_date):
    """
    ex)
    jsonfile: path to blm source dataset
    start_date = 2020-06-01
    end_date = 2020-06-14
    """

    tweets = []
    retweets = []
    label_tweets = ["created", "tweet_id", "author_id", "tags", "text"]
    label_retweets = ["created", "tweet_id", "author_id", "mention_id", "text"]

    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    with open(jsonfile, 'rt') as f:
        for l, line in enumerate(tqdm(f)):
            elm = json.loads(line)
            data_list = elm.get("data")

            for l in range(len(data_list)):
                tags = []
                data = dict(data_list[l])
                created_at = datetime.strptime(data["created_at"], '%Y-%m-%dT%H:%M:%S.%fZ')                
                if start_date <= created_at <= end_date:
                    if data.get("entities") is not None:    #  and data["entities"].get("mentions") is not None
                        
                        created_at = data["created_at"]
                        tweet_id = data["id"]
                        author_id = data["author_id"]
                        text = data["text"]
                        entities = dict(data.get("entities"))

                        status = get_tweet_status(data)
                        
                        if status == "tweet":
                            if entities.get("hashtags") is not None:
                                tag_status = entities.get("hashtags")
                                for i in tag_status:
                                    tag = i.get("tag")
                                    tags.append(tag)
                                tweets.append([created_at, tweet_id, author_id, str(tags), text])

                        elif status == "retweet":
                            if entities.get("mentions") is not None:
                                mentions = entities.get("mentions")
                                mention = dict(mentions[0])
                                mention_id = mention["id"]
                                retweets.append([created_at, tweet_id, author_id, mention_id, text])


        tweets = pd.DataFrame(tweets, columns = label_tweets)
        retweets = pd.DataFrame(retweets, columns = label_retweets)

    return tweets, retweets

def read_jsonl(jsonfile, start_date, end_date):
    """
    jsonfile: path to blm source dataset
    start_date = 2020-06-01
    end_date = 2020-06-14
    """

    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    with open(jsonfile, 'rt') as f:
        for l, line in enumerate(tqdm(f)):
            elm = json.loads(line)
            # print(elm)
            # print("###################")
            time.sleep(1)
            data_list = elm.get("data")
            # print(data_list)
            # print("##################################")
            for l in range(len(data_list)):
                data = dict(data_list[l])
                print(data)
                print("#################################")
                time.sleep(1)
                # created_at = datetime.strptime(data["created_at"], '%Y-%m-%dT%H:%M:%S.%fZ')                
            #     if start_date <= created_at <= end_date:
            #         if data.get("entities") is not None and data["entities"].get("mentions") is not None:
            #             entities = dict(data.get("entities"))
            #             mentions = entities.get("mentions")
            #             status = get_tweet_status(data)
            #             tweet_id = data["id"]
            #             author_id = data["author_id"]
            #             text = data["text"]

            #             # RT only
            #             if status == "retweet":
            #                 mention = dict(mentions[0])
            #                 mention_id = mention["id"]
            #                 retweets.append([data["created_at"], tweet_id, author_id, mention_id, text])
        
        # df = pd.DataFrame(retweets, columns = label_retweets)
    return 0