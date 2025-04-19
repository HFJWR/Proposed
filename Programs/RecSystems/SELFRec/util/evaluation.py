import math

import numpy as np
from scipy.stats import entropy
import csv
import time
from datetime import datetime
import os


class Metric(object):
    def __init__(self):
        pass

    @staticmethod
    def hits(origin, res):
        hit_count = {}
        for user in origin:
            items = list(origin[user].keys())
            predicted = [item[0] for item in res[user]]
            hit_count[user] = len(set(items).intersection(set(predicted)))
        return hit_count

    @staticmethod
    def hit_ratio(origin, hits):
        """
        Note: This type of hit ratio calculates the fraction:
         (# retrieved interactions in the test set / #all the interactions in the test set)
        """
        total_num = 0
        for user in origin:
            items = list(origin[user].keys())
            total_num += len(items)
        hit_num = 0
        for user in hits:
            hit_num += hits[user]
        return round(hit_num/total_num,5)

    # # @staticmethod
    # def hit_ratio(origin, hits):
    #     """
    #     Note: This type of hit ratio calculates the fraction:
    #      (# users who are recommended items in the test set / #all the users in the test set)
    #     """
    #     hit_num = 0
    #     for user in hits:
    #         if hits[user] > 0:
    #             hit_num += 1
    #     return hit_num / len(origin)

    @staticmethod
    def precision(hits, N):
        prec = sum([hits[user] for user in hits])
        return round(prec / (len(hits) * N),5)

    @staticmethod
    def recall(hits, origin):
        recall_list = [hits[user]/len(origin[user]) for user in hits]
        recall = round(sum(recall_list) / len(recall_list),5)
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return round(2 * prec * recall / (prec + recall),5)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error+=abs(entry[2]-entry[3])
            count+=1
        if count==0:
            return error
        return round(error/count,5)

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += (entry[2] - entry[3])**2
            count += 1
        if count==0:
            return error
        return round(math.sqrt(error/count),5)

    @staticmethod
    def NDCG(origin,res,N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            #1 = related, 0 = unrelated
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG+= 1.0/math.log(n+2,2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG+=1.0/math.log(n+2,2)
            sum_NDCG += DCG / IDCG
        return round(sum_NDCG / len(res),5)


    @staticmethod
    def coverage(origin,res,N):
        """
        res: top-k items
        origin: pos items
        """

        # save result
        now_date = f"{datetime.now():%m%d-%H%M%S}"
        output_dir = "./output/"+now_date+"/"
        os.makedirs(output_dir, exist_ok=True)

        res_dict = {}
        print("save result")
        for user in res:
            rec_items = []
            for _, item in enumerate(res[user]):
                rec_items.append(int(item[0]))
            res_dict[user] = rec_items
        
        with open (output_dir+"results_xsimgcl.txt", "w") as file:
            for user, rec_items in res_dict.items():
                rec_items_str = str(rec_items)
                file.write(f'{{"{user}":{rec_items_str}}}\n')        

        # top-k category
        item_category = get_category_dict()
        topk_categories = {}
        for user in res:
            topk_categories_per_user= []
            for _, item in enumerate(res[user]):
                if int(item[0]) in item_category:
                    topk_categories_per_user.append(item_category[int(item[0])])
            topk_categories[user] = topk_categories_per_user

        # カテゴリ保存
        with open(output_dir+"result_xsimgcl_category.txt", "w") as file:
            for user, categories in topk_categories.items():
                file.write(f'{{"{user}":{categories}}}\n')
        
        # coverage
        cov = {'cov_max': 0, 'cov_min': 10000}
        cat = []
        users = len(topk_categories)
        for user, categories in topk_categories.items():
            unique = list(set(categories))        
            cat.append(len(unique))
            # coverage 最大、最小保存
            if len(unique)>cov['cov_max']:
                cov['cov_max'] = len(unique)
                cov['max_categories'] = categories
                cov['max_user'] = user
            if len(unique)<cov['cov_min']:
                cov['cov_min'] = len(unique)
                cov['min_categories'] = categories
                cov['min_user'] = user
        
        # save min,max
        with open(output_dir+"diversity_maxmin_xsimgcl.txt", "a") as file:
            file.write("\n"+str(cov))

        return round(sum(cat)/users, 5)


    @staticmethod
    def entropy(origin,res,N):
        """
        res: top-k items
        origin: pos items
        """
        item_category = get_category_dict()
        topk_categories = {}
        for user in res:
            topk_categories_per_user= []
            for _, item in enumerate(res[user]):
                if int(item[0]) in item_category:
                    topk_categories_per_user.append(item_category[int(item[0])])
            topk_categories[user] = topk_categories_per_user

        entropies=[]
        ent = {'ent_max': 0, 'ent_min': 10000}
        for user, categories in topk_categories.items():
            a, count = np.unique(categories, return_counts=True)
            user_entropy = entropy(count)
            entropies.append(user_entropy)
            # entropy 最大、最小保存
            if user_entropy>ent['ent_max']:
                ent['ent_max'] = user_entropy
                ent['max_categories'] = categories
                ent['max_user'] = user
            if user_entropy<ent['ent_min']:
                ent['ent_min'] = user_entropy
                ent['min_categories'] = categories
                ent['min_user'] = user
        
        # save min,max
        with open("entcate_maxmin_xsimgcl.txt", "a") as file:
            file.write("\n"+str(ent))
        
        return round(np.mean(entropies), 5)
    
    @staticmethod
    def gini(origin,res,N):
        """
        res: top-k items
        origin: pos items
        """
        item_category=get_category_dict()
        topk_category_list = []
        for user in res:
            topk_categories_per_user= []
            for _, item in enumerate(res[user]):
                if int(item[0]) in item_category:
                    topk_categories_per_user.append(item_category[int(item[0])])
            topk_category_list.append(topk_categories_per_user)

        ginis=[]
        for i in topk_category_list:
            a, count = np.unique(i, return_counts=True)
            count = np.sort(count)
            n = len(count)
            cum_count = np.cumsum(count)
            ginia = (n+1-2*np.sum(cum_count)/cum_count[-1])/n
            ginis.append(ginia)
        return round(np.mean(ginis), 5)
        

    # @staticmethod
    # def MAP(origin, res, N):
    #     sum_prec = 0
    #     for user in res:
    #         hits = 0
    #         precision = 0
    #         for n, item in enumerate(res[user]):
    #             if item[0] in origin[user]:
    #                 hits += 1
    #                 precision += hits / (n + 1.0)
    #         sum_prec += precision / min(len(origin[user]), N)
    #     return sum_prec / len(res)

    # @staticmethod
    # def AUC(origin, res, rawRes):
    #
    #     from random import choice
    #     sum_AUC = 0
    #     for user in origin:
    #         count = 0
    #         larger = 0
    #         itemList = rawRes[user].keys()
    #         for item in origin[user]:
    #             item2 = choice(itemList)
    #             count += 1
    #             try:
    #                 if rawRes[user][item] > rawRes[user][item2]:
    #                     larger += 1
    #             except KeyError:
    #                 count -= 1
    #         if count:
    #             sum_AUC += float(larger) / count
    #
    #     return float(sum_AUC) / len(origin)

def get_category_dict():
    item_category_dict = {}
    dataset = "BLM"
    with open("dataset/"+dataset+"/item_category.txt", newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            item_id, category_id = map(int, row)
            item_category_dict[item_id] = category_id
    return(item_category_dict)

def ranking_evaluation(origin, res, N):
    measure = []
    for n in N:
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
        indicators = []
        if len(origin) != len(predicted):
            print('The Lengths of test set and predicted set do not match!')
            exit(-1)
        hits = Metric.hits(origin, predicted)
        hr = Metric.hit_ratio(origin, hits)
        indicators.append('Hit Ratio:' + str(hr) + '\n')
        prec = Metric.precision(hits, n)
        indicators.append('Precision:' + str(prec) + '\n')
        recall = Metric.recall(hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')
        # F1 = Metric.F1(prec, recall)
        # indicators.append('F1:' + str(F1) + '\n')
        #MAP = Measure.MAP(origin, predicted, n)
        #indicators.append('MAP:' + str(MAP) + '\n')
        NDCG = Metric.NDCG(origin, predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        # AUC = Measure.AUC(origin,res,rawRes)
        # measure.append('AUC:' + str(AUC) + '\n')
        coverage = Metric.coverage(origin, predicted, n)
        indicators.append('coverage:' + str(coverage) + '\n')
        entropy = Metric.entropy(origin, predicted, n)
        indicators.append('entropy:' + str(entropy) + '\n')
        gini = Metric.gini(origin, predicted, n)
        indicators.append('gini:' + str(gini) + '\n')
        measure.append('Top ' + str(n) + '\n')
        measure += indicators
    return measure

def rating_evaluation(res):
    measure = []
    mae = Metric.MAE(res)
    measure.append('MAE:' + str(mae) + '\n')
    rmse = Metric.RMSE(res)
    measure.append('RMSE:' + str(rmse) + '\n')
    return measure