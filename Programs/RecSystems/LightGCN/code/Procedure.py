'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score
import csv
import pandas as pd
from collections import Counter
import numpy as np
from scipy.stats import entropy
from parse import parse_args


CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def get_category_dict():
    args = parse_args()
    dataset = args.dataset
    item_category_dict = {}
    with open("../data/"+str(dataset)+"/item_category.txt", newline='') as csvfile:  # データセット変更時
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            item_id, category_id = map(int, row)
            item_category_dict[item_id] = category_id
    return(item_category_dict)

    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def Test(dataset, Recmodel, epoch, w_epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        topk_list=[]
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1 ##バッチ数　ユーザ数　バッチサイズ

        topk_items_dict = {}
        for batch_users in tqdm(utils.minibatch(users, batch_size=u_batch_size)):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)

            for i, user in enumerate(batch_users):
                topk_items_dict[user] = rating_K[i].tolist()

            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())       
            groundTrue_list.append(groundTrue)

            a = rating_K.cpu().numpy().tolist()
            for i in a:
                topk_list.append(i)
            del a

            topk_items_per_user = rating_K.cpu().numpy()

        item_category_dict = get_category_dict()

        topk_items_per_user=topk_list

        print("save topk_results")
        with open("results_lgcn.txt", "w") as file:
            for user, items in topk_items_dict.items():
                items_str=str(items)
                file.write(f'{{"{user}":{items_str}}}\n')

        topk_categories = {}
        for user, items in topk_items_dict.items():
            topk_categories_per_user = []
            for item in items:
                category = item_category_dict[item]
                topk_categories_per_user.append(category)                
            topk_categories[user] = topk_categories_per_user


        cov_users = len(topk_categories)
        cat=[]
        entropies=[]
        ginis=[]
        cov = {'cov_max': 0, 'cov_min': 10000}
        ent = {'ent_max': 0, 'ent_min': 10000}

        for user, categories in topk_categories.items():
            # coverage
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
            # entropy
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
            # gini
            count = np.sort(count)
            n=len(count)
            cum_count = np.cumsum(count)
            ginia = (n+1-2*np.sum(cum_count)/cum_count[-1])/n
            ginis.append(ginia)

        print(cov)
        print("="*90)
        print(ent)
        # 最大最小保存
        print("save max,min entropy, category")
        with open("entcate_maxmin_lgcn.txt", "w") as file:
            file.write(str(cov)+"\n"+str(ent))
        # カテゴリ保存
        print("save category result")
        with open("category_results_lgcn.txt", "w") as file:
            for user, categories in topk_categories.items():
                file.write(f'{{"{user}":{categories}}}\n')

        # coverage
        coverage = sum(cat)/cov_users
        print("coverage: ", coverage)

        # entropy
        mean_entropy = np.mean(entropies)
        print("entropy: ", mean_entropy)

        # gini
        gini = np.mean(ginis)
        print("gini: ",gini)

        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)

        # with open("output.txt", "w") as file:
        #     for pair in X:
        #         file.write(f"{pair[0]}, {pair[1]}\n")

        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        return results
