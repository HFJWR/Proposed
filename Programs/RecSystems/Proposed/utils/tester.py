import csv
import pdb
import logging
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy
from collections import Counter
import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import time
from scipy.special import comb
import numpy as np
import torch
from functools import lru_cache
from datetime import datetime
import os




class Tester(object):
    def __init__(self, args, model, dataloader):
        self.args = args
        self.model = model
        # self.model_mf = args.model_mf
        self.history_dic = dataloader.historical_dict
        self.history_csr = dataloader.train_csr
        self.dataloader = dataloader.dataloader_test
        self.test_dic = dataloader.test_dic
        self.category_dic = dataloader.category_dic
        self.cate = np.array(list(dataloader.category_dic.values()))
        self.metrics = args.metrics

    def judge(self, users, items):

        results = {metric: 0.0 for metric in self.metrics}
        # for ground truth test
        # items = self.ground_truth_filter(users, items)
        stat = self.stat(items)
        for metric in self.metrics:
            f = Metrics.get_metrics(metric)
            for i in range(len(items)):
                results[metric] += f(items[i], test_pos = self.test_dic[users[i]], num_test_pos = len(self.test_dic[users[i]]), count = stat[i], model = self.model, k=len(items))
        return results

    def ground_truth_filter(self, users, items):
        batch_size, k = items.shape
        res = []
        for i in range(len(users)):
            gt_number = len(self.test_dic[users[i]])
            if gt_number >= k:
                res.append(items[i])
            else:
                res.append(items[i][:gt_number])
        return res

    def cos_sim(self, topk_emb):
        cosine_similarity = F.cosine_similarity(topk_emb.unsqueeze(1), topk_emb.unsqueeze(0),dim=2)
        return cosine_similarity

    def MMR(self, scores, similarity_matrix, lambdaCons, topk):
        s = []
        r = list(range(len(scores)))
        
        # Move tensors to CPU if necessary and detach
        if scores.is_cuda:
            scores = scores.cpu().detach()
        else:
            scores = scores.detach()
        
        if similarity_matrix.is_cuda:
            similarity_matrix = similarity_matrix.cpu().detach()
        else:
            similarity_matrix = similarity_matrix.detach()

        scores = scores.numpy()  # Convert to numpy after detaching
        similarity_matrix = similarity_matrix.numpy()

        # Precompute all pairwise similarities and store them
        all_similarities = similarity_matrix.copy()

        while len(s) < topk:
            score = -np.inf
            selectOne = None
            
            # Vectorized computation of the first part (scores)
            firstParts = np.array([scores[i] for i in r])
            
            # Precomputed max similarities for items in r to items in s
            if len(s) > 0:
                secondParts = np.max(all_similarities[r][:, s], axis=1)
            else:
                secondParts = np.zeros(len(r))
            
            # Compute MMR scores for all candidates in r
            mmr_scores = lambdaCons * firstParts - (1 - lambdaCons) * secondParts
            
            # Select the item with the highest MMR score
            selectOne = r[np.argmax(mmr_scores)]
            
            r.remove(selectOne)
            s.append(selectOne)
            
        return s[:topk]

    def Bernoulli(self, N, k, p):
        """
        N個のアイテムの中から発生確率pのジャンルがk回サンプリングされる確率
        """
        return comb(N, k, exact=True) * p**k * (1-p)**(N-k)
    
    def Probability(self, nuggets):
        """
        ジャンルの発生確率を算出
        nuggets: ndaaray
            row: アイテム
            col: アイテムのジャンル
        """
        n_row, _ = nuggets.shape
        p = np.sum(nuggets, axis=0) / n_row

        return  p

    def fBinomDiv(self, R, scores, nuggets, com_size=0, lambd=0.00):
        n_users, topk = R.shape
        ranked_R = np.zeros_like(R)
        
        nuggets_values = np.array(list(nuggets.values()))
        p = self.Probability(nuggets_values)
        G = np.arange(com_size)

        # Coverage の事前計算
        precomputed_coverage = np.power(self.Bernoulli(topk, 0, p), 1 / len(G))

        # NonRedundancy の事前計算
        precomputed_nonred = {g: [self.Bernoulli(topk, l+1, p[g]) / (1 - self.Bernoulli(topk, 0, p[g])) for l in range(topk)] for g in G}

        for user in tqdm(range(n_users)):
            S = []  # リランキングされたセット
            available_items = R[user].tolist()  # 推薦リスト中のアイテムを処理対象に
            user_score = scores[user].tolist()
            
            score = dict(zip(available_items, user_score))
            
            G_R = np.zeros(com_size)
            k_g_R = np.zeros(com_size)
            
            bin_before = 1
            for _ in range(100):
                max_score = -float('inf')
                best_item = None

                for item in available_items:
                    S_with_item = S + [item]
                    
                    # ベクトル化による k_g_R の更新
                    new_k_g_R = k_g_R + np.array(nuggets[item])

                    # G_R_new をベクトル化
                    G_R_new = np.where(new_k_g_R > 0)[0]
                    G_NR = np.setdiff1d(G, G_R_new)

                    # Coverage の計算をベクトル化
                    Coverage = np.prod(precomputed_coverage[G_NR])

                    # NonRedundancy の計算をベクトル化
                    NonRed = 1
                    g_r = G_R_new
                    for g in g_r:
                        k_g = int(new_k_g_R[g])
                        if k_g > 1:
                            sumat = np.sum([precomputed_nonred[g][l] for l in range(k_g - 1)])
                            NonRed *= np.power((1 - sumat), 1 / len(g_r))

                    if np.isnan(NonRed):
                        NonRed = 1

                    bin_after = Coverage * NonRed
                    
                    # fBinomDiv(i; S) の計算
                    rel = score[item]
                    fBinomDiv_score = (1 - lambd) * rel + lambd * (bin_after - bin_before)
                    
                    if fBinomDiv_score > max_score:
                        max_score = fBinomDiv_score
                        best_item = item
                        best_k_g_R = new_k_g_R
                        best_G_R_new = G_R_new
                        best_bin_after = bin_after

                S.append(best_item)
                ranked_R[user, len(S) - 1] = best_item
                available_items.remove(best_item)
                k_g_R = best_k_g_R
                G_R = best_G_R_new
                bin_before = best_bin_after
        
        return ranked_R



    def test(self):
        results = {}
        h = self.model.get_embedding()
        item_emb = h['item']        
        count = 0

        # new item_category_dict
        item_category_dict = {}
        for item in range(0,46374):
            category = self.cate[item]
            item_category_dict[item] = category
        
        # nuggets 作成
        com_size = max(item_category_dict.values())+1
        nuggets = {}
        for item, com in item_category_dict.items():
            com_list = [0]*com_size
            com_list[com] = 1
            nuggets[item] = com_list

        for k in self.args.k_list:
            results[k] = {metric: 0.0 for metric in self.metrics}

        results_data = {'User': [], 'Top_K_Recommendations': []}
        category_data = {'User': [], 'Top_K_Categories': []}
        for batch in tqdm(self.dataloader):

            users = batch[0]
            count += users.shape[0]
            # count += len(users)
            scores = self.model.get_score(h, users)

            users = users.tolist()
            mask = torch.tensor(self.history_csr[users].todense(), device = scores.device).bool()
            scores[mask] = -float('inf')

            score, recommended_items = torch.topk(scores, k = max(self.args.k_list))


            # Totally Random（学習スコアを破棄し、ランダムなスコアを割当）
            # ==============================================
            # k = max(self.args.k_list)
            # recommended_items = []
            # for user_scores in scores:
            #     random_indices = torch.randperm(user_scores.size(0))
            #     recommended_items.append(random_indices[:k])
            # recommended_items = torch.stack(recommended_items)
            # print(recommended_items.shape)
            # ==============================================


            # Reranking Random（ランダムリランキング）
            # ==========================================
            # indices = torch.randperm(recommended_items.size(0))
            # recommended_items = recommended_items[indices]
            # ==========================================

            # MMR
            # =========================================
            # new_scr = []
            # new_topk = []
            # for scr, topk in tqdm(zip(score, recommended_items)):
            #     topk_emb=[]
            #     for item in topk:
            #         topk_emb.append(item_emb[item].clone().detach())
            #     topk_emb = torch.stack(topk_emb, dim=0)

            #     # 各ユーザのリランキング
            #     sim_matrix = self.cos_sim(topk_emb)
            #     rerank = self.MMR(scr, sim_matrix, lambdaCons=0.8, topk=300) # 1で元のランキング

            #     new_scr.append(scr[rerank])
            #     new_topk.append(topk[rerank])
            # # リランキング反映
            # score = torch.stack(new_scr)
            # recommended_items = torch.stack(new_topk)
            # =========================================
            
            # # Binomial Diversity
            # # =========================================
            # """
            # R: recommended_items
            # scores: score
            # nuggets: nuggets
            # """
            
            # recommended_items = recommended_items.cpu()
            # score = score.cpu()

            # recommended_items = self.fBinomDiv(recommended_items, score, nuggets, com_size=69, lambd=0.75) # 0で元のランキング
            # recommended_items = torch.tensor(recommended_items)
            # # # =========================================



            recommended_items = recommended_items.cpu()
            for k in self.args.k_list:

                results_batch = self.judge(users, recommended_items[:, :k])

                for metric in self.metrics:
                    results[k][metric] += results_batch[metric]

            for user, items in zip(users, recommended_items):
                results_data['User'].append(user)
                results_data['Top_K_Recommendations'].append(items.tolist())
                
                categories = [self.cate[item] for item in items]
                category_data['User'].append(user)
                category_data['Top_K_Categories'].append(categories)


        # 推薦結果保存
        now_date = f"{datetime.now():%m%d-%H%M%S}"
        output_dir = "./output/"+now_date+"/"
        os.makedirs(output_dir, exist_ok=True)

        with open(output_dir+'result_item.txt', 'w') as f:
            for user, items in zip(results_data['User'], results_data['Top_K_Recommendations']):
                items_str = str(items)
                f.write(f'{{"{user}":{items_str}}}\n')
        # カテゴリ結果保存
        with open(output_dir+'result_category.txt', 'w') as f:
            for user, categories in zip(category_data['User'], category_data['Top_K_Categories']):
                f.write(f"{user},{categories}\n")

        for k in self.args.k_list:
            for metric in self.metrics:
                results[k][metric] = results[k][metric] / count
        self.show_results(results)

        # 各種評価指標
        cov = {'cov_max': 0, 'cov_min': 10000}
        ent = {'ent_max': 0, 'ent_min': 10000}
        topk_categories = {}
    
        # entropy
        ents=[]
        for user, items in zip(results_data["User"], results_data["Top_K_Recommendations"]):
            # item to category
            topk_categories_per_user = []
            for item in items:
                category = self.cate[item]
                topk_categories_per_user.append(category)
            topk_categories[user] = topk_categories_per_user

            # entropy
            user_entropy = entropy(np.unique(topk_categories_per_user, return_counts=True)[1])
            ents.append(user_entropy)
            # entorpy 最大、最小保存
            if user_entropy>ent['ent_max']:
                ent['ent_max'] = user_entropy
                ent['max_categories'] = topk_categories_per_user
                ent['max_user'] = user
            if user_entropy<ent['ent_min']:
                ent['ent_min'] = user_entropy
                ent['min_categories'] = topk_categories_per_user
                ent['min_user'] = user

        # print('entropy(max-k指標): ', round(np.mean(ents), 5))

        # Coverage
        cov_users = len(topk_categories)
        cat=[]
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
        coverage = sum(cat)/cov_users
        # print("coverage: "+str(coverage))

        # Coverage, Entropy最大ユーザの推薦リスト（カテゴリ）表示
        # print(cov)
        # print("="*90)
        # print(ent)

        # 最大最小保存
        print("save max,min entropy, category")
        with open(output_dir+"diversity_maxmin.txt", "w") as file:
            file.write(str(cov)+"\n"+str(ent))

    def show_results(self, results):
        for metric in self.metrics:
            for k in self.args.k_list:
                logging.info('For top{}, metric {} = {}'.format(k, metric, results[k][metric]))

    def stat(self, items):
        stat = [np.unique(self.cate[item], return_counts=True)[1] for item in items]
        return stat


class Metrics(object):

    def __init__(self):
        pass

    @staticmethod
    def get_metrics(metric):

        metrics_map = {
            'recall': Metrics.recall,
            'precision': Metrics.precision,
            'ndcg': Metrics.ndcg,
            'hit_ratio': Metrics.hr,
            'coverage': Metrics.coverage,
            'entropy': Metrics.entropy,
            'gini': Metrics.gini
        }

        return metrics_map[metric]

    @staticmethod
    def recall(items, **kwargs):
        test_pos = kwargs['test_pos']
        num_test_pos = kwargs['num_test_pos']
        hit_count = np.isin(items, test_pos).sum()

        return round(hit_count/num_test_pos, 5)
    
    def precision(items, **kwargs):
        test_pos = kwargs['test_pos']
        num_test_pos = kwargs['num_test_pos']
        hit_count = np.isin(items, test_pos).sum()

        return hit_count/len(items)
    
    def ndcg(items, **kwargs):  #items=top-k, kwargs['test_pos']=true_items
        true_items = kwargs['test_pos']  # 正解アイテム
        #print('items',len(items))
        #print('kwargs', len(kwargs['test_pos']))
        
        test_pos = items   # pos-item(top-k)
        k = kwargs['k']

        # DCG
        # test_posを正解アイテムに従い、バイナリ変換
        rel = [1 if item in true_items else 0 for item  in test_pos]    #test_posベース
        dcg = rel[0]   # top-1の計算除外
        for i in range(1,min(k,len(rel))):
            dcg += rel[i]/np.log2(i+1)
        
        # iDCG
        #i_rel = [1] * len(true_items)
        #i_rel.sort(reverse=True)
        i_rel = np.sort(rel)[::-1]
        idcg = i_rel[0]
        for i in range(1, min(k, len(i_rel))):
            idcg+=i_rel[i]/np.log2(i+1)

        #NDCG
        ndcg = dcg/idcg if idcg>0 else 0
        return ndcg

        
    @staticmethod
    def hr(items, **kwargs):

        test_pos = kwargs['test_pos']
        hit_count = np.isin(items, test_pos).sum()

        if hit_count > 0:
            return 1.0
        else:
            return 0.0

    @staticmethod
    def coverage(items, **kwargs):

        count = kwargs['count']

        return count.size

    @staticmethod
    def entropy(items, **kwargs):

        count = kwargs['count']

        return entropy(count)

    @staticmethod
    def gini(items, **kwargs):

        count = kwargs['count']
        count = np.sort(count)
        n = len(count)
        cum_count = np.cumsum(count)

        return (n + 1 - 2 * np.sum(cum_count) / cum_count[-1]) / n
    
