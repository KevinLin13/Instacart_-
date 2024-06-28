# 參考網址: https://www.kaggle.com/code/mmueller/f1-score-expectation-maximization-in-o-n
# 根據預測的機率最大化 F1-score，從而選擇最佳的產品列表。

import numpy as np
from operator import itemgetter

class F1Optimizer():
    # 初始化
    def __init__(self):
        pass

    # 計算不同情況下的 F1-score 期望值，返回期望值矩陣。
    @staticmethod
    def get_expectations(P, pNone=None):
        expectations = []
        P = np.sort(P)[::-1]

        n = np.array(P).shape[0]
        DP_C = np.zeros((n + 2, n + 1))
        if pNone is None:
            pNone = (1.0 - np.array(P)).prod()

        DP_C[0][0] = 1.0
        for j in range(1, n):
            DP_C[0][j] = (1.0 - P[j - 1]) * DP_C[0, j - 1]

        for i in range(1, n + 1):
            DP_C[i, i] = DP_C[i - 1, i - 1] * P[i - 1]
            for j in range(i + 1, n + 1):
                DP_C[i, j] = P[j - 1] * DP_C[i - 1, j - 1] + (1.0 - P[j - 1]) * DP_C[i, j - 1]

        DP_S = np.zeros((2 * n + 1,))
        DP_SNone = np.zeros((2 * n + 1,))
        for i in range(1, 2 * n + 1):
            DP_S[i] = 1. / (1. * i)
            DP_SNone[i] = 1. / (1. * i + 1)
        for k in range(n + 1)[::-1]:
            f1 = 0
            f1None = 0
            for k1 in range(n + 1):
                f1 += 2 * k1 * DP_C[k1][k] * DP_S[k + k1]
                f1None += 2 * k1 * DP_C[k1][k] * DP_SNone[k + k1]
            for i in range(1, 2 * k - 1):
                DP_S[i] = (1 - P[k - 1]) * DP_S[i] + P[k - 1] * DP_S[i + 1]
                DP_SNone[i] = (1 - P[k - 1]) * DP_SNone[i] + P[k - 1] * DP_SNone[i + 1]
            expectations.append([f1None + 2 * pNone / (2 + k), f1])

        return np.array(expectations[::-1]).T

    # 最大化 F1-score 的期望值，返回最佳的產品數量 best_k，是否預測 None 以及最大 F1-score。
    @staticmethod
    def maximize_expectation(P, pNone=None):
        expectations = F1Optimizer.get_expectations(P, pNone)
        ix_max = np.unravel_index(expectations.argmax(), expectations.shape)
        max_f1 = expectations[ix_max]

        predNone = True if ix_max[0] == 0 else False
        best_k = ix_max[1]

        return best_k, predNone, max_f1

    # 計算 F1-score 和 F-beta score 的靜態方法。
    @staticmethod
    def _F1(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn)
    @staticmethod
    def _Fbeta(tp, fp, fn, beta=1.0):
        beta_squared = beta ** 2
        return (1.0 + beta_squared) * tp / ((1.0 + beta_squared) * tp + fp + beta_squared * fn)

# 根據預測機率選擇最佳產品列表，最大化 F1-score
def get_best_prediction(items=None, preds=None, pNone=None, showThreshold=False):

    # 將產品和對應的機率組合並排序，以機率從大到小排列
    # itemgetter(1) 在排序或選擇操作中，根據每個元素的第二個子元素進行排序
    items_preds = sorted(list(zip(items, preds)), key=itemgetter(1), reverse=True)
    P = [p for i, p in items_preds]
    L = [i for i, p in items_preds]

    # 計算 None 類別的機率，如果未提供則根據所有產品都不被重新訂購的機率來計算。這個計算方式假設每個產品被重新訂購的機率是獨立的。
    if pNone is None:
        pNone = (1.0 - np.array(P)).prod()

    # 使用 maximize_expectation 方法找到最佳的產品數量和是否選擇 None 類別。
    opt = F1Optimizer.maximize_expectation(P, pNone)

    # 根據最大化的結果生成最佳產品列表
    best_prediction = ['None'] if opt[1] else []
    best_prediction += (L[:opt[0]])

    if showThreshold:
        print("Threshold : P(X) > {:.4f}".format(P[:opt[0]][-1]))
        print("Maximum F1 : {:.4f}".format(opt[2]))

    return ' '.join(list(map(str, best_prediction)))