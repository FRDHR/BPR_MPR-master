import random
from collections import defaultdict              #构造一个字典，普通的字典调用不存在的元素报错；defaultdict生成的字典调用不存在的元素，返回的是一个默认值
import numpy as np
from sklearn.metrics import roc_auc_score
import scores

class BPR:
    user_count = 943  #用户
    item_count = 1682  #物品
    latent_factors = 20  #矩阵维度k
    lr = 0.01  #梯度步长α
    reg = 0.01    #正则化参数
    train_count = 1000 #迭代次数
    train_data_path = 'train.txt'    #训练集路径
    test_data_path = 'test.txt'      #测试集路径
    size_u_i = user_count * item_count
    # latent_factors of U & V
    U = np.random.rand(user_count, latent_factors) * 0.01     #返回一个或一组服从“0~1”均匀分布的随机样本值
    V = np.random.rand(item_count, latent_factors) * 0.01
    biasV = np.random.rand(item_count) * 0.01
    test_data = np.zeros((user_count, item_count))            #生成一个全零矩阵
    test = np.zeros(size_u_i)
    predict_ = np.zeros(size_u_i)
    #对训练集操作
    def load_data(self, path):
        user_ratings = defaultdict(set)                       #生成一个字典，将其设为set类型
        max_u_id = -1
        max_i_id = -1
        with open(path, 'r') as f:
            for line in f.readlines():                        #readlines()从文件中一行一行地读数据，返回一个列表
                u, i = line.split(" ")                        #以空格为分隔符
                u = int(u)
                i = int(i)
                user_ratings[u].add(i)                       #{u1:{i_1,i_3,...},u2:{i_2,i_6,...},...}
                max_u_id = max(u, max_u_id)                  #训练集中U的最大值
                max_i_id = max(i, max_i_id)                  #训练集中i的最大值
        return user_ratings
    #对测试集操作
    def load_test_data(self, path):
        file = open(path, 'r')
        for line in file:
            line = line.split(' ')
            user = int(line[0])
            item = int(line[1])
            self.test_data[user - 1][item - 1] = 1

    def train(self, user_ratings_train):
        for user in range(self.user_count):
            # sample a user
            u = random.randint(1, self.user_count)  #随机找寻一个用户U
            if u not in user_ratings_train.keys():   #当U不在训练集中重新随机生成
                continue
            # sample a positive item from the observed items
            i = random.sample(user_ratings_train[u], 1)[0]   #从与u相关的物品中随机选一个
            # sample a negative item from the unobserved items
            j = random.randint(1, self.item_count)   #从与u无关的物品中随机选一个
            while j in user_ratings_train[u]:
                j = random.randint(1, self.item_count)
            u -= 1
            i -= 1
            j -= 1
            r_ui = np.dot(self.U[u], self.V[i].T) + self.biasV[i]  #由矩阵乘法计算用户u对物品i的期望
            r_uj = np.dot(self.U[u], self.V[j].T) + self.biasV[j]  #由矩阵乘法计算用户u对物品j的期望
            r_uij = r_ui - r_uj
            loss_func = -1.0 / (1 + np.exp(r_uij))   #损失函数sigmod
            # update U and V  更新模型参数
            self.U[u] += -self.lr * (loss_func * (self.V[i] - self.V[j]) + self.reg * self.U[u])
            self.V[i] += -self.lr * (loss_func * self.U[u] + self.reg * self.V[i])
            self.V[j] += -self.lr * (loss_func * (-self.U[u]) + self.reg * self.V[j])
            # update biasV
            self.biasV[i] += -self.lr * (loss_func + self.reg * self.biasV[i])
            self.biasV[j] += -self.lr * (-loss_func + self.reg * self.biasV[j])

    def predict(self, user, item):
        predict = np.mat(user) * np.mat(item.T)
        return predict

    def main(self):
        user_ratings_train = self.load_data(self.train_data_path) #将训练集变为{u1:{i_1,i_3,...},u2:{i_2,i_6,...},...}的形式
        self.load_test_data(self.test_data_path)  #将测试集数据以二维矩阵的形式存在test_data中
        for u in range(self.user_count):
            for item in range(self.item_count):
                if int(self.test_data[u][item]) == 1:      #对于存在于测试集的物品在一维test中标记为1
                    self.test[u * self.item_count + item] = 1
                else:                                      #不存在标记为0
                    self.test[u * self.item_count + item] = 0
        # training
        for i in range(self.train_count):  #迭代次数
            self.train(user_ratings_train)
        predict_matrix = self.predict(self.U, self.V)
        # prediction  模型评价
        self.predict_ = predict_matrix.getA().reshape(-1)
        self.predict_ = pre_handel(user_ratings_train, self.predict_, self.item_count)
        auc_score = roc_auc_score(self.test, self.predict_)
        print('AUC:', auc_score)
        # Top-K evaluation 自定义模型评价
        str(scores.topK_scores(self.test, self.predict_, 5, self.user_count, self.item_count))

def pre_handel(set, predict, item_count):
    # Ensure the recommendation cannot be positive items in the training set.
    for u in set.keys():
        for j in set[u]:
            predict[(u - 1) * item_count + j - 1] = 0
    return predict

if __name__ == '__main__':
    bpr = BPR()
    bpr.main()