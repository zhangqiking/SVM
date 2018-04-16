import numpy as np 
from utils import *
import csv, sys
import os
filepath = os.path.dirname(os.path.abspath(__file__))


class SVM(object):
    def __init__(self, C, epsi, max_iter):
        self.C = C
        self.epsi = epsi
        self.max_iter = max_iter
    
    def init_data(self, train_x, train_y):
        self.train_x = train_x 
        self.train_y = train_y
        self.train_num = train_x.shape[0]
        self.alpha = np.zeros(self.train_num)
        self.b = 0
         
        self.E = np.zeros(self.train_num)
        for i in range(self.train_num):
            self.E[i] = self.calcuate_E(train_x[i, :], train_y[i])
        
    def predict(self, x):
        g_x = 0.0
        for i in range(self.train_num):
            g_x += self.alpha[i] * self.train_y[i] * kernel(self.train_x[i,:], x)
        g_x += self.b
        return self.sgn(g_x)
    
    def calculate_eta(self, x_1, x_2):
        eta = kernel(x_1, x_1) + kernel(x_2, x_2) -2 * kernel(x_1, x_2)
        return eta

    def calcuate_E(self, x, y):
        E = self.predict(x) - y
        return E
    
    def alpha_cut(self, L, H, alpha_unc):
        if alpha_unc > H:
            return H
        elif alpha_unc < L:
            return L
        else:
            return alpha_unc

    def sgn(self, e):
        if e >= 0:
            return 1
        else:
            return -1
    
    def select_out(self, train_X, train_y):
        alpha_1_index = None
        for i in range(self.train_num):
            if 0 < self.alpha[i] < self.C:
                if abs(train_y[i] * self.predict(train_X[i,:]) - 1) > self.epsi:
                    alpha_1_index = i
        if alpha_1_index == None:
            for i in range(self.train_num):
                if self.alpha[i] == 0:
                    if train_y[i] * self.predict(train_X[i,:]) < 1 - self.epsi:
                        alpha_1_index = i
        if alpha_1_index == None:
            for i in range(self.train_num):
                if self.alpha[i] == self.C:
                    if train_y[i] * self.predict(train_X[i,:]) > 1 + self.epsi:
                        alpha_1_index = i
        
        return alpha_1_index
    
    def select_inner(self, E_1, train_X, train_y):
        E_2 = None
        max_abs_diff = 0
        alpha_2_index = None
        for i in range(self.train_num):
            E_i = self.calcuate_E(train_X[i,:], train_y[i])
            abs_diff = abs(E_1 - E_i)
            if abs_diff > max_abs_diff:
                max_abs_diff =abs_diff
                alpha_2_index = i

        return alpha_2_index
    
    def check_KTT(self, train_x, train_y):
        flag = False
        sum_alpha_y = np.dot(self.alpha, train_y)
        if abs(sum_alpha_y) > self.epsi:
            flag = True
        for i in range(self.train_num):
            y_gx = train_y[i] * self.predict(train_x[i, :])
            if self.alpha[i] < -self.epsi or self.alpha[i] > self.C+self.epsi:
                flag = True
                break
            if self.alpha[i] == 0:
                if y_gx < 1 - self.epsi:
                    flag = True
                    break
            elif self.alpha[i] == self.C:
                if y_gx > 1 + self.epsi:
                    flag = True
                    break
            else:
                if abs(y_gx - 1) > self.epsi:
                    flag = True
                    break
        return flag


    def fit(self, train_x, train_y, select_ij_mode='random'):
        self.init_data(train_x, train_y)
        
        check_diff = [1 for i in range(10)]
        global_step = 0
        while True:
            alpha_pre = np.copy(self.alpha)
            alpha_1_index = None
            alpha_2_index = None
            E_1 = None
            E_2 = None
            
            if select_ij_mode == 'random':
                alpha_1_index = np.random.randint(0, self.train_num-1)
                alpha_2_index = get_rnd_int(0, self.train_num-1, alpha_1_index)
                E_1 = self.calcuate_E(train_x[alpha_1_index,:], train_y[alpha_1_index])
            else:
                alpha_1_index = self.select_out(train_x, train_y)
                if alpha_1_index == None:
                    continue
                E_1 = self.calcuate_E(train_x[alpha_1_index,:], train_y[alpha_1_index])
                alpha_2_index = self.select_inner(E_1, train_x, train_y) 
                
            E_2 = self.calcuate_E(train_x[alpha_2_index,:], train_y[alpha_2_index])

            # update alpha_2
            eta = self.calculate_eta(train_x[alpha_1_index,:], train_x[alpha_2_index, :])
            alpha_2_unc = self.alpha[alpha_2_index] + (train_y[alpha_2_index]*(E_1 - E_2)) / eta
            L, H = range_L_H(self.C, 
                             self.alpha[alpha_1_index],
                             self.alpha[alpha_2_index],
                             train_y[alpha_1_index],
                             train_y[alpha_2_index])

            alpha_2_new = self.alpha_cut(L, H, alpha_2_unc)

            # update alpha_1
            alpha_1_new = self.alpha[alpha_1_index] + train_y[alpha_1_index]*train_y[alpha_2_index]*(self.alpha[alpha_2_index] - alpha_2_new)

            # update threshold b
            b_1_new = -1 * E_1- \
                      train_y[alpha_1_index]*kernel(train_x[alpha_1_index], train_x[alpha_1_index])*(alpha_1_new - self.alpha[alpha_1_index]) - \
                      train_y[alpha_2_index]*kernel(train_x[alpha_2_index], train_x[alpha_1_index])*(alpha_2_new - self.alpha[alpha_2_index]) + \
                      self.b
            b_2_new = -1 * E_2 - \
                      train_y[alpha_1_index]*kernel(train_x[alpha_1_index], train_x[alpha_2_index])*(alpha_1_new - self.alpha[alpha_1_index]) - \
                      train_y[alpha_2_index]*kernel(train_x[alpha_2_index], train_x[alpha_2_index])*(alpha_2_new - self.alpha[alpha_2_index]) + \
                      self.b
            b_new = (b_1_new + b_2_new) / 2


            self.alpha[alpha_1_index] = alpha_1_new
            self.alpha[alpha_2_index] = alpha_2_new
            self.b = b_new
                
    
            # update E
            tmp_e_1 = 0.0
            tmp_e_2 = 0.0
            for i in range(self.train_num):
                if 0 < self.alpha[i] < self.C:
                    tmp_e_1 += train_y[alpha_1_index] * alpha_1_new * kernel(train_x[alpha_1_index,:], train_x[i,:])
                    tmp_e_2 += train_y[alpha_2_index] * alpha_2_new * kernel(train_x[alpha_2_index,:], train_x[i,:])
            self.E[alpha_1_index] = self.sgn(tmp_e_1 + b_new) - train_y[alpha_1_index]
            self.E[alpha_2_index] = self.sgn(tmp_e_2 + b_new) - train_y[alpha_2_index]


            # check stop point
            diff = np.linalg.norm(self.alpha - alpha_pre)
            check_diff.pop(0)
            check_diff.append(abs(diff))
            print(check_diff)
            if np.mean(check_diff) < 0.01:
                break

            flag = self.check_KTT(train_x, train_y)
            if flag == True and global_step<self.max_iter:
                global_step += 1
                continue 
            else:
                print('ktt')
                break

def get_rnd_int(a,b,z):
        i = z
        cnt=0
        while i == z and cnt<1000:
            i = np.random.randint(a,b)
            cnt=cnt+1
        return i
def readData(filename, header=True):
    data, header = [], None
    with open(filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        if header:
            header = spamreader.next()
        for row in spamreader:
            #print(row)
            data.append(row)
    return (np.array(data), np.array(header))
            

if __name__ == '__main__':

    # from sklearn.datasets import load_iris
    filename='SVM-w-SMO/data/iris-virginica.txt'
    (data, _) = readData('%s/%s' % (filepath, filename), header=False)
    X, y = data[:,0:-1].astype(float), data[:,-1].astype(int)

    print(X.shape)
    print(y)
    from sklearn.svm import SVC
    clf = SVC(gamma='auto')
    clf.fit(X, y) 
    pre = clf.predict(X)
    print(np.mean(pre==y))

    
    model = SVM(C = 1.0, epsi = 0.001, max_iter = 100)
    model.fit(X, y, 'd')
    print(model.alpha)
    print(model.b)
    label = []
    for i in range(150):
        pre = model.predict(X[i,:])
        if pre > 0:
            label.append(1)
        else:
            label.append(-1)
    print(y)
    print(label)
    print(np.mean(y==label))
    
    


                
            


                
                        


            

    

                      
            

            




            
            



                        





