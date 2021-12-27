import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
import argparse
from argparse import RawTextHelpFormatter
import pdb

random.seed(123)
np.random.seed(123)

def load_data_and_norm(path, label_num):
    img_list = []
    label_list = []
    for filename in os.listdir(path):
        img = cv2.imread(path + "/" + filename, cv2.IMREAD_GRAYSCALE)
        img = (img - 0)/(255 - 0)
        img_list.append(img)
        one_hot_vec = np.zeros(3)
        one_hot_vec[label_num] = 1.
        label_list.append(one_hot_vec)
    img_arr = np.array(img_list, dtype=np.float32)
    label_arr = np.array(label_list, dtype=np.float32)
    return img_arr, label_arr

class CNN:
    def __init__(self, kernel_size, kernel_depth, input_img_w_h, fc_units, batch_size, lr, weights):
        self.kernel_size = kernel_size
        self.kernel_depth = kernel_depth
        self.input_img_w_h = input_img_w_h
        self.fc_units = fc_units
        self.batch_size = batch_size
        self.lr = lr
        self.weights = weights
        
        ##網路結構
        ##Input(1*28*28)=>convlution(3*5*5)=>relu()=>maxpooling(3*3)=>flatten()=>fullconnected(64)=>Output(10)=>softmax(10)
        weights["K1"] = 1e-2 * np.random.randn(1, self.kernel_depth, self.kernel_size, self.kernel_size).astype(np.float64) #conv1
        weights["b1"] = np.zeros(self.kernel_depth).astype(np.float64) #conv1
        weights["W2"] = 1e-2 * np.random.randn(self.kernel_depth * 13 * 13, self.fc_units).astype(np.float64) #mlp layer1
        weights["b2"] = np.zeros(self.fc_units).astype(np.float64) #mlp layer1
        weights["W3"] = 1e-2 * np.random.randn(self.fc_units, 3).astype(np.float64) #mlp layer2
        weights["b3"] = np.zeros(3).astype(np.float64) #mlp layer2

        self.neurons = {}
        self.gradients = {}
        
    def fullyconnected_forward(self, X, W, b):
        """
        :param X: 當前層的输出, (N,ln)
        :param W: 當前層的weights
        :param b: 當前層的bias
        :return: 下一層輸出
        """
        return np.dot(X, W) + b
    
    def fullyconnected_backward(self, next_dX, W, X):
        """
        :param next_dX: 下一層的梯度
        :param W: 當前層的weights
        :param X: 當前層的bias
        :return:
        """
        N = X.shape[0]
        delta = np.dot(next_dX, W.T)  # 当前层的梯度
        dw = np.dot(X.T, next_dX)  # 当前层权重的梯度
        db = np.sum(next_dX, axis=0)  # 当前层偏置的梯度, N个样本的梯度求和
        return dw / N, db / N, delta
    
    def Zeros_remove(self, X, padding): #移除padding
        """
        :param X: (N,C,H,W)
        :param paddings: (p1,p2)
        :return:
        """
        if padding[0] > 0 and padding[1] > 0:
            return X[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]
        elif padding[0] > 0:
            return X[:, :, padding[0]:-padding[0], :]
        elif padding[1] > 0:
            return X[:, :, :, padding[1]:-padding[1]]
        else:
            return X
    
    def Zeros_padding(self, dX, strides): #想多維數组最後兩位，每個行列之間增加指定的個數的零填充
        """
        :param dX: (N,D,H,W),H,W為卷積输出層的高度和寬度
        :param strides: 步長
        :return:
        """
        _, _, H, W = dX.shape
        pX = dX
        if strides[0] > 1:
            for h in np.arange(H - 1, 0, -1):
                for o in np.arange(strides[0] - 1):
                    pX = np.insert(pX, h, 0, axis=2)
        if strides[1] > 1:
            for w in np.arange(W - 1, 0, -1):
                for o in np.arange(strides[1] - 1):
                    pX = np.insert(pX, w, 0, axis=3)
        return pX

    def convolution_forward(self, X_input, Kernel, b, padding=(0, 0), strides=(1, 1)):
        """
        多通道卷積前向過程
        :param X: 卷基層矩陣, 形状(N,C,H,W)，N为batch_size，C為通道数
        :param Kernel: 卷積核, 形状(C,D,k1,k2), C為输入通道数，D為输出通道数
        :param b: bias, 形状(D,)
        :param padding: padding
        :param strides: 步長
        :return: 卷積結果
        """
        padding_X = np.lib.pad(X_input, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant',
                               constant_values=0)
        N, _, height, width = padding_X.shape
        C, D, k1, k2 = Kernel.shape
    
        ##強制整除
        ##卷積後的長度，padding為0
        H_ = 1 + (height - k1) // strides[0]
        W_ = 1 + (width - k2) // strides[1]
        conv_X = np.zeros((N, D, H_, W_))
    
        ##求和操作
        for n in np.arange(N):
            for d in np.arange(D):
                for h in np.arange(height - k1 + 1):
                    for w in np.arange(width - k2 + 1):
                        conv_X[n, d, h, w] = np.sum(padding_X[n, :, h:h + k1, w:w + k2] * Kernel[:, d]) + b[d]
        return conv_X

    def convolution_backward(self, next_dX, Kernel, X, padding=(0, 0), strides=(1, 1)):
        """
        多通道卷積層的反向過程
        :param next_dX: 卷積输出層的梯度,(N,D,H',W'),H',W'为卷积输出层的高度和宽度
        :param Kernel: 當前層卷積核，(C,D,k1,k2)
        :param X: 卷積層矩陣,形状(N,C,H,W)，N为batch_size，C为通道数
        :param padding: padding
        :param strides: 步長
        :return:
        """
        N, C, H, W = X.shape
        C, D, k1, k2 = Kernel.shape
    
        # 卷積核梯度
        padding_next_dX = self.Zeros_padding(next_dX, strides)
        # 增加高度和寬度0填充
        ppadding_next_dX = np.lib.pad(padding_next_dX, ((0, 0), (0, 0), (k1 - 1, k1 - 1), (k2 - 1, k2 - 1)), 'constant',
                                      constant_values=0)
    
        #旋轉180度
        # 卷積核高度和寬度翻轉180度
        flip_K = np.flip(Kernel, (2, 3))
        # 交換C,D為D,C；D變為輸入通道數了，C變為輸出通道數了
        switch_flip_K = np.swapaxes(flip_K, 0, 1)
    
        ##rot(180)*W
        dX = self.convolution_forward(ppadding_next_dX.astype(np.float64), switch_flip_K.astype(np.float64), np.zeros((C,), dtype=np.float64))
    
        # 求卷積核的梯度dK
        swap_W = np.swapaxes(X, 0, 1)  # 变为(C,N,H,W)与
        dW = self.convolution_forward(swap_W.astype(np.float64), padding_next_dX.astype(np.float64), np.zeros((D,), dtype=np.float64))
    
        # 偏置的梯度
        db = np.sum(np.sum(np.sum(next_dX, axis=-1), axis=-1), axis=0)  # 在高度、寬度上相加；批量大小上相加
    
        # 把padding减掉
        dX = self.Zeros_remove(dX, padding)
    
        return dW / N, db / N, dX
    
    def maxpooling_forward(self, X, pooling, strides=(2, 2), padding=(0, 0)):
        """
        最大池化前向過程
        :param X: 卷積層矩陣,形状(N,C,H,W)，N为batch_size，C為通道數
        :param pooling: 池化大小(k1,k2)
        :param strides: 步長
        :param padding: 0填充
        :return:
        """
        N, C, H, W = X.shape
        # 零填充
        padding_X = np.lib.pad(X, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)
    
        # 输出的高度和宽度
        H_ = (H + 2 * padding[0] - pooling[0]) // strides[0] + 1
        W_ = (W + 2 * padding[1] - pooling[1]) // strides[1] + 1
    
        pool_X = np.zeros((N, C, H_, W_))
        
        for n in np.arange(N):
            for c in np.arange(C):
                for i in np.arange(H_):
                    for j in np.arange(W_):
                        ##參考公式中i*s< <i*s+k
                        pool_X[n, c, i, j] = np.max(padding_X[n, c, strides[0] * i:strides[0] * i + pooling[0], strides[1] * j:strides[1] * j + pooling[1]])
        return pool_X ##輸出可以加一個assert

    def maxpooling_backward(self, next_dX, X, pooling, strides=(2, 2), padding=(0, 0)):
        """
        最大池化反向過程
        :param next_dX: 損失函數關於最大池化输出的損失
        :param X: 卷積層矩陣,形状(N,C,H,W)，N為batch_size，C為通道數
        :param pooling: 池化大小(k1,k2)
        :param strides: 步长
        :param padding: 0填充
        :return:
        """
        N, C, H, W = X.shape
        _, _, H_, W_ = next_dX.shape
        # 零填充
        padding_X = np.lib.pad(X, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant',
                               constant_values=0)
    
        # 零填充後的梯度
        padding_dX = np.zeros_like(padding_X)
    
        for n in np.arange(N):
            for c in np.arange(C):
                for i in np.arange(H_):
                    for j in np.arange(W_):
                        # 找到最大值的那個元素坐標，將梯度傳给這個坐標
                        # 參考公式s1*i+k1和s2*j+k2
                        flat_idx = np.argmax(padding_X[n, c, strides[0] * i:strides[0] * i + pooling[0], strides[1] * j:strides[1] * j + pooling[1]])
    
                        h_idx = strides[0] * i + flat_idx // pooling[1]
                        w_idx = strides[1] * j + flat_idx % pooling[1]
                        padding_dX[n, c, h_idx, w_idx] += next_dX[n, c, i, j]
        # 返回時剔除零填充
        return self.Zeros_remove(padding_dX, padding)
    
    
    def flatten_forward(self, X): #將多維數组展平，前向傳播
        """
        :param X: 多維數组,形狀(N,d1,d2,..)
        :return:
        """
        N = X.shape[0]
        return np.reshape(X, (N, -1))
    
    def flatten_backward(self, next_dX, X): #打平層反向傳播
        """
        :param next_dX:
        :param X:
        :return:
        """
        return np.reshape(next_dX, X.shape)
    
    def relu_forward(self, X):
        """
        relu前向傳播
        :param X: 待激活層
        :return: 激活後的结果
        """
        return np.maximum(0, X)

    def relu_backward(self, next_dX, X):
        """
        relu反向傳播
        :param next_dX: 激活後的梯度
        :param X: 激活前的值
        :return:
        """
        dX = np.where(np.greater(X, 0), next_dX, 0)
        return dX
    
    def softmax_forward(self, y_predict):
        y_shift = y_predict - np.max(y_predict, axis=-1, keepdims=True)
        y_exp = np.exp(y_shift)
        y_probability = y_exp / np.sum(y_exp, axis=-1, keepdims=True)
        return y_probability
    
    def forward_all_flow(self, input_imgs): # input_imgs.shape: (16, 1, 32, 32)
        self.neurons["conv1"] = self.convolution_forward(input_imgs.astype(np.float64), self.weights["K1"], self.weights["b1"]) # (16, 3, 28, 28)
        self.neurons["conv1_relu"] = self.relu_forward(self.neurons["conv1"]) # (16, 3, 28, 28)
        self.neurons["maxp1"] = self.maxpooling_forward(self.neurons["conv1_relu"].astype(np.float64), pooling=(3,3)) # (16, 3, 13, 13) (不滿三個的不補)
        self.neurons["flatten"] = self.flatten_forward(self.neurons["maxp1"]) # (16, 3*13*13)
        self.neurons["fc2"] = self.fullyconnected_forward(self.neurons["flatten"], self.weights["W2"], self.weights["b2"]) #(16, 64)
        self.neurons["fc2_relu"] = self.relu_forward(self.neurons["fc2"]) #(16, 64)
        self.neurons["y"] = self.fullyconnected_forward(self.neurons["fc2_relu"], self.weights["W3"], self.weights["b3"]) #(16, 3)
        self.neurons["y_prob"] = self.softmax_forward(self.neurons["y"])
        return self.neurons["y_prob"]
    
    def cross_entropy_loss(self, y_probability, y_true):
        """
        :param y_probability: 預測機率,shape (N,d)，N为批量样本数
        :param y_true: 真實值,shape(N,d)
        :return:
        """
        #loss = np.mean(np.sum(-y_true * np.log(y_probability), axis=-1))  # 損失函數
        loss = np.sum(-y_true * np.log(y_probability))  # 損失函數, 直接加總，未取平均
        dy = y_probability - y_true
        return loss, dy
        
    def backprop(self, input_imgs, batch_labels):
        loss, dy = self.cross_entropy_loss(self.neurons["y_prob"], batch_labels)
        self.gradients["W3"], self.gradients["b3"], self.gradients["fc2_relu"] = self.fullyconnected_backward(dy, self.weights["W3"], self.neurons["fc2_relu"])
        self.gradients["fc2"] = self.relu_backward(self.gradients["fc2_relu"], self.neurons["fc2"])
        
        self.gradients["W2"], self.gradients["b2"], self.gradients["flatten"] = self.fullyconnected_backward(self.gradients["fc2"], self.weights["W2"], self.neurons["flatten"])
        self.gradients["maxp1"] = self.flatten_backward(self.gradients["flatten"], self.neurons["maxp1"])
           
        self.gradients["conv1_relu"] = self.maxpooling_backward(self.gradients["maxp1"].astype(np.float64), self.neurons["conv1_relu"].astype(np.float64), pooling=(3,3))
        self.gradients["conv1"] = self.relu_backward(self.gradients["conv1_relu"], self.neurons["conv1"])
        self.gradients["K1"], self.gradients["b1"], _ = self.convolution_backward(self.gradients["conv1"], self.weights["K1"], input_imgs)
        return loss
        
    def update_parameter(self):
        for key in self.weights.keys():
            self.weights[key] = self.weights[key] - self.lr * self.gradients[key]
    
def shuffle_data(train_imgs_all, train_labels_all):
    shuffle_train_index = [i for i in range(train_labels_all.shape[0])]
    shuffle_train_index = random.sample(shuffle_train_index, len(shuffle_train_index))
    
    train_imgs_all = train_imgs_all[shuffle_train_index]
    train_labels_all = train_labels_all[shuffle_train_index]
    return train_imgs_all, train_labels_all
    
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-l', "--lr", type=float, help="learning rate", default=0.05)
    parser.add_argument("-e", "--epoch", type=int, help="total training epoch", default=30)
    parser.add_argument("-b", "--batch_size", type=int, help="batch size for training & testing", default = 16)
    args = parser.parse_args()
    print(args)
    
    train_Carambula_imgs, train_Carambula_labels = load_data_and_norm('./Data_train/Carambula', label_num = 0)
    train_Lychee_imgs, train_Lychee_labels = load_data_and_norm('./Data_train/Lychee', label_num = 1)
    train_Pear_imgs, train_Pear_labels = load_data_and_norm('./Data_train/Pear', label_num = 2)
    train_imgs_all = np.concatenate([train_Carambula_imgs, train_Lychee_imgs, train_Pear_imgs], axis = 0)
    train_labels_all = np.concatenate([train_Carambula_labels, train_Lychee_labels, train_Pear_labels], axis = 0)
    
    train_imgs_all, train_labels_all = shuffle_data(train_imgs_all, train_labels_all)
    
    # split train:val == 7:3
    val_indices = random.sample(range(train_labels_all.shape[0]), int(train_labels_all.shape[0]*0.3)) # get 1470*0.3 = 441 random indices
    val_imgs_all = train_imgs_all[val_indices]
    val_labels_all = train_labels_all[val_indices]
    
    train_imgs_all = np.delete(train_imgs_all, val_indices, axis=0)
    train_labels_all = np.delete(train_labels_all, val_indices, axis=0)
    
    test_Carambula_imgs, test_Carambula_labels = load_data_and_norm('./Data_test/Carambula', label_num = 0)
    test_Lychee_imgs, test_Lychee_labels = load_data_and_norm('./Data_test/Lychee', label_num = 1)
    test_Pear_imgs, test_Pear_labels = load_data_and_norm('./Data_test/Pear', label_num = 2)
    test_imgs_all = np.concatenate([test_Carambula_imgs, test_Lychee_imgs, test_Pear_imgs], axis = 0)
    test_labels_all = np.concatenate([test_Carambula_labels, test_Lychee_labels, test_Pear_labels], axis = 0)
    print('model init...')
    model = CNN(kernel_size=5, kernel_depth=3, input_img_w_h=train_imgs_all.shape[2], fc_units=64, batch_size=args.batch_size, lr=args.lr, weights={})
    
    train_avg_loss_list, val_avg_loss_list = [], []
    best_val_loss = 10000.
    select_epoch = -1
    best_weight = {}
    print('start training...')
    for e in range(0, args.epoch, 1):
        train_loss_sum = 0
        val_loss_sum = 0
        train_imgs_all, train_labels_all = shuffle_data(train_imgs_all, train_labels_all) # random shuffle of training data each epoch
        # training
        for batch_index_s in range(0, train_imgs_all.shape[0], args.batch_size):
            if (batch_index_s + args.batch_size) < train_imgs_all.shape[0]:
                batch_index_e = batch_index_s + args.batch_size
            else:
                batch_index_e = train_imgs_all.shape[0]
            batch_imgs = train_imgs_all[batch_index_s:batch_index_e]
            batch_labels = train_labels_all[batch_index_s:batch_index_e]
            y_probs = model.forward_all_flow(batch_imgs[:, np.newaxis, :, :])
            train_loss = model.backprop(batch_imgs[:, np.newaxis, :, :], batch_labels)
            train_loss_sum += train_loss
            model.update_parameter()
        train_avg_loss_list.append(train_loss_sum/train_imgs_all.shape[0])
        
        # validation
        gt_total, pred_total = [], []
        for batch_index_s in range(0, val_imgs_all.shape[0], args.batch_size):
            if (batch_index_s + args.batch_size) < val_imgs_all.shape[0]:
                batch_index_e = batch_index_s + args.batch_size
            else:
                batch_index_e = val_imgs_all.shape[0]
            batch_imgs = val_imgs_all[batch_index_s:batch_index_e]
            batch_labels = val_labels_all[batch_index_s:batch_index_e]
            y_probs = model.forward_all_flow(batch_imgs[:, np.newaxis, :, :])
            val_loss, _ = model.cross_entropy_loss(y_probs, batch_labels)
            val_loss_sum += val_loss
            
            batch_gt = np.argmax(batch_labels, axis=1)
            batch_pred = np.argmax(y_probs, axis=1)
            gt_total += batch_gt.tolist()
            pred_total += batch_pred.tolist()
        # calculate val acc
        correct_cnt = 0
        for i in range(0, len(gt_total), 1):
            if gt_total[i] == pred_total[i]:
                    correct_cnt += 1
        val_avg_loss_list.append(val_loss_sum/val_imgs_all.shape[0])
        if val_avg_loss_list[-1] < best_val_loss:
            best_val_loss = val_avg_loss_list[-1]
            select_epoch = e + 1
            for k in model.weights.keys():
                best_weight[k] = model.weights[k]
                
        print('EPOCH:', e+1, ', train_loss:', round(train_avg_loss_list[-1], 4), ', val_loss:', round(val_avg_loss_list[-1], 4), ', val_acc:', round(100 * correct_cnt/len(gt_total), 2), '%')
    
    # testing
    model = CNN(kernel_size=5, kernel_depth=3, input_img_w_h=train_imgs_all.shape[2], fc_units=64, batch_size=args.batch_size, lr=args.lr, weights={})
    for k in best_weight.keys(): # load the best weight
        model.weights[k] = best_weight[k]
    
    gt_total, pred_total = [], []
    for batch_index_s in range(0, test_imgs_all.shape[0], args.batch_size):
        if (batch_index_s + args.batch_size) < test_imgs_all.shape[0]:
            batch_index_e = batch_index_s + args.batch_size
        else:
            batch_index_e = test_imgs_all.shape[0]
        batch_imgs = test_imgs_all[batch_index_s:batch_index_e]
        batch_labels = test_labels_all[batch_index_s:batch_index_e]
        y_probs = model.forward_all_flow(batch_imgs[:, np.newaxis, :, :])
        batch_gt = np.argmax(batch_labels, axis=1)
        batch_pred = np.argmax(y_probs, axis=1)
        gt_total += batch_gt.tolist()
        pred_total += batch_pred.tolist()
    # calculate testing acc
    correct_cnt = 0
    for i in range(0, len(gt_total), 1):
        if gt_total[i] == pred_total[i]:
                correct_cnt += 1
    print('select epoch:', select_epoch)
    print('Testing accuracy:', round(100 * correct_cnt/len(gt_total), 2), '%')
    plt.figure()
    plt.plot(np.arange(args.epoch), train_avg_loss_list, color = 'r', label="training loss")
    plt.plot(np.arange(args.epoch), val_avg_loss_list, color = 'b', label="validation loss")
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('train_val_loss.png')