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
    def __init__(self, kernel_size, kernel_num, input_img_w_h, batch_size, lr):
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.input_img_w_h = input_img_w_h
        self.batch_size = batch_size
        self.lr = lr
        # weight init
        self.k_w = np.random.normal(0, 1, (kernel_size * kernel_size + 1, kernel_num)) * 0.01 # kernel & bias, (10, 20)
        self.h1_w = np.random.normal(0, 1, ((self.input_img_w_h - kernel_size + 1)*(self.input_img_w_h - kernel_size + 1)*(kernel_num) + 1, 128)) * 0.01 # no padding, stride=1, with bias, (30*30*20+1, 128)
        self.h2_w = np.random.normal(0, 1, (128 + 1, 3)) * 0.01 # DNN output layer #(129, 3)
        
    def relu(self, x, derive=False):
        if derive:
            return 1. * (x > 0)
        return x * (x > 0)

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x, axis=1)[:, np.newaxis])
        #e_x = np.exp(x)
        return e_x / e_x.sum(axis=1)[:, np.newaxis]
    
    def forward(self, input_imgs):
        # transform matrix C, 紀錄kernel的每一個element會乘到的img pixel value
        self.C = np.ones((self.batch_size, self.input_img_w_h - self.kernel_size + 1, self.input_img_w_h -  self.kernel_size + 1,  self.kernel_size *  self.kernel_size)) #(16, 30, 30, 9)
        for m in range(self.kernel_size):
            for n in range(self.kernel_size):
                self.C[:,:,:,m * self.kernel_size + n] = input_imgs[:,m:m+(self.input_img_w_h - self.kernel_size + 1),n:n+(self.input_img_w_h - self.kernel_size + 1)]
        self.C = np.concatenate((self.C, np.ones([self.batch_size, self.input_img_w_h - self.kernel_size + 1, self.input_img_w_h - self.kernel_size + 1, 1])), axis=3) #(16, 30, 30, 10), for bias
        
        # 2D conv layer -> ReLU layer
        self.h_1 = self.relu(np.dot(self.C, self.k_w)) # (16, 30, 30, 10)*(10, 20) => (16, 30, 30, 20)
        
        #FC layer
        #self.h_1 = self.h_1.transpose(1, 2, 3, 0) #(30, 30, 20, 16)
        self.h_1 = np.reshape(self.h_1, [self.batch_size, (self.input_img_w_h - self.kernel_size + 1)*(self.input_img_w_h - self.kernel_size + 1)*(self.kernel_num)])[:, :, np.newaxis] #(16, 30*30*20, 1)
        self.h_1 = np.concatenate((self.h_1, np.ones((self.batch_size, 1, 1))), axis=1) # (16, 30*30*20+1, 1)
        self.h_2 = self.relu(np.dot(self.h_1.transpose(0, 2, 1), self.h1_w)) # (16, 1, 128)
        self.h_2 = np.concatenate((self.h_2, np.ones((self.batch_size, 1, 1))), axis=2) # (16, 1, 129)
        out = self.softmax(np.dot(self.h_2, self.h2_w)) # (16, 3)
        return out.squeeze()
    
    def cal_CE_loss(self, batch_labels, batch_predicts):
        return (-1/batch_labels.shape[0]) * (np.sum(batch_labels * np.log(batch_predicts)))
        
    def backprop(self, batch_labels, batch_predicts):
        # with err
        '''
        delta_1 = (-1.0) * (batch_labels - batch_predicts) # (16, 3)
        delta_2 = np.ones_like(delta_1)
        
        # update the second fully connected hidden layer
        
        delta_h2_w = np.dot(self.h_2.T, delta_1 * delta_2).squeeze() # (129, 3)
        # update the first fully connected hidden layer
        delta_3 = self.relu(self.h_2, derive=True) # (16, 1, 129)
        pdb.set_trace()
        delta_h1_w = np.dot(self.h_1, np.sum(np.dot(delta_3.transpose(1, 2, 0), delta_1 * delta_2) * self.h2_w, axis=1)[:-1].reshape([self.batch_size, -1])) # (30*30*20+1, 128)
        '''
        
def shuffle_data(train_imgs_all, train_labels_all):
    shuffle_train_index = [i for i in range(train_labels_all.shape[0])]
    shuffle_train_index = random.sample(shuffle_train_index, len(shuffle_train_index))
    
    train_imgs_all = train_imgs_all[shuffle_train_index]
    train_labels_all = train_labels_all[shuffle_train_index]
    return train_imgs_all, train_labels_all
    
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-l', "--lr", type=float, help="learning rate", default=0.001)
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
    
    model = CNN(kernel_size=3, kernel_num=20, input_img_w_h=train_imgs_all.shape[2], batch_size=args.batch_size, lr=args.lr)
    
    for e in range(0, args.epoch, 1):
        print('EPOCH:', e)
        train_imgs_all, train_labels_all = shuffle_data(train_imgs_all, train_labels_all) # random shuffle of training data each epoch
        for batch_index_s in range(0, train_imgs_all.shape[0], args.batch_size):
            if (batch_index_s + args.batch_size) < train_imgs_all.shape[0]:
                batch_index_e = batch_index_s + args.batch_size
            else:
                batch_index_e = train_imgs_all.shape[0]
            batch_imgs = train_imgs_all[batch_index_s:batch_index_e]
            batch_labels = train_labels_all[batch_index_s:batch_index_e]
            
            batch_predicts = model.forward(batch_imgs)
            batch_avg_loss = model.cal_CE_loss(batch_labels, batch_predicts)
            model.backprop(batch_labels, batch_predicts)
            