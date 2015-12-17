'''
Created on Apr 28, 2015

@author: rsong_admin
'''

import numpy as np
from numpy import linalg as LA, linalg



class pca():
    
    def __init__(self,descriptors_raw,user_k=0,threshold=0.9):
        '''delete the col that has all zero values in it'''
        
        self.raw_input=descriptors_raw
        self.threshold=threshold
        self.dimension=user_k
        self.num_row=descriptors_raw.shape[0]
        self.output_matrix=None
        
        
        self.deleted_zero_input=self._deleteZeros(self.raw_input)
        
        '''do the feature normoralization'''
        self.normed_input=self._featureNorm(self.deleted_zero_input)
        
        
        '''do the calcuation of PCA'''
        self.output_matrix=self._pca(self.normed_input, self.dimension, self.threshold)
    
    
    def get_pca(self):
        return self.output_matrix
    
    
    def _load_matrix(self,fileName):
        #laod the csv file that contain all chemical descriptors
        array = np.loadtxt(fileName, delimiter=',', skiprows=0)
        return array
    
    def _deleteZeros(self,input_matrix):
        #this function delete the col that has all values equal to zeros
        sigma=input_matrix.std(axis=0)
        index=np.where(sigma==0)
        new_matrix=np.delete(input_matrix, index, 1) # delete the colm that has all zeros in 
        return new_matrix
    
    def _featureNorm(self,input_matrix):
        #do the feature normalization after delete the row that has all zeros. 
        #do it by hand
        size=input_matrix.shape
        num_row=size[0]
        num_col=size[1]
        norm_matrix=np.zeros((num_row,num_col))
        
        for col in range(num_col):
            #compute the mean and std for each col
            col_mean=input_matrix[:,col].mean()
            col_std=input_matrix[:,col].std()
            for row in range(num_row):
                norm_matrix[row,col]=(input_matrix[row,col]-col_mean)/col_std
        return norm_matrix
    
    
    def _pca(self,inp_matrix,k=0,threshold=0):
        
        num_row=inp_matrix.shape[0]
        num_col=inp_matrix.shape[1]
        #run the PCA for the input matrix
        sig_ma=np.dot(np.transpose(inp_matrix),inp_matrix)/(num_row)
        
        #calculating the svd function
        U,S,Vh=linalg.svd(sig_ma)
        
        if threshold !=0 and k==0: #if the threshold argument exist 
            fenmu=sum(S)
            for i in range (num_col):
                fenzi=sum(S[:i])
                var_ratain=fenzi/fenmu
                if var_ratain>=threshold:
                    k=i
                    print "The number of variables that retain the required variance is: ",k
                    break
        
        else:
            print "User input: ", k
            
        U_reduce=U[:,:k]
        out_matrix=np.dot(inp_matrix,U_reduce)
        
        return out_matrix
    

