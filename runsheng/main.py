'''
Created on Apr 29, 2015

@author: rsong_admin
'''

import regression as ann
import pca
import numpy as np


def train_or_use(descs_name,target_name=None,network_name=None,train=True):
    if train:
        '''
        This train a new network
        '''
        CED=ann.training(descs_name,target_name)
        CED.set_dataset(0.15, featureNorm = True)
        
        CED.train_net(training_times_input=120, num_neroun=100,
                      learning_rate_input=0.001, weight_decay=0.1, momentum_in=0,
                      verbose_input = True)
        
        prediction_tst = CED.predict(CED.network, CED.tstdata['input'])
        prediction_tst = np.exp(prediction_tst)
        
        CED.plot_diff(np.exp(CED.tstdata['target']), prediction_tst)
        print CED.calc_Diff(np.exp(CED.tstdata['target']), prediction_tst)
        raw_input("Save?")
        CED.save_network(network_name)
    else:
        '''
        This load the exsisting network and run prediction on it
        '''
        CED=ann.predicting(network_name = network_name,
                           predict_descs= descs_name,
                           training_descs='./data/183_descs.csv',
                           featureNorm=True)
        descs_normed = CED.featureNorm(CED.descsForPred)
        
        prediction_results=CED.predict(CED.network,descs_normed);
        Y=np.loadtxt(target_name,delimiter=',')
        prediction = np.exp(prediction_results)
        dist = CED.calcDist(CED.descsForPred, CED.TrainingData)
        np.savetxt('dist.csv',dist,delimiter = ',')
        CED.plot_diff(Y, prediction)
        
        save_name=target_name.replace('.csv','')
        save_name=save_name+'_prediction.csv'
        np.savetxt(save_name, prediction_results, delimiter=',')
  
if __name__ == '__main__':
    
    train_or_use(descs_name='./data./183_descs.csv', 
                 target_name='./data./CED_Y_183.csv', 
                 network_name='./net/CED_June26.xml', 
                 train=False)
    
    
    
    
    
    
    
    

    
   