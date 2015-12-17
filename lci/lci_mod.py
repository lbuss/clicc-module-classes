'''
Created on Apr 29, 2015

@author: rsong_admin
'''

import regression as ann
import pca
import numpy as np

class LCI:
    def __init__(self):
        self.config = {
            'descs_name': './data./183_descs.csv',
            'target_name': './data./CED_Y_183.csv',
            'network_name': './net/CED_June26.xml',
        }

    def train(self,descs_name,target_name,network_name='new_network'):
        CED=ann.training(descs_name,target_name)
        CED.set_dataset(0.15, featureNorm = True)

        CED.train_net(training_times_input=120, num_neroun=100,
                      learning_rate_input=0.001, weight_decay=0.1, momentum_in=0,
                      verbose_input = True)

        prediction_tst = CED.predict(CED.network, CED.tstdata['input'])
        prediction_tst = np.exp(prediction_tst)

        CED.plot_diff(np.exp(CED.tstdata['target']), prediction_tst)
        print CED.calc_Diff(np.exp(CED.tstdata['target']), prediction_tst)
        CED.save_network(network_name)


    def run(self):
        self.config = {
            'descs_name': './data./183_descs.csv',
            'target_name': './data./CED_Y_183.csv',
            'network_name': './net/CED_June26.xml',
        }
        '''
        This load the exsisting network and run prediction on it
        '''
        descs_name=self.config['descs_name']
        # target_name=self.config['target_name']

        CED=ann.predicting(network_name = self.config['network_name'],
                           predict_descs= descs_name,
                           training_descs='./data/183_descs.csv',
                           featureNorm=True)
        descs_normed = CED.featureNorm(CED.descsForPred)
        # import pdb; pdb.set_trace()
        prediction_results=CED.predict(CED.network,descs_normed);
        # Y=np.loadtxt(target_name,delimiter=',')
        # prediction = np.exp(prediction_results)
        # dist = CED.calcDist(CED.descsForPred, CED.TrainingData)
        # np.savetxt('dist.csv',dist,delimiter = ',')

        # CED.plot_diff(Y, prediction)

        save_name=target_name.replace('.csv','')
        save_name=save_name+'_prediction.csv'
        np.savetxt(save_name, prediction_results, delimiter=',')
