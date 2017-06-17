# Neural Network Training: Testing
# #
# Josh Gordon {ML} Recipes
#   -> https://youtu.be/cKxRvEZd3Mw?list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal
#
# Developed by Nathan Shepherd

#from sklearn import tree
import pandas as pd
import codecs
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
import numpy as np
from random import randint

class Incident():
    def __init__(self, close_notes, sho_des, description):        
        self.sho_des = sho_des
        self.close_notes = close_notes
        self.description = description

''' ################################################################ '''
'''                    Get raw training data                         '''

# Get .csv files from working dir and store in 'panda dataframe object'
file = codecs.open('SD_wHardware_TrainingData.csv', "r",encoding='utf-8', errors='ignore')
sd_wHardware = pd.read_csv(file)
sd_wHardware = sd_wHardware.set_index('number')

file = codecs.open('SD_SoftApp_TrainingData.csv', "r",encoding='utf-8', errors='ignore')
sd_softApp = pd.read_csv(file)
sd_softApp = sd_softApp.set_index('number')

#Prediction Data
file = codecs.open('SD_wHardware_PredictionData.csv', "r",encoding='utf-8', errors='ignore')
sd_wHardware_predict = pd.read_csv(file)
sd_wHardware_predict = sd_wHardware_predict.set_index('number')
file.close()

# 0 -> close_notes
# 1 -> short_description
# 2 -> description
corpera = {'Workstation Hardware':{0:[], 1:[], 2:[]},
           'Software Application':{0:[], 1:[], 2:[]},
           'wHardware Prediction':{0:[], 1:[], 2:[]}}

# Step one: Process text and append to corpera
def process(dataframe, field, corpera_key):
    dataframe = dataframe[field].tolist()
    raw = []
    outs = []
    for i in range(0, len(dataframe)):
        raw.append(sent_tokenize(str(dataframe[i])))

    #FIXME: recover /keep sentence structure
    for sent in raw:
        clean = re.sub("[^a-zA-Z0-9]", " ", str(sent))
        words = word_tokenize(clean)
        for word in words:
            if word in set(stopwords.words("english")):
                words.remove(word)
        outs.append(words)

    #outs contains matrix -> outs [ incidents[ words[] ] ]
    if field == 'close_notes':
        corpera[corpera_key][0] = outs
    if field == 'short_description':
        corpera[corpera_key][1] = outs
    if field == 'description':
        corpera[corpera_key][2] = outs

print('Initializing corpera matrix w/ Workstation Hardware lexicons ...')
process(sd_wHardware, 'close_notes', 'Workstation Hardware')
process(sd_wHardware, 'short_description', 'Workstation Hardware')
process(sd_wHardware, 'description', 'Workstation Hardware')
print('Initialization successful.\n')

print('Initializing corpera matrix w/ Software Application lexicons ...')
process(sd_softApp, 'close_notes', 'Software Application')
process(sd_softApp, 'short_description', 'Software Application')
process(sd_softApp, 'description', 'Software Application')
print('Initialization successful.\n')

print('Initializing corpera matrix w/ Prediction Data ...')
process(sd_wHardware_predict, 'close_notes', 'wHardware Prediction')
process(sd_wHardware_predict, 'short_description', 'wHardware Prediction')
process(sd_wHardware_predict, 'description', 'wHardware Prediction')
print('Initialization successful.\n')

for field in corpera:
    print('\t', field)
    for key in field:
        print(key)

# Step Two: Convert text to numerical representations. TODO: word2vec, GloVe
dictionary = {}
for field in corpera:
    if field != 'wHardware Prediction':
        for key in corpera[field]:
            for sent in corpera[field][key]:
                for word in sent:
                    if word not in dictionary:
                        dictionary[word] = len(dictionary)
print('\nThere are', len(dictionary), ' unique lexicons in this corpera.\n')
print('Press enter to convert lexicons into integer representations.')
wait_for = input()

lexical_ints = {'Workstation Hardware':{0:[], 1:[], 2:[]},
                'Software Application':{0:[], 1:[], 2:[]},
                'wHardware Prediction':{0:[], 1:[], 2:[]}}

for field in corpera:
    for key in corpera[field]:
        for sent in corpera[field][key]:
            for word in sent:
                if word in dictionary:
                    #loss of sentence structure here
                    lexical_ints[field][key].append(dictionary[word])
                else:
                    lexical_ints[field][key].append(randint(0, len(dictionary)))
print('Workstation Hardware', lexical_ints['Workstation Hardware'][0][0:10])
print('Software Application', lexical_ints['Software Application'][0][0:10], '\n')
''' ################################################################ '''
'''                lexical_ints Matrix Decomposition            '''

#Features: close_notes, short_description, description
#Labels: 'Workstation Hardware', 'Software Application'

for i in range(0, 2):
    print(len(lexical_ints['Workstation Hardware'][i]))

'''
#recreate incident structure
work_hard_incs = []
for incident in range(0, len(lexical_ints['Workstation Hardware'][0])):
    work_hard_incs.append(np.array([lexical_ints['Workstation Hardware'][0][incident],
                                    lexical_ints['Workstation Hardware'][1][incident],
                                    lexical_ints['Workstation Hardware'][2][incident]]))
soft_app_incs = []
for incident in range(0, len(lexical_ints['Software Application'][0])):
    soft_app_incs.append(np.array([lexical_ints['Software Application'][0][incident],
                                    lexical_ints['Software Application'][1][incident],
                                    lexical_ints['Software Application'][2][incident]]))
wHar_predict_incs = []
for incident in range(0, len(lexical_ints['wHardware Prediction'])):
    wHar_predict_incs.append(np.array([lexical_ints['wHardware Prediction'][0][incident],
                                    lexical_ints['wHardware Prediction'][1][incident],
                                    lexical_ints['wHardware Prediction'][2][incident]]))

features = []
labels = []
prediction_data = []
for i in range(len(work_hard_incs)):
    features.append(work_hard_incs[i])
    labels.append(0)

for i in range(len(soft_app_incs)):
    features.append(soft_app_incs[i])
    labels.append(1)

for i in range(len(wHar_predict_incs)):
    features.append(wHar_predict_incs[i])
'''

''' ################################################################ '''
'''                      compose neural network                      '''
# train classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

#make prediction
print(clf.predict(prediction_data))












