## Training Data Preprocessing
#
# Developed by Nathan Shepherd

import sklearn
import pandas as pd
import codecs
from nltk import *
from nltk.corpus import  stopwords
import re

''' ################################################################ '''
'''                    Get raw training data                         '''

##print('\tService Detail: Printing')
###########
file = codecs.open('SD_Printing_TrainingData.csv', "r",encoding='utf-8', errors='ignore')
sd_printing = pd.read_csv(file)

sd_printing = sd_printing.set_index('number')

##print(sd_printing.head(), '\n')
##print(sd_printing.info())
file.close()
######

##print('\tService Detail: Software Application')
###########
file = codecs.open('SD_SoftApp_TrainingData.csv', "r",encoding='utf-8', errors='ignore')
sd_softApp = pd.read_csv(file)

sd_softApp = sd_softApp.set_index('number')

##print(sd_softApp.head(), '\n')
##print(sd_softApp.info())
file.close()
######

##print('\tService Detail: Classrom / Event Support')
###########
file = codecs.open('SD_ClassrmSupprt_TrainingData.csv', "r",encoding='utf-8', errors='ignore')
sd_classrmSupprt = pd.read_csv(file)

sd_classrmSupprt = sd_classrmSupprt.set_index('number')

##print(sd_classrmSupprt.head(), '\n')
##print(sd_classrmSupprt.info())
file.close()
######

#print('\tService Detail: Workstation Hardware')
###########
file = codecs.open('SD_wHardware_TrainingData.csv', "r",encoding='utf-8', errors='ignore')
sd_wHardware = pd.read_csv(file)

sd_wHardware = sd_wHardware.set_index('number')

##print(sd_wHardware.head(), '\n')
##print(sd_wHardware.info())
file.close()
######

''' ################################################################ '''
'''                    Initialize Corpera

sd_wHardware
sd_classrmSupprt
sd_softApp
sd_printing

close_notes
short_description
description
'''

# 0 -> close_notes
# 1 -> short_description
# 2 -> description
corpera = {'Workstation Hardware':{0:[], 1:[], 2:[]},
           'Classroom / Event Support':{0:[], 1:[], 2:[]},
           'Software Application':{0:[], 1:[], 2:[]},
           'Printing':{0:[], 1:[], 2:[]}}

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

print('Initializing corpera matrix w/ Classroom / Event Support lexicons ...')
process(sd_classrmSupprt, 'close_notes', 'Classroom / Event Support')
process(sd_classrmSupprt, 'short_description', 'Classroom / Event Support')
process(sd_classrmSupprt, 'description', 'Classroom / Event Support')
print('Initialization successful.\n')

''' First Incident in corpera matrix
print(corpera['Classroom / Event Support'][0][0])
print(corpera['Classroom / Event Support'][1][0])
print(corpera['Classroom / Event Support'][2][0])
'''

print('Initializing corpera matrix w/ Software Application lexicons ...')
process(sd_softApp, 'close_notes', 'Software Application')
process(sd_softApp, 'short_description', 'Software Application')
process(sd_softApp, 'description', 'Software Application')
print('Initialization successful.\n')

print('Initializing corpera matrix w/ Printing lexicons ...')
process(sd_printing, 'close_notes', 'Printing')
process(sd_printing, 'short_description', 'Printing')
process(sd_printing, 'description', 'Printing')
print('Initialization successful.\n')

'''
corpera = {'Workstation Hardware':{0:[], 1:[], 2:[]},
           'Classroom / Event Support':{0:[], 1:[], 2:[]},
           'Software Application':{0:[], 1:[], 2:[]},
           'Printing':{0:[], 1:[], 2:[]}}
'''
# Step Two: Convert text to numerical representations TODO: word2vec, GloVe
dictionary = {}
for field in corpera:
    for key in corpera[field]:
        for sent in corpera[field][key]:
            for word in sent:
                if word not in dictionary:
                    dictionary[word] = len(dictionary)
print('\nThere are', len(dictionary), ' unique lexicons in this corpera.\n')
print('Press enter to convert lexicons into integer representations.')
wait_for = input()

# Lexical_ints dictionary contains numerical representations of all words in corpera
lexical_ints = {'Workstation Hardware':{0:[], 1:[], 2:[]},
                'Classroom / Event Support':{0:[], 1:[], 2:[]},
                'Software Application':{0:[], 1:[], 2:[]},
                'Printing':{0:[], 1:[], 2:[]}}

for field in corpera:
    for key in corpera[field]:
        for sent in corpera[field][key]:
            for word in sent:
                #loss of sentence structure here
                lexical_ints[field][key].append(dictionary[word])

print('Workstation Hardware', lexical_ints['Workstation Hardware'][0][0:10])
print('Classroom / Event Support', lexical_ints['Classroom / Event Support'][0][0:10])
print('Software Application', lexical_ints['Software Application'][0][0:10])
print('Printing', lexical_ints['Printing'][0][0:10], '\n')

''' ################################################################ '''
'''                      compose neural network                      '''






























