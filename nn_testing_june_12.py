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
#from nltk.corpus import stopwords
import re
import numpy as np
from random import randint
import tensorflow

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
#stop words do not contribute to overal symantic interpretation
#err_context will be used to remove all newline characters that
#   are erroneously added to front of words
stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the", "n"]
err_context = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
for char in range(0, len(err_context)):
    err_context[char] = 'n' + err_context[char].upper()

#preprocessing to reduce computational complexity
def process(dataframe, field, corpera_key):
    dataframe = dataframe[field].tolist()
    raw = []
    outs = []
    for i in range(0, len(dataframe)):
        raw.append(sent_tokenize(str(dataframe[i])))

    #FIXME: recover /keep sentence structure
    for sent in raw:
        for index in range(0, len(sent) - 1):
            #remove some of new line characters
            if sent[index]=='\\' and sent[index+1]=='n':
                sent = sent[0:index] + sent[index + 2]

        #find and replace all non letters and integers to blank space
        clean = re.sub("[^a-zA-Z0-9]", " ", str(sent))
        words = word_tokenize(clean.lower())#method .lower() decreases dictionary size by 15%

        for word in words:
            if word in stopwords:
                words.remove(word)
        
        #remove part of newline character from front of words if it appears
        for word in words:
            if word[:2] in err_context:
                word_index = words.index(word)
                words[word_index] = word[1:]
                
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

print('Press enter to create incident objects.')
wait_for=input()

class Incident():
    lexicon = {} #Array representing input (unique lexicons in dictionary) - ref ln 319
    def __init__(self, close_notes, sho_des, description, service_detail):        
        self.sho_des = sho_des
        self.close_notes = close_notes
        self.description = description
        self.service_detail = service_detail
# 0 -> close_notes
# 1 -> short_description
# 2 -> description
all_incidents = []
for field in corpera:
    for inc in range(0, len(corpera[field][0])):
        all_incidents.append(Incident(corpera[field][0][inc],
                                      corpera[field][1][inc],
                                      corpera[field][2][inc],
                                      field))

print('Succesfully created',len(all_incidents),'Incident objects with these fields:')
print('\n   ---->   Short description, Description, Close notes')

''' # for quality checking
for field in corpera:
    for inc in range(0, 1):
        print('\n\t\t', field)
        print('\t\nclose_notes\n\n', corpera[field][0][inc])
        print('\t\nshort_description\n\n', corpera[field][1][inc])
        print('\t\ndescription\n\n', corpera[field][2][inc], '\n---------------')
   '''     
        
            

# Step Two: Convert text to numerical representations. TODO: word2vec, GloVe
clock_speed = 42
print('\nThis next part is computationally expensive. Please wait ...')

dictionary = {}
lexio2 = [] #matrix associating word tags to their occurance in corpera
count = 0
tag = 0
for field in corpera:
    if field != 'wHardware Prediction':
        for key in corpera[field]:
            for sent in corpera[field][key]:
                for word in sent:
                    if word not in dictionary.keys():
                        lexio2.append([tag, 0])
                        dictionary[word] = tag
                        tag += 1
'''
for field in corpera:
    if field != 'wHardware Prediction':
        for key in corpera[field]:
            for sent in corpera[field][key]:
                for word in sent:
                    if word in dictionary.keys():
                        count += 1
                        if count % 2000 == 0:
                            print('counted:',count//(700),'%')
                        # integer represents a word
                        for pair in lexio2:
                            if pair[0] == dictionary[word]:
                                pair[1] += 1
'''
# memory locations of left and right are mirrored symetrically
left = [] #word ID
right = [] #occurance of word
for tag in lexio2:
    left.append(tag[0])
    right.append(tag[1])
left = np.array(left)
right = np.array(right)
print('\nThe occurance of first 15 words in lexio2 matrix:\n')
print('\t',left[:10])
print('\t      - ::::::: -')
print('\t',right[:10],'\n')
print('Identifying principle component and decreasing dimensionality, please wait ...\n')

upper_bound = 40000
lower_bound = 6500
for word in lexio2:
    if upper_bound> word[1] > lower_bound:
        pass #TODO: identify the principle component
    else:
        lexio2.remove(word)

freq = [] #populate with str(word) : int(word_occurrence)
top_few = []
matched = 0
time_step = [i for i in range(0, len(lexio2), 900)]
for pair in lexio2:
    for word in dictionary:
        if pair[0] == dictionary[word]:
            freq.append([word, pair[1]])
            matched += 1
            if len(freq) in time_step:
                print('Matched:',matched,'words from lexio plane to new dictionary')
            

        
print('\nMost frequent words were:')
maximum = 90
for pair in freq:
    if pair[1] > maximum:
        maximum = pair[1]
        print(pair[0], pair[1], '\n')

print('There are',len(lexio2),'words that occured between',upper_bound,'and',lower_bound,'times')
print('\nThere are', len(dictionary), ' unique lexicons in this corpera.')
print('This information will be represented with',len(freq),'|',len(lexio2),'words.\n')
print('Press enter to convert lexicons into integer representations.')
wait_for = input()

# word to int
missed_words = 0
for inc in all_incidents:
    for word in inc.sho_des:
        word_index = inc.sho_des.index(word)
        if word in dictionary.keys():
            inc.sho_des[word_index] = dictionary[word]
        else:
            missed_words += 1
            inc.sho_des[word_index] = randint(0, len(dictionary))
            dictionary[word] = inc.sho_des[word_index]
            
    for word in inc.close_notes:
        word_index = inc.close_notes.index(word)
        if word in dictionary.keys():
            inc.close_notes[word_index] = dictionary[word]
        else:
            missed_words += 1
            inc.close_notes[word_index] = randint(0, len(dictionary))
            dictionary[word] = inc.close_notes[word_index]
            
    for word in inc.description:
        word_index = inc.description.index(word)
        if word in dictionary.keys():
            inc.description[word_index] = dictionary[word]
        else:
            missed_words += 1
            inc.description[word_index] = randint(0, len(dictionary))
            dictionary[word] = inc.description[word_index]

for inc in all_incidents:
    if inc.service_detail == 'Workstation Hardware':
        inc.service_detail = 1
    if inc.service_detail == 'Software Application':
        inc.service_detail = 0
    if inc.service_detail == 'wHardware Prediction':
        inc.service_detail = 1
            
print('Converted text into integer representations from',len(all_incidents),'incidents')
print('Randomly assigned', (((missed_words/len(dictionary)) * 100**2)//100),'% of words.')

print('\n\t --- Preprocessing Complete ---\n','\nPress enter to begin training model.')
wait_for = input()


''' ################################################################ '''
'''                      compose neural network                      '''

#prototype from Sentdex
# #
# How the network will run - Sentdex
#   - > https://pythonprogramming.net/train-test-tensorflow-deep-learning-tutorial/?completed=/preprocessing-tensorflow-deep-learning-tutorial/
# ------
# Developed by Nathan Shepherd


#How to use TensorFlow
'''
inputs > weights > hidden layer 1 (activation function)
       > weights > hidden layer 2 (activation function)
       > weights > output layer

compare output to intended output > cost function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer,
                                                       ...      ,
                                                   Stochastic Gradient Descent,
                                                   AdaGrad)
backpropagation
feed forward + backprop = epoch
'''
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

#10 classes, 0-9
'''
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
        ...
'''

#hidden layer nodes composition
#TODO: increase complexity
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)

    #difference between prediction and label
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    # AKA: stochastic gradient descent
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #epoch is one iteration of feed forward and backpropagation
    num_epochs = 10

    with tf.Session() as sess:
        # initialialize neural network with placeholders
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            epoch_loss = 0

            #method next_batch comes from MNIST directories
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',num_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        #reduce dimensionality of tensors into float objects
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

def initialize_placeholders(x_length):
    #close_notes, sho_des, description
    #lexicon is array representing input (unique lexicons in dictionary)
    #lexicon == {12:1231, 12312:123123, ...}

    time_step = [i for i in range(len(dictionary)) if i % (len(all_incidents)//10) == 0]
    checkpoint = [i for i in range(0, len(dictionary), 250)]
    #42 seconds a word on AMD 4.2 GHZ, 8 Core CPU
    clock_speed, count = 42, 0
    print('This next part is computationally expensive.')
    print('The tensors representing information in each incident are being initialized.')
    print('Representing each incident as a',len(lexio2),'dimensional vector.','\n\nPlease wait ~1.4 minutes.\n')
    for inc in all_incidents:
        count += 1
        if count in time_step:
            print('Initialized placeholders for',count,'of',len(all_incidents),'incidents')
        if count in checkpoint:
            print('\n\t', (count // 10), '% complete\n')
        for i in range(0, x_length):
            inc.lexicon[i] = 0    
            for word in inc.sho_des:
                if word in dictionary.keys():
                    inc.lexicon[i] += 1
            for word in inc.close_notes:
                if word in dictionary.keys():
                    inc.lexicon[i] += 1
            for word in inc.description:
                if word in dictionary.keys():
                    inc.lexicon[i] += 1
    #print(np.array(all_incidents[0].lexicon))
            


features = []
labels = []
prediction_data = []


x_shape = len(dictionary)

initialize_placeholders(x_shape)

train_neural_network(x)














