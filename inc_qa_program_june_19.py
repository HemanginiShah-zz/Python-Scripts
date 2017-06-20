# Incident QA Program
# # #
# Developed by Nathan Shepherd

import pandas as pd
import tensorflow as tf
import numpy as np
from nltk import sent_tokenize, word_tokenize
import codecs
import re
import random

'''############'''
'''Preprocess Data'''
#Unverified Data
print('Pulling incident data and appending to corpus ...\n')
file = codecs.open('SD_PredictionData.csv', "r",encoding='utf-8', errors='ignore')
full_report = pd.read_csv(file)
full_report = full_report.set_index('number')
file.close()
num_incs = len(full_report['u_service_detail'].tolist())#4350

corpus = {'index':[i for i in range(0, num_incs)],
           'description':[],
           'short_description':[],
           'close_notes':[],
           'service_detail':[]}

corpus['description'] = full_report['description'].tolist()
corpus['short_description'] = full_report['short_description'].tolist()
corpus['close_notes'] = full_report['close_notes'].tolist()
corpus['service_detail'] = full_report['u_service_detail'].tolist()

stopwords = ["a",'n', "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the", "n"]
def process_corpus(field):
    raw = full_report[field]
    outs = []

    #FIXME: recover /keep sentence structure
    for inc in raw:
        #find and replace all non letters and integers to blank space
        clean = re.sub("[^a-zA-Z]", " ", str(inc))
        inc = word_tokenize(clean.lower())#method .lower() decreases dictionary size by 15%

        for word in inc:
            if word in stopwords:
                inc.remove(word)
                   
        outs.append(inc)
        
    if field == 'u_service_detail':
        corpus['service_detail'] = outs
    else:
        corpus[field] = outs
    
for field in corpus:
    if field not in ('index', 'service_detail'):
        process_corpus(field)

'''############'''
'''Word to int'''
'''
corpus = {'index':[i for i in range(0, num_incs)],
           'description':[],
           'short_description':[],
           'close_notes':[],
           'service_detail':[]}
'''
print('Converting Incident text fields into matrix representation ...\n')
dictionary = {}
word_count = 0
freq_matrix = [] #contains: [[word], [frequency]]
for field in corpus:
    if field not in ['index', 'service_detail']:
        print('Calculating coocurrance of word vectors in', field)
        for inc in corpus[field]:
            for word in inc[:3]:
                if word not in dictionary.keys():
                    dictionary[word] = word_count
                    word_count += 1
                    freq_matrix.append([word_count, 0])
                else:
                    for pair in freq_matrix:
                        if pair[0] == dictionary[word]:
                            pair[1] +=  1
###Remove words that do not occur often or too much from dictionary
upper_bound = 550 #word must occur at least once in every ticket given service_detail
lower_bound = 5 #includes incorrectly spelled words, specific dialect, and names
upper_bound = [pair[0] for pair in freq_matrix if pair[1] > upper_bound]
lower_bound = [pair[0] for pair in freq_matrix if pair[1] < lower_bound]
word_num = []
for word, num in dictionary.items():
    word_num.append([word, num])
for pair in word_num:
    for num in upper_bound:
        if pair[1] == num:
            del dictionary[pair[0]]
            #print('Above upper_bound:', pair[0])
#print(len(upper_bound))

for pair in word_num:
    for num in lower_bound:
        if pair[1] == num:
            del dictionary[pair[0]]
            
x_shape = len(dictionary)

class Incident():
    def __init__(self, close_notes, sho_des, description, service_detail):
        self.lexicon = [float(i-i) for i in range(0, len(dictionary))]
            #Array representing input (unique lexicons in dictionary)
        self.sho_des = sho_des
        self.close_notes = close_notes
        self.description = description
        self.service_detail = service_detail
        if service_detail == 'Departmental System':
            self.label = [1, 0, 0, 0, 0, 0, 0]
        if service_detail == 'Peripherals':
            self.label = [0, 1, 0, 0, 0, 0, 0]
        if service_detail == 'Printing':
            self.label = [0, 0, 1, 0, 0, 0, 0]
        if service_detail == 'Software Application ':#there was a space in testfile
            #print('Assigned label 3')
            self.label = [0, 0, 0, 1, 0, 0, 0]
        if service_detail == 'System / Application Access':
            self.label = [0, 0, 0, 0, 1, 0, 0]
        if service_detail == 'Workstation Hardware':
            self.label = [0, 0, 0, 0, 0, 1, 0]
        if service_detail == 'Workstation OS Configuration':
            self.label = [0, 0, 0, 0, 0, 0, 1]
    '''
    0 = Department System
    1 = Peripherals
    2 = Printing
    3 = Software Application
    4 = System / Application Access
    5 = Workstation Hardware
    6 = Workstation OS Configuration
    '''
    
all_incidents = []
for i in corpus['index']:
    all_incidents.append(Incident(corpus['close_notes'][i],
                                  corpus['short_description'][i],
                                  corpus['description'][i],
                                  corpus['service_detail'][i]))
random.shuffle(all_incidents)

print('Number of Incidents created:', len(all_incidents))
print('Length of lexicon will be:', len(dictionary))

words = []
numbers = []
for word, num in dictionary.items():
    words.append(word)
    numbers.append(num)

count = 0
for inc in all_incidents:
    count += 1
    if count % 435 == 0:
        print('Created feature set for',(count*100)//len(all_incidents),'% of incidents')
    all_words = ""
    all_words = inc.sho_des + inc.description + inc.close_notes
    for i in range(0, len(words)):
        for word in all_words:
            if word == words[i]:
                inc.lexicon[i] += 1
                
for inc in all_incidents[:3]:
    print(inc.label)



#####Features and labels identified, contained in [(inc.label & inc.lexicon) for inc in all_incidents]

#TODO: Feed features and labels into machine learning algorithm
import tensorflow as tf

features = []
labels = []
for inc in all_incidents[:4000]:
    features.append(inc.lexicon)
    labels.append(inc.label)
    
test_incs = all_incidents[4000:] #len = 350
test_x = []
test_y = []
for inc in test_incs:
    test_x.append(inc.lexicon)
    test_y.append(inc.label)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 7
batch_size = 100

x = tf.placeholder(tf.float32, [1, x_shape])
y = tf.placeholder(tf.float32, [1, n_classes])

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([x_shape, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}


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
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            for i in range(0, len(features)):
                _, c = sess.run([optimizer, cost], feed_dict={x: features[i], y: labels[i]})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(prediction, y)

        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        print('Accuracy:',accuracy.eval(feed_dict={x:test_x, y:test_y}))

train_neural_network(x)


















