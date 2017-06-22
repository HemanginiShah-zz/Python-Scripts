#scalable support vector machine

from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd
import codecs
from nltk import sent_tokenize, word_tokenize
#from nltk.corpus import stopwords
import re
import numpy as np
from random import randint


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
    if field == 'Software Application':
        for key in corpera[field]:
            for sent in corpera[field][key]:
                for word in sent:
                    if word not in dictionary.keys():
                        lexio2.append([tag, 0])
                        dictionary[word] = tag
                        tag += 1
for field in corpera:
    if field == 'Workstation Hardware':
        for key in corpera[field]:
            for sent in corpera[field][key]:
                for word in sent:
                    if word not in dictionary.keys():
                        lexio2.append([tag, 0])
                        dictionary[word] = tag
                        tag += 1
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

#remove some irrelivant data here
upper_bound = 1000
lower_bound = 30
within_bounds = []
average = 0 #sum(inputs) / num(inputs)
for word in lexio2:
    if upper_bound> word[1] > lower_bound:
        within_bounds.append(word[0])
        pass #TODO: identify the principle component
    
deleted = 0
del_key = []
for key in dictionary:
    if dictionary[key] not in within_bounds:
        del_key.append(key)
for key in del_key:
    if key in dictionary:
        deleted += 1
        del dictionary[key]

print('Identified and deleted',deleted,'words that were outside of frequency bounds')
sum_inputs = sum([pair[1] for pair in lexio2])
num_inputs = len(lexio2)
average = sum_inputs / num_inputs
print('\nThe average word frequency is:',average)

'''
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
for word in freq:
    if ((upper_bound > word[1]) and (word[1] > lower_bound)):
        maximum = word[1]
        #print(word[0], word[1], '\n')
'''
freq_dict = {}
time_step = [i for i in range(0, len(lexio2), 900)]
matched = 0
for pair in lexio2:
    matched += 1
    freq_dict[pair[0]] = pair[1]
    if len(freq_dict) in time_step:
        print('Matched:',matched,'words from lexio plane to frequency dictionary')  

print('\nThere are', len(dictionary), 'usefull lexicons in this corpera.')
print('This information will be represented with',len(dictionary),'words.\n')
print('Press enter to convert lexicons into vector representations.')
wait_for = input()

#identify words exclusively of individual service details
work_hard = []
soft_app = []
for inc in all_incidents:
    if inc.service_detail == 'Workstation Hardware':
        for word in inc.description:
            if word not in work_hard:
                work_hard.append(word)
        for word in inc.sho_des:
            if word not in work_hard:
                work_hard.append(word)
        for word in inc.close_notes:
            if word not in work_hard:
                work_hard.append(word)
    if inc.service_detail == 'Software Application':
        for word in inc.description:
            if word not in soft_app:
                soft_app.append(word)
        for word in inc.sho_des:
            if word not in soft_app:
                soft_app.append(word)
        for word in inc.close_notes:
            if word not in soft_app:
                soft_app.append(word)
#delete all words in dictionary that cooccur in these two service detail
del_words = 0
for word in work_hard:
    if word in soft_app:
        #print(word)
        if word in dictionary:
            del_words += 1
            del dictionary[word]
##for word in soft_app:
##    if word in work_hard:
##        #print(word)
##        if word in dictionary:
##            del_words += 1
##            del dictionary[word]
print('\nDeleted',del_words,'words from cooccurance matrix\n')

# word to frequency vector
missed_words = 0
assigned_words = 0
for inc in all_incidents:
    for word in inc.sho_des:
        word_index = inc.sho_des.index(word)
        if word in dictionary.keys():
            assigned_words += 1
            freq = freq_dict[dictionary[word]]
            inc.sho_des[word_index] = [dictionary[word], freq]
        else:
            missed_words += 1
            freq = average//1
            inc.sho_des[word_index] = [randint(0, len(dictionary)), freq]
            
    for word in inc.close_notes:
        word_index = inc.close_notes.index(word)
        if word in dictionary.keys():
            assigned_words += 1
            freq = freq_dict[dictionary[word]]
            inc.close_notes[word_index] = [dictionary[word], freq]
        else:
            missed_words += 1
            freq = average//1
            inc.close_notes[word_index] = [randint(0, len(dictionary)), freq]
            
    for word in inc.description:
        word_index = inc.description.index(word)
        if word in dictionary.keys():
            assigned_words += 1
            freq = freq_dict[dictionary[word]]
            inc.description[word_index] = [dictionary[word], freq]
        else:
            missed_words += 1
            freq = average//1
            inc.description[word_index] = [randint(0, len(dictionary)), freq]

for inc in all_incidents:
    if inc.service_detail == 'Workstation Hardware':
        inc.service_detail = 1
    if inc.service_detail == 'Software Application':
        inc.service_detail = 0
    if inc.service_detail == 'wHardware Prediction':
        inc.service_detail = -1
            
print('Converted text into integer representations from',len(all_incidents),'incidents')
print('Randomly assigned', ((missed_words//assigned_words) * 100),'% of words.')

print('\n\t --- Preprocessing Complete ---\n','\nPress enter to begin training model.')
wait_for = input()
''' ################################################################ '''
'''                Get Predictions and Visualize Data                '''

x = [[1,7],[2,8],[3,8], [5, 1], [6,-1], [7, 3], [0,1], [2,3], [3,1]] #inputs
y = [1, 1, 1, -1, -1, -1, 0, 0, 0] #labels
dictionary = {1:[1,7],2:[2,8],3:[3,8], 4:[5, 1], 5:[6,-1], 6:[7, 3], 7:[0,1], 8:[2,3], 9:[3,1]}

x_ = [1,2,3,5,7,4,2,3,2,8,3,2,7,8,2,8,2,8]
y_ = [0,7,6,8,9,8,7,4,2,6,8,6,2,3,4,7,8,6]
#plt.scatter(x_, y_, label='scatter', color='k')
'''
for word, occur in freq_dict.items():
    if word % 500 == 0:
        print('Added',word,'of 8000 words to the plot')
    plt.scatter(word, occur, color='k')
'''
train_vectors = []
test_vectors = []
labels = []
for inc in all_incidents: #TODO: Get averages for each incident
    if inc.service_detail == 1:
        num_words = len(inc.description) + len(inc.sho_des) + len(inc.close_notes)
        word_scalar = 0
        freq_scalar = 0
        for word in inc.description:
            word_scalar += word[0]
            freq_scalar += word[1]
        for word in inc.sho_des:
            word_scalar += word[0]
            freq_scalar += word[1]
        for word in inc.close_notes:
            word_scalar += word[0]
            freq_scalar += word[1]
        train_vectors.append([word_scalar/num_words, freq_scalar/num_words])
        labels.append(1)
        plt.scatter(word_scalar/num_words, freq_scalar/num_words, color='k')
    if inc.service_detail == 0:
        num_words = len(inc.description) + len(inc.sho_des) + len(inc.close_notes)
        word_scalar = 0
        freq_scalar = 0
        for word in inc.description:
            word_scalar += word[0]
            freq_scalar += word[1]
        for word in inc.sho_des:
            word_scalar += word[0]
            freq_scalar += word[1]
        for word in inc.close_notes:
            word_scalar += word[0]
            freq_scalar += word[1]
        train_vectors.append([word_scalar/num_words, freq_scalar/num_words])
        labels.append(0)
        plt.scatter(word_scalar/num_words, freq_scalar/num_words, color='b')
    if inc.service_detail == -1:
        num_words = len(inc.description) + len(inc.sho_des) + len(inc.close_notes)
        word_scalar = 0
        freq_scalar = 0
        for word in inc.description:
            word_scalar += word[0]
            freq_scalar += word[1]
        for word in inc.sho_des:
            word_scalar += word[0]
            freq_scalar += word[1]
        for word in inc.close_notes:
            word_scalar += word[0]
            freq_scalar += word[1]
        #test_series.append(inc)
        test_vectors.append([word_scalar/num_words, freq_scalar/num_words])
        plt.scatter(word_scalar/num_words, freq_scalar/num_words, color='r', marker='*')

plt.xlabel('Integer Representation')
plt.ylabel('Frequency')

plt.title('Integer Representations by Frequency')
plt.legend()


classifier = svm.SVC()

classifier.fit(train_vectors, labels)

outputs = classifier.predict(test_vectors)
print(outputs)
num_correct = 0
for prediction in outputs:
    if prediction == 1:
        num_correct += 1

print('Accuracy:',num_correct/len(outputs))
plt.show()













