#test

'''
corpera = {'Workstation Hardware':{0:[], 1:[], 2:[]},
           'Classroom / Event Support':{0:[], 1:[], 2:[]},
           'Software Application':{0:[], 1:[], 2:[]},
           'Printing':{0:[], 1:[], 2:[]}}
'''

dictionary = {}

words1 = ['help', 'this', 'is', 'not', 'a', 'test', 'test']
words2 = ['help', 'the', 'is', 'never', 'a', 'help', 'test']

sents = []
sents.append(words1)
sents.append(words2)

for sent in sents:
    for word in sent:
        if word not in dictionary.keys():
            dictionary[word] = len(dictionary)

print(dictionary,'')

for sent in sents:
    for word in sent:
        for key, value in dictionary.items():
            if word == key:
                sents[word] = dictionary[key]

print(sents)
