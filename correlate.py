# Correlate: Categorize words using word2vec, correlate specific words with fields
# #
# Powered by NLTK
# Steven Bird, Ewan Klein, and Edward Loper (2009).
#    Natural Language Processing with Python.  O'Reilly Media Inc.
#    http://nltk.org/book
# ------------
# For use with QA Program at U of M: ITS
#   Developed by Nathan Shepherd

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import collections

all_incidents = {'Number':[],
                 'Resolution category':[],
                 'Short description':[],
                 'Service Detail':[],
                 'Category':[]}
all_fields = ['Number',
              'Resolution category',
              'Short description',
              'Service Detail',
              'Category']

stop_words = set(stopwords.words('english'))

num_incs = 0
corr_data = open("corr_data.txt", "r")
num_incs = int(corr_data.readline())

class Incident():
        def __init__(self, num, res_cat, sho_des, ser_det, category):
                self.num = num
                self.res_cat = res_cat
                self.sho_des = sho_des
                self.ser_det = ser_det
                self.category = category

        
        
def check(num_incs):
    corr_data = open("corr_data.txt", "r")
    num_incs = int(corr_data.readline())
    
    print("The number of incidents you entered was " + str(num_incs) +
      ", is this correct? [Y/n]")
    reply = input()

    if reply.lower() == 'y':
        corr_data.close()
    elif reply.lower() == 'n':
        print('\nSave and exit notepad and try again.\n')
        corr_data.close()
        check(num_incs)
    else:
        print("\nEnter either \'y\' for yes, or \'n\' for no.\n")
        corr_data.close()
        check(num_incs)

def initialize_dict():
     corr_data = open("corr_data.txt", "r")
     num_tickets = int(corr_data.readline())

     number = []
     resolution_category = []
     short_description = []
     service_detail = []
     category = []
     
     for field in range(0, 5):
        entry = corr_data.readline()
        for i in range(0, num_tickets):
            if entry == 'Number\n':
                number.append(corr_data.readline())
            if entry == 'Resolution category\n':
                resolution_category.append(corr_data.readline())
            if entry == 'Short description\n':
                short_description.append(corr_data.readline())
            if entry == 'Service Detail\n':
                service_detail.append(corr_data.readline())
            if entry == 'Category\n':
                category.append(corr_data.readline())

     # removing \n at end of line
     for i in range(0, num_tickets):
        number[i] = number[i][:-1]
        resolution_category[i] = resolution_category[i][:-1]
        short_description[i] = short_description[i][:-1]
        service_detail[i] = service_detail[i][:-1]
        category[i] = category[i][:-1]

     all_incidents['Number'] = number
     all_incidents['Resolution category'] = resolution_category
     all_incidents['Short description'] = short_description
     all_incidents['Service Detail'] = service_detail
     all_incidents['Category'] = category

     corr_data.close()

def tokenize(field):
        #TODO: return matricies
        inc_num_matrix = {}
        if field == 'Number':
            for i in range(0, num_incs):
                inc_num_matrix[i] = all_incidents['Number'][i]
        for num in inc_num_matrix:
            if num < 10:
                print(num, inc_num_matrix[num])
        
        res_cat_matrix = {}
        cache = []
        if field == 'Resolution category':
            for i in range(0, num_incs):
                all_incidents['Resolution category'][i] = sent_tokenize(all_incidents['Resolution category'][i])
                for sent in all_incidents['Resolution category'][i]:
                    cache.append(sent)
            for sent in cache:
                if sent not in res_cat_matrix.keys():
                    res_cat_matrix[sent] = 0
                else:
                    res_cat_matrix[sent] += 1
            #res_cat_matrix = value_sort(res_cat_matrix)
            
            for value in res_cat_matrix:
                if res_cat_matrix[value] > 2:
                    print(value, res_cat_matrix[value])    

        #FIXME: Improve tokenization algorithm for sho_des
        sho_des_matrix = {}
        if field == 'Short description':
            for i in range(0, num_incs):
                all_incidents['Short description'][i] = word_tokenize(all_incidents['Short description'][i])
                for word in all_incidents['Short description'][i]:
                    if word not in set(stopwords.words("english")):
                        if word in sho_des_matrix:
                            sho_des_matrix[word] += 1
                        else:
                            sho_des_matrix[word] = 0
        for key in sho_des_matrix:
            if sho_des_matrix[key] > 15:
                print(key, sho_des_matrix[key])

        ser_det_matrix = {}
        if field == 'Service Detail':
            for detail in all_incidents['Service Detail']:
                if detail in ser_det_matrix:
                    ser_det_matrix[detail] += 1
                else:
                    ser_det_matrix[detail] = 0
        for detail in ser_det_matrix:
            print(detail, ser_det_matrix[detail])

        category_matrix = {}
        if field == 'Category':
            for category in all_incidents['Category']:
                if category in category_matrix:
                    category_matrix[category] += 1
                else:
                    category_matrix[category] = 0
        for category in category_matrix:
            print(category, category_matrix[category])
                    
def value_sort(matrix):
        refresh = matrix
        sorts = {}
        maximum = max(matrix.items())
        length = len(matrix)
        
        for i in range(0, length):
            for key in matrix:
                if matrix[key] == maximum:
                    sorts[key] = matrix[key]
                    del(matrix[key])
                    maximum = max(matrix.items())
        matrix = refresh
        return sorts

incidents = {}
def init_objects():
    for i in range(0, num_incs):
        incidents[i] = Incident(all_incidents['Number'][i],
                                all_incidents['Resolution category'][i],
                                all_incidents['Short description'][i],
                                all_incidents['Service Detail'][i],
                                all_incidents['Category'][i])
        
        
        

print("\nWelcome to step one of the Incident QA Program.\nThis program will train the AI System.\n\nPress enter to begin.")
wait_for = input()

print('''
These are the steps you must follow:

1.)\tOpen schedualed Automated QA Report
2.)\tOpen file called \"corr_data.txt\"
3.)\tDelete old contents
4.)\tEnter number of incidents on first line
5.)\tCopy and paste each column from excel sheet, including title
6.)\tAdd new line at end of file
7.)\tSAVE and exit corr_data.txt

Press enter when the above steps are completed.
''')
wait_for = input()

check(num_incs)

initialize_dict()

for field in all_fields:
    print('\n\n\t'+field+'\n')
    tokenize(field)

print('\n\n^ Verify the above extracted fields are easily readable. ^')
print('\nCurrently the matrix represents the frequency of each entry in each field.')
print('\nPress enter to define the training perameters.\n')
wait_for = input()

init_objects()
for i in range(0, 10):
    print(incidents[i])

#TODO: word2vector(tokenized_matrix)

#TODO: determine percent simularity between tickets with the same service detail

'''switch
def f(x):
    return {
        'a': 1,
        'b': 2,
    }[x]
'''
            
