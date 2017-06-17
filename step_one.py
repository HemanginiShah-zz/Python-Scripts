#Step one: get all incident #'s resolved before today
# #
# Powered by NLTK
# Steven Bird, Ewan Klein, and Edward Loper (2009).
#    Natural Language Processing with Python.  O'Reilly Media Inc.
#    http://nltk.org/book
# ------------
# For use with QA Program at U of M: ITS
#   Developed by Nathan Shepherd

from nltk.tokenize import sent_tokenize, word_tokenize

class Incident():
        '''
        def __init__(self, Number, Resolution_category, Short_description, Service_Detail, Service_Catelog_Entry, Close_code, Description):
                self.Number = Number
                self.Short_description = Resolution_category
                self.Service_Detail = Service_Detail
                self.Service_Catelog_Entry = Service_Catelog_Entry
                self.Close_code = Close_code
                self.Description = Description
        '''
        def __init__(self, Number, Service_Detail):
                self.Number = Number
                self.Service_Detail = Service_Detail

        def showAll(self):
                print(self.Number, self.Service_Detail)
        
# All data fields to be collected
fields = {'Number':[], 'Resolution category':[], 'Short description':[], 'Service Detail':[],
          'Service Catalog Entry':[], 'Close code':[], 'Description':[]}

service_detail = ['Classroom Event Support', 'Departmental System', 'Desktop Backup Configuration',
                  'Mobile Device', 'Desktop Backup Configuration', 'Peripherals', 'Printing',
                  'Software Application', 'Storage Permissions Configeration', 'System Application Access',
                  'Workstation Hardware', 'Workstation Network Configuration', 'Workstation OS Configuration',
                  'Workstation Security']
end_service_detail = ['Classroom Event Support', 'Departmental System', 'Desktop Backup Configuration',
                  'Mobile Device', 'Desktop Backup Configuration', 'Peripherals', 'Printing',
                  'Software Application', 'Storage Permissions Configeration', 'System Application Access',
                  'Workstation Hardware', 'Workstation Network Configuration', 'Workstation OS Configuration',
                  'Workstation Security']

inc_sinc_yest = []

for detail in service_detail:
        fields['Service Detail'].append(detail)              

# Takes copypasta and correlates it with the associated fields
    
######    for word in outs:
######        if word not in word_tokenize("()Preview Expand / Collapse GroupService Service Detail:"):
######            raw.append(word)

    

print("Welcome to step one of the Incident QA Program. Press enter to begin.\n")
test = input()

print("Please copy and past all incidents #'s resolved before today in sample_numbers.txt: ")
sample_numbers = open("sample_numbers.txt", "r")

def get_detail_and_num(file):
    outs = ""
    detail = file.readline()
    print(detail)
    detail = word_tokenize(detail)
    for word in detail:
        if word not in word_tokenize("()Preview Expand / Collapse GroupService Service Detail:"):
            outs += word + " "
    return outs
            
for i in range(0, len(fields['Service Detail'])):
        string = get_detail_and_num(sample_numbers)
        print(string)
        detail = string[:-3]
        length = int(word_tokenize(string)[-1])
        if length > 50:
                length = 50
        for incident in range(0, length):
                num = sample_numbers.readline()[7:17]
                inc = Incident(num, detail)
                inc_sinc_yest.append(inc)
        
for inc in inc_sinc_yest:
        inc.showAll()


sample_numbers.close()

#numbers = input()

#print(numbers + "\n\nVerify the above data is collected. Press enter to continue.")
test = input()



        



















