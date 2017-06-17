#count: correlate fields and data, output word frequency
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
        def __init__(self, Number, Resolution_category, Short_description, Service_Detail, Service_Catelog_Entry, Close_code, Description):
                self.Number = Number
                self.Resolution_category = Resolution_category
                self.Short_description = Short_description
                self.Service_Detail = Service_Detail
                self.Service_Catelog_Entry = Service_Catelog_Entry
                self.Close_code = Close_code
                self.Description = Description


        def show_all(self):
                print(self.Number, self.Short_description, self.Service_Detail, self.Service_Catelog_Entry, self.Close_code, self.Description)

        def get_Number(self):
            return self.Number
        def get_Short_description(self):
            return self.Short_description
        def get_Resolution_category(self):
            return self.Resolution_category
        def get_Service_Detail(self):
            return self.Service_Detail
        def get_Service_Catelog_Entry(self):
            return self.Service_Catelog_Entry
        def get_Close_code(self):
            return self.Close_code
        def get_Description(self):
            return self.Description

        def set_Number(self, Number):
            self.Number = Number
        def set_Resolution_category(self, Resolution_category):
            self.Resolution_category = Resolution_category
        def set_Short_description(self, Short_description):
            self.Short_description = Short_description
        def set_Service_Detail(self, Service_Detail):
            self.Service_Detail = Service_Detail
        def set_Service_Catelog_Entry(self, Service_Catelog_Entry):
            self.Service_Catelog_Entry = Service_Catelog_Entry
        def set_Close_code(self, Close_code):
            self.Close_code = Close_code
        def set_Description(self, Description):
            self.Description = Description
        
# All data fields to be collected
test_data = {'Number':['INC1036118','INC1064252', 'INC1063881'],
             'Resolution category':['Setup Support Provided',
                                    'How-to: Information Provided',
                                    'Other'],
             'Short description':['Event Support:  presentation set up 11:30am 5/31',
                                  'Adding funds for printing',
                                  'Printer Jam m-rack-1530-color-1'],
             'Service Detail':['Classroom / Event Support', 'Printing', 'Printing'],
             'Service Catalog Entry':['Other', '--none--', '--none--'],
             'Close code':['Solved (Confirmed by Customer',
                           'Solved (Permanently)',
                           'Solved (Customer Confirmation Pending)'],
             'Description':['''
Good morning,

I would like to request assistance with presentation set up for three upcoming events.

2.       Wednesday, May 31, help with set up at 11:40-11:45 a.m.

SNB 1250

Thank you for your help,

Alisa Maiville
Project Coordinator
National Program Office, Alliance to Advance Patient-Centered Cancer Care
University of Michigan School of Nursing
Suite 1174, 400 North Ingalls Building
Ann Arbor MI, 48109
ph. 734-647-6848 | maiville@med.umich.edu<mailto:maiville@med.umich.edu>
''',
                            '''
How do I add money to my account for printing?  I found the MPrint website but can't find anywhere that says add to current balance - Contact user by Email
''',
                            'Arah called in to report that m-rack-1530-color-1 is showing a paper jam, but she is not able to locate the jam.'
                            ]}

service_detail = ['Classroom Event Support', 'Departmental System', 'Desktop Backup Configuration',
                  'Mobile Device', 'Desktop Backup Configuration', 'Peripherals', 'Printing',
                  'Software Application', 'Storage Permissions Configeration', 'System Application Access',
                  'Workstation Hardware', 'Workstation Network Configuration', 'Workstation OS Configuration',
                  'Workstation Security']


test_incidents = []

for key in test_data:
    for i in range(0,3):
        test_data[key][i] = word_tokenize(test_data[key][i])

for i in range(0,3):
    test_incidents.append(
        Incident(test_data['Number'][i],
                   test_data['Resolution category'][i],
                   test_data['Short description'][i],
                   test_data['Service Detail'][i],
                   test_data['Service Catalog Entry'][i],
                   test_data['Close code'][i],
                   test_data['Description'][i]
                   ))

for inc in test_incidents:
    inc.show_all()












