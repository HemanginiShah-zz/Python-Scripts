# Python-Scripts
This is the source code for the semantic analysis program for the Incident QA Process at U of M's IT Services. 
    - > Source code

All developers:
Nathan Shepherd

Team Members:
Hema Shah (Project Supervisor)
Chuck Sulikowski (Manager)

    pro_svm == Production scale-able support vector machine
The support vector machine classifies a given service incident based on the occurrence of words in fields of text. For example, an incident that mentions ['print', 'printing', 'can't connect', 'jam'] should be classified as Printing. A seperate incident that mentions ['Monitor','SSD','Keyboard'] should be classified as Workstation Hardware.
  - > The prod_svm automatically imports all dependencies automatically if they are in Python35\site-packages
  - > SD_SoftApp_TrainingData.csv, SD_wHardware_PredictionData.csv, SD_wHardware_TrainingData.csv: must all be in the same directory as this program

The nn_testing file is oriented around the neural network. Fully documented code will lead one familair with neural network around the TensorFlow session. The current version of the nn_testing is meant to be a model for what the optimized version of the final neural network will become.
  
 
Our frontend process to date:
 - > 1.) Daily output file of incident fields in shared folder (in .csv format, fields parsed as strings and denoted by commas)
 - > 2.) Automatically pick up file and send to GitHub (via .git/commit incantation)
 - > 3.) Train model in Cloud (AWS, TensorCloud) and get prediction Output as prediction file. This will represent the correct configuration of each Incident input.
 - > 4.) GitHub sends output file to FTP to update ServiceLink Incidents
