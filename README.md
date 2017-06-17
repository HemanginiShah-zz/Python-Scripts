# Python-Scripts
This is the source code for the semantic analysis program for the Incident QA Process at U of M's IT Services. 
    - > Source code

All developers:
Nathan Shepherd

Team Members:
Hema Shah (Project Supervisor)
Chuck Sulikowski (Manager)

    lexical analysis machine
The lexical analysis machine is a work in progress that will use C++ arrays in order to reduce the computation complexity of identifying a principle component in the input vectors (Incident lexicon array). Currently the code from my neural network in this program is optimized for the mnist training set. Running the current version (June 13th) will yeild an error after the lexical analysis.
  - > The lexical_analysis_machine automatically imports all dependencies automatically if they are in Python35\site-packages
  - > wHardware_prediction.csv, workstation_hardware.csv, software_application: must all be in the same directory as this program

The nn_testing file is oriented around the neural network. Fully documented code will lead one familair with neural network around the TensorFlow session. The current version of the nn_testing is meant to be a model for what the optimized version of the final neural network will become.
  
 
Our frontend process to date:
 - > 1.) Daily output file of incident fields in shared folder (in .csv format, fields parsed as strings and denoted by commas)
 - > 2.) Automatically pick up file and send to GitHub (via .git/commit incantation)
 - > 3.) Train model in Cloud (AWS, TensorCloud) and get prediction Output as prediction file. This will represent the correct configuration of each Incident input.
 - > 4.) GitHub sends output file to FTP to update ServiceLink Incidents
