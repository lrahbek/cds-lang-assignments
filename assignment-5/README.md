# Assignment 5 - Evaluating Environmental Impact of Exam Portfolio

*Date: 02/05/2024*

Laura Givskov Rahbek 

## Description 

This folder contains assignment 5 for Language Analytics. The objective of the assignment is to think critically about the effects and impact machhine learne has on the environment, to write code that can extract approximate benchmarks showing this impact and to present and discuss the results in an understandable way. More specifically, the package ```CodeCarbon```'s class ```EmissionsTracker``` is utilised to measure the approximate CO₂-equivalents (CO₂eq) in kg for assignments 1-4 in this repository. The resulting values are used to investigate the environmental impact of the different tasks performed in the four previous assignments done in this class. 

The ```assignment-5``` folder contains two folders: 
- An ```out``` folder which contains the .csv files with the emmision data from each of the four assignments. 
- A ```src``` folder containing the notebook ```emmisions.ipynb```, the handeling of the emmisions data and subsequent visualization used for analysis is in this notebook. 





## Usage and Reproducing of Analysis

All code used to handle the emmisions data as well as visualisations of these can be found in the notebook in the ```src``` folder. Packages and dependencies to the run the notebook can be installed by running the bash script ```setup.sh```. To view the packages that are installed when running this, see the ```requirements.txt``` file. 

To track the emissions from the assignments the ```EmissionTracker``` from ```CoddeCarbon``` was used, to see how precisely this was done, the code can be found in the scripts in the assignments. Loosely however, the tracker was ininialised and then used to track individual taksks, to be able to investigate the differences. The emissions were added up to infer the total emissions per assignment. 

## Discussion 


- Which assignment generated the most emissions in terms of CO₂eq? Explain why this might be.
- Which specific tasks generated the most emissions in terms of CO₂eq? Again, explain why this might be.
- How robust do you think these results are and how/where might they be improved? 

*change code to have less enviormental impact*

