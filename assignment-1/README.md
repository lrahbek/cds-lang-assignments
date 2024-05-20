# Assignment 1 - Extracting Linguistic Features Using spaCy

*Date: 23/02/2024*

Laura Givskov Rahbek 

## Description 

This folder contains assignment 1 for Language Analytics. The objective of the assignment is to work with input data arranged hierarchically in folders, to use ```spaCy``` to extract information from text data, and to save results and outputs in a clear and understandable way. More specifically the analysis aims to extract relevant textual features from the texts used (see Data section), and investigate whether these can capture the differences in the text. The folder contains two scripts, each described below; 

The ```feature_extraction.py``` script does the folllowing: 

- Loads an English language ```spaCy``` model, the default being ```en_core_web_md```. 
- Goes through each text in each of the subfolders in the ```in``` folder, for every text: 
  - The meta data is removed 
  - The relavtive frequency per 10,000 words of nouns, verbs, adjectives and adverbs are found 
  - The number og unique persons, locations and organisations are found
  - The word frquencies and named enteties are saved in a row in a .csv file, named according to the given subfolder, placed in the ```out``` folder. 

The ```plot_features.py``` script does the following: 
- Loads in the .csv files with the extracted features from the ```out``` folder and adds three columns; 
  - subfolder: the name of the subfolder, indicating which specific essay prompt the text was written from. 
  - term: a, b or c, indicating whether the essay was written in term 1, 2 or 3. 
  - type: in the documentation for the corpus, the subfolders all belong to one of the following categories (in terms of type of text); Evaluation, Argumentation, Discussion, Linguistics, Literature and Culture. 
- Saves different plots, viusaling the different features by the different possible groups. The plots are saved in the ```out``` folder.
 
## Data

The data used is the *The Uppsala Student English Corpus (USE)*, which is a collection of essays in English written by Swedish university students. The data can be accesed at [this link](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2457), where additional information on the corpus can be found too. 

The subfolders reflect the term the essay was written in, subfolder starting with *a* was written in the first term, *b* in the second and *c* in the third. Further each subfolder in the three categories represent a specific essay prompt. 

## Usage and Reproducing of Analysis

To reproduce the analysis: 
- Download and unzip the ```USEcoprus``` folder, and place it in the ```in``` folder in the repository. 
- Run the bash script ```setup.sh``` from the command line, it creates a virtual environment and installs packages and dependencies in to it. 
- Run the bash script ```run.sh``` from the command line, this opens the virtual environment  and runs both scripts in ```src``` folder. 

To use another spaCy model than ```en_core_web_md```, pass another model when running ```run.sh```, either ```en_core_web_sm``` or ```en_core_web_lg```. 

  ```
  bash run.sh -m {'model'} 
  ```

## Discussion 

Before discussing the potential tendencies found in the visualization of the features extracted from the *USE corpus*, several limitations to this method are worth mentioning. First, ```en_core_web_md``` was trained on web data, (blogs, news, comments etc. [see the documentation for more](https://github.com/explosion/spacy-models)). Which is not the same type of text used on in this assignment. Word use, phrasing and the theme of the texts are very different to general web text, which might introduce some issues or a loss of information. Further, the different texts are of varying length, and the different essay categories and subfolders contain a different number of essays, which introduces differences in variance when aggregating the different variables. 

![Named Entety Recognition](https://github.com/lrahbek/cds-lang-assignments/blob/main/assignment-1/out/plots/NER.png)

With that, I will discuss tendencies found in the plots. First the Named Entety Recognition plot (```NER.png```), the first thing that stands out are the subfolders with much higher means than the rest. Specifically across all three NERs (location, persons and organisations) the following subfolders have high means in some or all of named enteties; ```a5```, ```b3```, ```b8``` and ```c1```.  However, for ```b8``` the errorbar reveals high variance, in this case due to only three texts in this category. Different variables sets the remaining three subfolders apart from the rest, ```a5``` and ```b3``` are the only subfolders in their respective text-type categories, Culture and Linguistics. ```c1``` contains Literature essays, along with other subfolders, but is the only written in the third term. 

I have visualised the Parts Of Speech found in the text in three plots:
- ```pair_POS.png``` a pair plot visualising the correlation between the four POS variables in the data, grouped by subfolder.
- ```box_POS.png``` a boxplot visualising the spread of relative frequency in the four POS variables, grouped by subfolder.
- ```pair_POStype.png``` a pair plot visualising the correlation between the four POS variables in the data, grouped by text type. 

The pairplot grouped by subfolder indicates a couple of things. First, their seems to be a vague negative correlation between relative frequency of nouns and verbs, as well as between verbs and adjective and nouns and adverbs. These tendencies are then matched with positive correlation between verbs and adverbs and nouns and adjectives. These correlations seem to conform to the expectations of word use (verbs go with adverbs and nouns go with adjectives). Further, the diagonal distributions seem to reveal that the distributions of ```a1``` are distinct to the rest for relative frequency of nouns and verbs. However, with a purely visual inspection conclusions cannot be drawn. To inspect these distributions further, the boxplot was made. The boxplot affirms that ```a1``` has a lower relative frequency of nouns. Further, ```b3``` seem to have a lower relative frequency of verbs. Because of the many different subfolders, or groups, in the data, it is difficult to find clusters or tendencies. Therefore, the last plot that was inspected was the pairplot grouped by text-type. 

The most obvious distnictions can be found in the distribution showing relative frequency of nouns. Here, Discussion and Argumentation essays seem to be part of the same distribution (overlapping) and Evalutation and Literature essays seem to be part part of the same distribution. When looking at the distribution of relative frequency of adverbs, Argumentaion and Discussion essays are again overlapping, but Evaluation and Literature essays are distinct from each other. 

Across the visualizations there seems to be some tendencies unique to some of the essay prompts and text types, however to properly evaluate whether these tendencies are signficant to the individual groupings, further analysis would have to take place. 

*```codecarbon``` was used to track the environmental impact when running this code, the results and an exploration of this can be found in the ```Assignment-5``` folder in the repository.*
