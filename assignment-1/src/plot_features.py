import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from codecarbon import EmissionsTracker

def carbon_tracker(em_outpath):
    """ The function initalizes the carbon tracker """
    tracker = EmissionsTracker(project_name="Assignment-1",
                               output_dir=em_outpath)
    return tracker

def load_data(outfolder):
    """ Loads the .csv files saved to the out folder and returns them as one data frame """ 
    data = pd.DataFrame()
    type_dict ={'a1': "Evaluation",   'a2': "Argumentation", 
                'a3': "Discussion",   'a4': "Literature", 
                'a5': "Culture",      'b1': "Discussion",
                'b2': "Argumentation",'b3': "Linguistics",
                'b4': "Literature",   'b5': "Literature",
                'b6': "Discussion",   'b7': "Discussion",
                'b8': "Evaluation",   'c1': "Literature"}
    for file in sorted(os.listdir(outfolder)):
        if file.endswith(".csv"):
            dat = pd.read_csv(os.path.join(outfolder, file), index_col=0)
            dat["subfolder"] = file.split(".")[0]
            dat["term"] = file[0]
            data = pd.concat([data, dat])
    data['Type'] = data['subfolder'].map(type_dict)
    return data

def plot_NER(data, outpath):
    """ Saves a plot of the named entety recognition results in the data, as mean count per subfolder """
    fig, axs = plt.subplots(1, 3, figsize = (12,5))
    sns.barplot(data=data, x="subfolder", y="Unique LOC", hue="subfolder",  ax=axs[0])
    sns.barplot(data=data, x="subfolder", y="Unique PER", hue="subfolder",  ax=axs[1])
    sns.barplot(data=data, x="subfolder", y="Unique ORG", hue="subfolder", ax=axs[2])
    plt.savefig(os.path.join(outpath, "NER.png"))

def pairplot_POS(data, outpath):
    """ Saves  a plot of the named parts of speech found in the essays in relative frequencies """
    sns.pairplot(data, 
                 hue="subfolder", 
                 vars=["RelFreq VERB", "RelFreq NOUN", "RelFreq ADV", "RelFreq ADJ"], 
                 kind = "scatter", 
                 corner = True)
    plt.savefig(os.path.join(outpath, "pair_POS.png"))

def pairplot_POStype(data, outpath):
    """ Saves a plot of the named parts of speech found in the essays in relative frequencies, coloured by type of text """
    sns.pairplot(data, 
                 hue="Type", 
                 vars=["RelFreq VERB", "RelFreq NOUN", "RelFreq ADV", "RelFreq ADJ"], 
                 kind = "scatter", 
                 corner = True)
    plt.savefig(os.path.join(outpath, "pair_POStype.png"))

def boxplot_POS(data, outpath):
    """ Saves a boxplot of the named parts of speech found in the essays in relative frequencies """
    fig, axs = plt.subplots(2, 2, figsize = (10,7))
    sns.boxplot(data=data, x="subfolder", y="RelFreq NOUN",  hue= "subfolder", ax=axs[0,0])
    sns.boxplot(data=data, x="subfolder", y="RelFreq VERB", hue= "subfolder", ax=axs[0,1])
    sns.boxplot(data=data, x="subfolder", y="RelFreq ADV", hue= "subfolder", ax=axs[1,0])
    sns.boxplot(data=data, x="subfolder", y="RelFreq ADJ", hue= "subfolder", ax=axs[1,1])
    plt.savefig(os.path.join(outpath, "box_POS.png"))

def main():
    tracker = carbon_tracker("../assignment-5/out")
    tracker.start_task("Visualising features")                               
    data = load_data("out")
    plot_path = "out/plots"
    plot_NER(data, plot_path)
    pairplot_POS(data, plot_path)
    pairplot_POStype(data, plot_path)
    boxplot_POS(data, plot_path)
    tracker.stop_task()
    tracker.stop()

if __name__ == "__main__":
    main()
