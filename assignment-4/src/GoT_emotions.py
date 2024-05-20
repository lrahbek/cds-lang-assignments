from transformers import pipeline
import tensorflow
import keras
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from codecarbon import EmissionsTracker
import plot_emotions

def carbon_tracker(em_outpath):
    """ The function initalizes the carbon tracker """
    tracker = EmissionsTracker(project_name="Assignment-4",
                               output_dir=em_outpath)
    return tracker

def get_arguments():
    """ The filepath for the data used in the analysis """
    parser = argparse.ArgumentParser()    
    parser.add_argument("--neutral",
                        "-n", 
                        required = True,
                        choices = ["w_neut", "rm_neut"],
                        help="Choose whether or not to keep the neutral emotion when saving the plots")         
    args = parser.parse_args()
    return args


def load_classifier(tracker):
    """ The function loads and returns the emotion classifier """
    tracker.start_task("Load model")                               
    classifier = pipeline("text-classification", 
                          model="j-hartmann/emotion-english-distilroberta-base", 
                          return_all_scores=False) 
    tracker.stop_task()                      
    return classifier


def load_data(filepath, tracker):
    """
    The function takes a filepath where the data is, and returns a dataframe with the dataset, a dataframe with 
    additional emotion label and score columns, and the out path for the data with emotion labels and scores. 
    """
    tracker.start_task("Load data")                               
    data_emotions_path = "out/" + (filepath).split("/")[-1].split(".")[0] + "_with_emotion_scores.csv"
    data_in = pd.read_csv(filepath)
    data_emotions = pd.read_csv(filepath)
    data_emotions[["emotion_label", "emotion_score"]] = ""
    tracker.stop_task()
    return data_in, data_emotions, data_emotions_path


def emotion_classifier(data_in, data_emotions, classifier, data_emotions_path, tracker):
    """
    The function takes the two dataframes created by load_data(), the out path and the classifier loaded by 
    load_classifier(). For each sentence in the dataset it appends the most prominent emotion label as well as its 
    score to the data_emotions dataframe. When all sentences have been run through, it saves the data_emotions 
    dataframe to the path defined in load_data().
    """
    tracker.start_task("Emotion classification")                               
    for sent in tqdm(range(len(data_in["Sentence"]))):
        sentence = data_in["Sentence"][sent]
        if type(sentence) == str:
            emotion_label = classifier(sentence)[0]["label"]
            emotion_score = classifier(sentence)[0]["score"]
            data_emotions.loc[sent, "emotion_label"] = emotion_label
            data_emotions.loc[sent, "emotion_score"] = emotion_score
    data_emotions.to_csv(data_emotions_path)
    tracker.stop_task()
    return print("Emotion scores and labels for each sentence in the dataset have been saved to the outfolder")


def reshape_data(data_emotions_path, tracker):
    """
    The function takes path where the emotions data was saved, and reshapes it. The returned dataframe has three 
    columns, Season and emotion_label and count. For each season the number of each of the seven emotion labels are 
    counted, divided by the total number of lines in the given season and multiplied by 100, resulting in relative 
    requency of each emotion label per 100 lines for each season. 
    """
    tracker.start_task("Reshape data")                               
    data_emotions = pd.read_csv(data_emotions_path, index_col=0, usecols=[0,2,7])
    Seasons = list(data_emotions.Season.unique())
    season_len = []
    for n in range(len(Seasons)):
        season = Seasons[n]
        season_len = season_len + [len(data_emotions.loc[data_emotions["Season"] == f"{season}"])]
    data_counts =  data_emotions.value_counts().reset_index().rename(columns={"index": "value", 0: "counts"})
    data_counts["Relative Frequency"] = ""
    for season, length in zip(Seasons, season_len):
        data_counts.loc[data_counts["Season"] == season, "Relative Frequency"] = data_counts.loc[data_counts["Season"] == season, "count"]/length * 100
    tracker.stop_task()
    return data_counts


def main():
    tracker = carbon_tracker("../assignment-5/out")
    args = get_arguments()
    filepath = "in/GoT-scripts/Game_of_Thrones_Script.csv"
    classifier = load_classifier(tracker)    
    data_in, data_emotions, data_emotions_path = load_data(filepath, tracker)
    emotion_classifier(data_in, data_emotions, classifier, data_emotions_path, tracker)
    data_counts = reshape_data(data_emotions_path, tracker)
    plot_emotions(data_counts, args.neutral, tracker)
    tracker.stop()

if __name__ == "__main__":
    main()

