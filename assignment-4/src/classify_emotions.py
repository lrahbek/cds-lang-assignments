from transformers import pipeline
import tensorflow
import keras
import pandas as pd
import os
from tqdm import tqdm
from codecarbon import EmissionsTracker

def carbon_tracker(em_outpath):
    """ The function initalizes the carbon tracker """
    tracker = EmissionsTracker(project_name="Assignment-4",
                               output_dir=em_outpath)
    return tracker

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
    data_in = pd.read_csv(filepath)
    data_emotions = pd.read_csv(filepath)
    data_emotions[["emotion_label", "emotion_score"]] = ""
    tracker.stop_task()
    return data_in, data_emotions

def emotion_classifier(data_in, data_emotions, classifier, data_prepath, tracker):
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
    data_emotions.to_csv(data_prepath)
    tracker.stop_task()
    return print("Emotion scores and labels for each sentence in the dataset have been saved to the outfolder")

def main():
    tracker = carbon_tracker(os.path.join("..","assignment-5", "out"))
    args = get_arguments()
    filepath = os.path.join("in", "GoT-scripts", "Game_of_Thrones_Script.csv")
    data_prepath = os.path.join("out", "GoT_emotions.csv")
    classifier = load_classifier(tracker)    
    data_in, data_emotions = load_data(filepath, tracker)
    emotion_classifier(data_in, data_emotions, classifier, data_prepath, tracker)
    tracker.stop()

if __name__ == "__main__":
    main()