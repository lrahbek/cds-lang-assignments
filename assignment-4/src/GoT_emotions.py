from transformers import pipeline
import pandas as pd
import os
import matplotlib as plt
import seaborn as sns
import argparse

def get_arguments():
    """
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
                        "--filepath",
                        "-f", 
                        required = False,
                        default = "in/GoT-scripts/Game_of_Thrones_Script.csv",
                        help="The path where the data can be found")                  
    args = parser.parse_args()
    return args


def load_classifier():
    classifier = pipeline("text-classification", 
                          model="j-hartmann/emotion-english-distilroberta-base", 
                          return_all_scores=False) 
    return classifier


def load_data(filepath):
    data_emotions_path = "out/" + (filepath).split("/")[-1].split(".")[0] + "_with_emotion_scores.csv"
    data_in = pd.read_csv(filepath)
    data_emotions = pd.read_csv(filepath)
    data_emotions[["emotion_label", "emotion_score"]] = ""
    return data_in, data_emotions, data_emotions_path


def emotion_classifier(data_in, data_emotions, classifier, data_emotions_path):
    for sent in range(len(data_in["Sentence"])):
        sentence = data_in["Sentence"][sent]
        if type(sentence) == str:
            emotion_label = classifier(sentence)[0]["label"]
            emotion_score = classifier(sentence)[0]["score"]
            data_emotions.loc[sent, "emotion_label"] = emotion_label
            data_emotions.loc[sent, "emotion_score"] = emotion_score
    data_emotions.to_csv(data_emotions_path)
    return print("Emotion scores and labels for each sentence in the dataset have been saved to the outfolder")


def main():
    args = get_arguments()
    classifier = load_classifier()    
    data_in, data_emotions, data_emotions_path = load_data(args.filepath)
    emotion_classifier(data_in, data_emotions, classifier, data_emotions_path)
    

if __name__ == "__main__":
    main()

