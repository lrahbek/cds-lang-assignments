import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from codecarbon import EmissionsTracker

def bar_plot(data, xcol, subsep, suborder, xorder, neut, subcols):
    """
    The function creates a collection of barplots, conditioned on different parameters. In this case the following 
    variables should be defined: 
    
    data: The dataframe used for plotting, should contain a Relative Frequency column. 
    xcol: The column used for the x-axis 
    subsep: The column used to base the split into subplots on 
    suborder: Order of subplots 
    xorder: Order of labels on x-axis
    neut: string representing whether or not to keep the neutral emotion label 
    subcols: The number of columns the subplots are arranged in. 

    The function saves the plot to the out folder, its name indicates how the subplots 
    are defined and whether neutral is included.
    """
    colours = list(sns.color_palette("husl", 7))
    colour_dict = {"anger": colours[0], "disgust": colours[2], "fear": colours[1], "joy": colours[6], 
                   "neutral": colours[3], "sadness": colours[4], "surprise": colours[5]}
    sns.set_theme(style = "whitegrid")
    g = sns.catplot(data, 
                    x= xcol, 
                    y = "Relative Frequency",
                    hue = "emotion_label",  
                    col = subsep, 
                    kind = "bar",
                    col_wrap = subcols,
                    col_order = suborder,
                    fill = True, 
                    order = xorder, 
                    palette = colour_dict)
    g.set_xticklabels(labels=xorder, rotation=30) 
    plt.savefig(f"out/{subsep}_subplot_{neut}.png")


def plot_emotions(dataframe, neut, tracker):
    """
    The function takes a dataframe and a string indicating whether or not to keep the neutral emotion label. It 
    utilises the bar_plot() function, and saves both a plot with subplots based on seasons and emotion labels. 
    """
    tracker.start_task("Plot emotions")                               
    seasons =  ["Season 1", "Season 2", "Season 3", "Season 4", "Season 5", "Season 6", "Season 7", "Season 8"]
    emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    if neut == "w_neut":
        bar_plot(dataframe, "emotion_label", "Season", seasons, emotions, neut, 4)
        bar_plot(dataframe, "Season", "emotion_label", emotions, seasons, neut, 4)
    elif neut == "rm_neut":
        emotions.remove("neutral")
        dataframe_rm_neut = dataframe[dataframe["emotion_label"] != "neutral"]
        bar_plot(dataframe_rm_neut, "emotion_label", "Season", seasons, emotions, neut, 4)
        bar_plot(dataframe_rm_neut, "Season", "emotion_label", emotions, seasons, neut, 3)
    tracker.stop_task()
