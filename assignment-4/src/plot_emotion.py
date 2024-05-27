import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from codecarbon import EmissionsTracker

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

def reshape_data(data_prepath, tracker):
    """
    The function takes path where the emotions data was saved, and reshapes it. The returned dataframe has three 
    columns, Season and emotion_label and count. For each season the number of each of the seven emotion labels are 
    counted, divided by the total number of lines in the given season and multiplied by 100, resulting in relative 
    requency of each emotion label per 100 lines for each season. 
    """
    tracker.start_task("Reshape data")                               
    data_emotions = pd.read_csv(data_prepath, index_col=0, usecols=[0,2,7])
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
                    palette = sns.mpl_palette("viridis", 7))
    g.set_xticklabels(labels=xorder, rotation=30) 
    plt.savefig(os.path.join("out", f"{subsep}_subplot_{neut}.png"))


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


def main():
    tracker = carbon_tracker(os.path.join("..","assignment-5", "out"))
    args = get_arguments()
    data_prepath = os.path.join("out", "GoT_emotions.csv")
    data_counts = reshape_data(data_prepath, tracker)
    plot_emotions(data_counts, args.neutral, tracker)
    tracker.stop()

if __name__ == "__main__":
    main()