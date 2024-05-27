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

def reshape_data(data_prepath, column):
    """ 
    The function takes two arguments, the path were the data is stored and the column (variable) the relative frequency 
    should be calculated by. The function loads in the data, with just the Season and emotion_label columns, and 
    removes NaN values. Then the relative frequency of the given variable in the dataset is calculated, the resulting 
    values represent the percentage of the given variable that is represented by the remaining variable in the dataframe. 
    """ 
    data_emotions = pd.read_csv(data_prepath, index_col=0, usecols=[0,2,7])
    data_emotions.dropna(subset=["emotion_label"], inplace=True)
    unique_col = list(data_emotions[column].unique())
    col_length = []
    for n in range(len(unique_col)):
        col_value = unique_col[n]
        col_length = col_length + [len(data_emotions.loc[data_emotions[column] == f"{col_value}"])]
    rel_freq =  data_emotions.value_counts().reset_index().rename(columns={"index": "value", 0: "counts"})
    rel_freq["Relative Frequency"] = ""
    for col_value, length in zip(unique_col, col_length):
        rel_freq.loc[rel_freq[column] == col_value, "Relative Frequency"] = rel_freq.loc[rel_freq[column] == col_value, "count"]/length * 100
    return rel_freq
 
def bar_plot(data, X, neut):
    """
    The function saves a barplot of the relative frequency of the given variables to the out folder.
    In all cases, the bars a re coloured by the emotion label and the y axis = relative frequency. 
    The function takes a dataframe, as the one produced by reshape_data(). X will be on the x-axis 
    and the remaining categorical variable in the dataframe will be represented individual subplots. 
    """                              
    seasons =  ["Season 1", "Season 2", "Season 3", "Season 4", "Season 5", "Season 6", "Season 7", "Season 8"]
    emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    if neut == "rm_neut":
        emotions.remove("neutral")
        data = data[data["emotion_label"] != "neutral"]
    if X == "Season":
        order = seasons
        col = "emotion_label" 
        col_order = emotions
    elif X == "emotion_label":
        order = emotions 
        col = "Season"
        col_order = seasons
    if len(col_order)/2>3: col_wrap = 4
    else: col_wrap = 3
    sns.set_theme(style = "whitegrid")
    g = sns.catplot(data, 
                    x= X, 
                    order = order,
                    y = "Relative Frequency",
                    hue = "emotion_label",  
                    kind = "bar",
                    col = col, 
                    col_wrap = col_wrap,
                    col_order = col_order,
                    fill = True, 
                    palette = sns.mpl_palette("viridis", len(emotions)))
    g.set_xticklabels(labels=order, rotation=30) 
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    plt.savefig(os.path.join("out", f"{col}_subplot_{neut}.png"))
    return print("The plots have been saved to the out folder")


def main():
    tracker = carbon_tracker(os.path.join("..","assignment-5", "out"))
    args = get_arguments()
    data_prepath = os.path.join("out", "GoT_emotions.csv")
    tracker.start_task("Reshape data")    
    rel_freq_em = reshape_data(data_prepath, "emotion_label")
    rel_freq_se = reshape_data(data_prepath, "Season")
    tracker.stop_task()
    tracker.start_task("Plot emotions") 
    bar_plot(rel_freq_em, "Season", args.neutral)
    bar_plot(rel_freq_se, "emotion_label", args.neutral)
    tracker.stop_task()
    tracker.stop()

if __name__ == "__main__":
    main()