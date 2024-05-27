import plot_emotions
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from GoT_emotions import reshape_data

data_counts = reshape_data("out/Game_of_Thrones_Script_with_emotion_scores.csv")
plot_emotions(data_counts, "rm_neut")