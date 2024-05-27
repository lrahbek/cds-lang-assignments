import os
import pandas as pd 
import gensim.downloader as api
from collections import Counter
import string
from csv import writer
import argparse
from codecarbon import EmissionsTracker

def carbon_tracker(em_outpath):
    """ The function initalizes the carbon tracker """
    tracker = EmissionsTracker(project_name="Assignment-3",
                               output_dir=em_outpath)
    return tracker

def get_arguments():
    """ Get input arguments for query expansion; artist and keyword """
    parser = argparse.ArgumentParser()
    parser.add_argument("--artist",
                        "-a", 
                        required = True,
                        help="The artist should be given as a string")
    parser.add_argument("--keyword",
                        "-k", 
                        required = True,
                        help="The keyword shoule be given in a string")                    
    args = parser.parse_args()
    return args

def artist_find(artist_input, tracker):
    """
    The function loads in the Spotify Million Song Dataset and takes a string as input, if the string matches 
    with an artist name in the corpus, a list with the given artist's songs is returned.
    """
    tracker.start_task("Find artist")                               
    songs = pd.read_csv(os.path.join("in", "Spotify Million Song Dataset_exported.csv"))
    artist = artist_input.lower()
    songs["artist"] = (songs["artist"]).str.lower()
    if (songs['artist'].eq(artist).any()):
        artist_corpus = songs[songs["artist"] == artist]
        artist_songs = list(artist_corpus["text"])
    else: print('Artist not found') 
    tracker.stop_task()
    return artist_songs

def query_expansion(keyword, tracker):
    """
    The function first loads the 'glove-wiki-gigaword-50' gensim model and takes a keyword. The gensim model is 
    used to find the 10 most similar words, which the function returns as a list. 
    """
    tracker.start_task("Load model and find words") 
    model = api.load("glove-wiki-gigaword-50")
    query = model.most_similar(keyword)
    query_ls = ([i[0] for i in query])
    query_ls.append(keyword)
    tracker.stop_task()
    return query_ls

def clean_tokens(text):
    """
    The function takes a text as an input tokenizes it, makes it lowercase and strips the puctuation from all 
    words. It returns a list of the cleaned tokens.
    """
    tokens = text.lower().split()
    tokens_cleaned = []
    for token in tokens: 
        strip_punct = token.strip(string.punctuation)
        if not strip_punct == "":
            tokens_cleaned.append(strip_punct)
    return tokens_cleaned

def save_output(row):
    """ Save output to .csv file """
    with open(os.path.join('out', 'output.csv'), 'a') as output:
        write_output = writer(output)
        write_output.writerow(row)
        output.close()

def keyword_count(query_ls, keyword, song_texts, artist_input, tracker):
    """
    The function takes a keyword and a list of song texts, it returns the percentage of the given songs that 
    contains any of the 10 words mostly related to the keyword, and the keyword, of the number of songs from 
    that artist. 
    """
    tracker.start_task("Keyword query") 
    query_c = Counter(query_ls)
    total_songs = len(song_texts)
    song_count = 0
    for text in range(len(song_texts)):
        tokens_cleaned = clean_tokens(song_texts[text])       
        tokens_c = Counter(tokens_cleaned)
        keyword_count = (tokens_c & query_c).total()
        if keyword_count > 0: 
            song_count +=1
    songs_perc = round(((song_count/len(song_texts))*100), 2)
    row = [keyword, query_ls[0:10], artist_input, total_songs, song_count, songs_perc]
    save_output(row)
    tracker.stop_task()
    return print(f"{songs_perc}% of {artist_input}'s songs contains words related to {keyword}")    

def main():
    tracker = carbon_tracker(os.path.join("..","assignment-5", "out"))
    args = get_arguments()
    artist_songs = artist_find(args.artist, tracker)
    query_ls = query_expansion(args.keyword, tracker)
    keyword_count(query_ls, args.keyword, artist_songs, args.artist, tracker)
    tracker.stop()

if __name__ == "__main__":
    main()
