import os
import pandas as pd 
import gensim.downloader as api
from collections import Counter
import argparse



def get_arguments():
    """
    Get input arguments for query expansion; artist and keyword
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
                        "--artist",
                        "-a", 
                        required = True,
                        help="The artist should be given as a string")

    parser.add_argument(
                        "--keyword",
                        "-k", 
                        required = True,
                        help="The keyword shoule be given in a string")                    

    args = parser.parse_args()
    return args



def artist_find(artist_q):
    """
    The function loads in the Spotify Million Song Dataset and takes a string as input, 
    if the string matches with an artist name in the corpus, a list with the given 
    artists' songs is returned.
    """
    filepath = os.path.join("..", "in", "Spotify Million Song Dataset_exported.csv")
    songs = pd.read_csv(filepath)

    artist_q = artist_q.lower()
    songs["artist"] = (songs["artist"]).str.lower()
    
    if (songs['artist'].eq(artist_q).any()):
        artist_cor = songs[songs["artist"] == artist_q]
        artist_songs = list(artist_cor["text"])
    else: print('Artist not found') 

    return artist_songs



def query_expansion(keyword):
    """
    The function first loads the 'glove-wiki-gigaword-50' gensim model and takes
    a keyword. The gensim model is used to find the 10 most similar words, which
    the function returns as a list. 
    """
    model = api.load("glove-wiki-gigaword-50")

    query = model.most_similar(keyword)
    query_ls = ([i[0] for i in query])
    query_ls.append(keyword)

    return query_ls



def keyword_count(keyword, song_texts, artist_q):
    """
    The function takes a keyword and a list of song texts, it returns the number
    of songs in the input list that contains any of the words in the keyword list. 
    """
    query_c = Counter(query_expansion(keyword))
    song_count = 0

    for text in range(len(song_texts)):     
        text_c = Counter(song_texts[text].lower().split())
        keyword_count = (text_c & query_c).total()
        if keyword_count > 0: 
            song_count +=1
    songs_perc = round(((song_count/len(song_texts))*100), 2)

    return print(f"{songs_perc}% of {artist_q}'s songs contains words related to {keyword}")    


def main():
    args = get_arguments()
    artist_songs = artist_find(args.artist)
    keyword_count(args.keyword, artist_songs, args.artist)


if __name__ == "__main__":
    main()