import os
import pandas as pd 
import gensim.downloader as api
from collections import Counter
import string
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


def artist_find(artist_input):
    """
    The function loads in the Spotify Million Song Dataset and takes a string as input, 
    if the string matches with an artist name in the corpus, a list with the given 
    artist's songs is returned.
    """
    filepath = os.path.join("in", "Spotify Million Song Dataset_exported.csv")
    songs = pd.read_csv(filepath)

    artist = artist_input.lower()
    songs["artist"] = (songs["artist"]).str.lower()

    if (songs['artist'].eq(artist).any()):
        artist_corpus = songs[songs["artist"] == artist]
        artist_songs = list(artist_corpus["text"])
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

def clean_tokens(text):
    """
    The function takes a text as an input tokenizes it, makes it lowercase and 
    strips the puctuation from all words. It returns a list of the cleaned tokens.
    """
    tokens = text.lower().split()
    tokens_cleaned = []
    for token in tokens: 
        strip_punct = token.strip(string.punctuation)
        if not strip_punct == "":
            tokens_cleaned.append(strip_punct)
    return tokens_cleaned

def keyword_count(keyword, song_texts, artist_input):
    """
    The function takes a keyword and a list of song texts, it returns the percentage 
    of the given songs that contains any of the 10 words mostly related to the keyword,
    and the keyword, of the number of songs from that artist. 
    """
    query_c = Counter(query_expansion(keyword))
    song_count = 0

    for text in range(len(song_texts)):
        tokens_cleaned = clean_tokens(song_texts[text])       
        tokens_c = Counter(tokens_cleaned)
        keyword_count = (tokens_c & query_c).total()
        if keyword_count > 0: 
            song_count +=1

    songs_perc = round(((song_count/len(song_texts))*100), 2)

    return print(f"{songs_perc}% of {artist_input}'s songs contains words related to {keyword}")    

def save_output(keyword):
    keywords_output = open("keywords.txt", "a")
    keywords_output.write(query_expansion(keyword))
    keywords_output.close()


def main():
    args = get_arguments()
    artist_songs = artist_find(args.artist)
    keyword_count(args.keyword, artist_songs, args.artist)


if __name__ == "__main__":
    main()