import spacy
import os 
import pandas as pd
import glob
import re
import argparse
#import en_core_web_sm, en_core_web_md, en_core_web_lg  
import en_core_web_md

def get_arguments():
    """
    Allow for choosing another than the default spaCy model, used in feature extraction
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
                        "--model",
                        "-m", 
                        required = False,
                        default = en_core_web_md,
                        help="The artist should be given as a string")               
    args = parser.parse_args()
    return args

def rel_freq(count, len_doc): 
    """
    The function takes the number of the given POS in a document, and the total number of tokens in that document.
    It returns the rounded relative frequency (per 10,000 words) of the given POS of the document length.
    """
    rel = round((count/len_doc * 10000), 2)
    return rel


def unique_NE(doc):
    """ 
    The function takes a doc object, and returns the number of unique persons, locations and organisations.
    """
    enteties = []
    for e in doc.ents: 
        enteties.append((e.text, e.label_))
    ents_pd = pd.DataFrame(enteties, columns=["ent", "label"])
    ents_pd = ents_pd.drop_duplicates()
    unique_counts = ents_pd.value_counts(subset = "label")
    unique_labels = ['PERSON', 'LOC', 'ORG']
    unique_row = []
    for label in unique_labels:
        if label in (unique_counts.index):
            unique_row.append(unique_counts[label])
        else:
            unique_row.append(0)
    return unique_row

def clean_text(text, model):
    """
    The function takes a document and a spacy model, and returns the doc object, length and text name. 
    """
    text = text.read()
    text = re.sub(r'<*?>', '', text)
    doc = nlp(text)
    len_doc = len(doc)
    text_name = text.split("/")[-1]
    return doc, len_doc, text_name

def POS_count(doc):
    """
    The function takes a doc object and returns the number of nouns, verbs, adjectives and adverbs in it.
    """
    noun_count, verb_count, adj_count, adv_count = 0, 0, 0, 0   
    for token in doc:
        if token.pos_ == "NOUN":
            noun_count += 1
        if token.pos_ == "VERB":
            verb_count += 1
        if token.pos_ == "ADJ":
            adj_count +=1
        if token.pos_ == "ADV":
            adv_count +=1
    return noun_count, verb_count, adj_count, adv_count

def feature_extract(spacy_model):
    """
    The function takes the spacy model of choice to perform the feature extraction. 
    It runs through each subfolder in the USEcorpus, creating a .csv file for each, where 
    each text is represented in a row. For each text the name of the text, relative frequency
    of nouns, verbs, adjectives and adverbs, and the number of unique persons, locations and 
    organisations in the text is returned in the out folder.
    """
    for subfolder in sorted(os.listdir(os.path.join("..", "in", "USEcorpus"))):
        subfolder_path = os.path.join("..", "in", "USEcorpus", subfolder)
        out_df = pd.DataFrame(columns=("Filename", "RelFreq NOUN","RelFreq VERB","RelFreq ADJ",
        "RelFreq ADV","Unique PER","Unique LOC","Unique ORG"))
        outpath = os.path.join("..", "out", f"{subfolder}.csv")

        for file in sorted(glob.glob(os.path.join(subfolder_path, "*.txt"))):
            with open(file, "r", encoding="latin-1") as f:
                doc, len_doc, text_name = clean_text(f, spacy_model)

                noun_count, verb_count, adj_count, adv_count = POS_count(doc)         
                noun_rel, verb_rel, adj_rel, adv_rel = rel_freq(noun_count, len_doc), rel_freq(verb_count, len_doc), rel_freq(adj_count, len_doc), rel_freq(adv_count, len_doc)
                per, loc, org = unique_NE(doc)
        
                file_row = [text_name, noun_rel, verb_rel, adj_rel, adv_rel, per, loc, org]
                out_df.loc[len(out_df)] = file_row
        out_df.to_csv(outpath)
        print(outpath)                    

def main():
    args = get_arguments()
    feature_extract(args.model)

if __name__ == "__main__":
    main()