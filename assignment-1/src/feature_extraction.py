import spacy
import os 
import pandas as pd
import glob
import re
import argparse
import en_core_web_sm, en_core_web_md, en_core_web_lg  

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

#def import_model(spacy_model):
#    import spacy_model
#    nlp = spacy_model.load()
#
#    return print(nlp)


def main():
    args = get_arguments()
    nlp = import_model(args.model)

if __name__ == "__main__":
    main()