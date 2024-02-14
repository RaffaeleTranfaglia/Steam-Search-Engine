from TextUtilities.indexer import Indexer
from TextUtilities.d2v import D2V
import os
import argparse
from rich.console import Console
import nltk
from transformers import pipeline

'''
    Download nltk corpora which are needed to the analyzer
'''
def downloadNLTKCorpus():
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("averaged_perceptron_tagger")

def setup():
    # defining the parameters of the program
    parser = argparse.ArgumentParser(description="Create all indexes (base, sentiment)")
    parser.add_argument("-t", "--threads",
                        dest="nThreads",
                        type=int,
                        default=4,
                        help="Number of threads used to create the indexs. Default number = 4.",
                        metavar=4)
    
    console = Console()
    # arguments parsing
    args = parser.parse_args()
    if (args.nThreads not in range(1, 11)):
        console.log("[red] The number of threads must be in the following range: 1-10")
        return
    
    # download ai model used for sentiment analysis
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    
    # create indexes
    if not os.path.exists("indexdir"):
        os.mkdir("indexdir")
    with console.status("[bold green]Creating base version index...") as status:
        Indexer.openIndex("Dataset", "indexdir/base", console, False, args.nThreads)
    with console.status("[bold green]Creating sentiment versions index...") as status:
        Indexer.openIndex("Dataset", "indexdir/sentiment", console, True, args.nThreads)
    console.log(f"All indexes are complete")
    # create d2v models
    with console.status("[bold green]Building and training D2V models...") as status:
        D2V.load_model("Dataset", "indexdir/d2v", args.nThreads)
    console.log(f"All models are trained and saved")

if __name__ == "__main__":
    downloadNLTKCorpus()
    setup()