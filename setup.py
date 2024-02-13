from TextUtilities.indexer import Indexer
import os
import argparse
from MainImplementation.GameSearcher import GameSearcher
from rich.console import Console
import nltk

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
    
    if not os.path.exists("indexdir"):
        os.mkdir("indexdir")
    with console.status("[bold green]Creating base version index...") as status:
        Indexer.openIndex("Dataset", "indexdir/base", console, False, args.nThreads)
    with console.status("[bold green]Creating sentiment versions index...") as status:
        Indexer.openIndex("Dataset", "indexdir/sentiment", console, True, args.nThreads)
    console.log(f"All indexes are complete")

if __name__ == "__main__":
    downloadNLTKCorpus()
    setup()