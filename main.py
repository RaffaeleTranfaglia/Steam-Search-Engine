from TextUtilities.indexer import Indexer
import GUI.MainWindow
import os
import argparse
from MainImplementation.GameSearcher import GameSearcher
from rich.console import Console

def main():
    # defining the parameters of the program
    parser = argparse.ArgumentParser(description="Search engine based on Steam collection games")
    parser.add_argument("-s", "--sentiment",
                        dest = "vSentiment",
                        type = str,
                        default = "False",
                        help = "Version of sentiment analysis used to compute the average sentiment values "
                               "for every game: av = standard average, "
                               "inav = Inverted Neutral Weighted Average, "
                               "false = Sentiment Analysis not used.",
                        metavar = "inav")
    parser.add_argument("-t", "--threads",
                        dest ="nThreads",
                        type = int,
                        default = 4,
                        help = "Number of threads used to create the index. Default number = 4.",
                        metavar = 4)
    
    console = Console()
    # arguments parsing
    args = parser.parse_args()
    if (args.nThreads not in range(1, 11)):
        console.log("[red] The number of threads must be in the following range: 1-10")
        return
    
    if (args.vSentiment not in ["inav", "av", "false"]):
        console.log(f"[red] {args.vSentiment} is not a valid value for \"-s\" \"--sentiment\" option")
        return
    
    index_dir = "indexdir/base" if not args.vSentiment else "indexdir/sentiment"
    if not os.path.exists("indexdir"):
        os.mkdir("indexdir")
    with console.status("[bold green]Creating index...") as status:
        main_idx, reviews_idx = Indexer.openIndex("Dataset", index_dir, console, 
                                                  True if args.vSentiment != "false" else False, 
                                                  args.nThreads)
    searcher = GameSearcher(main_idx)
    GUI.MainWindow.launchGui(searcher)
    
if __name__ == "__main__":
    main()