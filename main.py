from TextUtilities.indexer import Indexer
import GUI.MainWindow
from MainImplementation.GameSearcher import GameSearcher
from rich.console import Console
import os


if __name__ == "__main__":
    console = Console()
    threads = 10
    do_sentiment_analysis = True
    index_dir = "indexdir/base" if not do_sentiment_analysis else "indexdir/sentiment"
    if not os.path.exists("indexdir"):
        os.mkdir("indexdir")
    with console.status("[bold green]Creating index...") as status:
        main_idx, reviews_idx = Indexer.openIndex("Dataset", index_dir, console, do_sentiment_analysis, threads)
    searcher = GameSearcher(main_idx, reviews_idx, do_sentiment_analysis, sentiment_version="inavg")
    GUI.MainWindow.launchGui(searcher)

