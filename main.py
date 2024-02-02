from TextUtilities.indexer import Indexer
import GUI.MainWindow
from MainImplementation.GameSearcher import GameSearcher
from rich.console import Console
import os

if __name__ == "__main__":
    console = Console()
    threads = 8
    do_sentiment_analysis = True
    index_dir = "indexdir/base" if not do_sentiment_analysis else "indexdir/sentiment"
    if not os.path.exists("indexdir"):
        os.mkdir("indexdir")
    with console.status("[bold green]Creating index...") as status:
        main_idx, reviews_idx = Indexer.openIndex("Dataset", index_dir, console, do_sentiment=do_sentiment_analysis, worker_threads=8)
    searcher = GameSearcher(main_idx)
    GUI.MainWindow.launchGui(searcher)
