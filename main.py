from TextUtilities.indexer import Indexer
import GUI.MainWindow
from MainImplementation.GameSearcher import GameSearcher
from rich.console import Console

if __name__ == "__main__":
    console = Console()
    with console.status("[bold green]Creating index...") as status:
        main_idx, reviews_idx = Indexer.openIndex("Dataset", "indexdir", console)
    searcher = GameSearcher(main_idx)
    GUI.MainWindow.launchGui(searcher)