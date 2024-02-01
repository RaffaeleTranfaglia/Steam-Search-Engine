from TextUtilities.indexer import Indexer
import GUI.MainWindow
from MainImplementation.GameSearcher import GameSearcher
from rich.console import Console

if __name__ == "__main__":
    console = Console()
    with console.status("[bold green]Creating index...") as status:
        ix = Indexer.openIndex("Dataset", "indexdir", console)
    searcher = GameSearcher(ix)
    GUI.MainWindow.launchGui(searcher)