from TextUtilities.indexer import Indexer
import GUI.MainWindow
from MainImplementation.GameSearcher import GameSearcher

if __name__ == "__main__":
    ix = Indexer.openIndex("Dataset", "indexdir")
    searcher = GameSearcher(ix)
    GUI.MainWindow.launchGui(searcher)