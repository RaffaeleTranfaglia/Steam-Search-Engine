from TextUtilities.indexer import Indexer
import GUI.MainWindow
from MainImplementation.GameSearcher import GameSearcher

if __name__ == "__main__":
    main_idx, reviews_idx = Indexer.openIndex("Dataset", "indexdir")
    searcher = GameSearcher(main_idx)
    GUI.MainWindow.launchGui(searcher)
