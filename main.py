from text_utilities.indexer import Indexer
import GUI.MainWindow

if __name__ == "__main__":
    ix = Indexer.openIndex("Dataset", "indexdir")
    GUI.MainWindow.launchGui()
