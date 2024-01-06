import json
import os, os.path
from whoosh.index import create_in, open_dir
from whoosh.fields import *
from text_utilities.analyzer import CustomWhooshAnalyzer

# Class that define methods to create an inverted index on the documents' collection.
class Indexer:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def openIndex(folder_path, folder_index):
        if os.path.exists(folder_index):
            # If it exists, open the index
            return open_dir(folder_index)
        
        # Else, create the folder and the index inside it
        os.mkdir(folder_index)
        return Indexer.indexing(folder_path, folder_index)
        

    @staticmethod
    def indexing(folder_path, folder_index):
        customAnalyzer = CustomWhooshAnalyzer()
        schema = Schema(app_id=ID(stored=True), name=TEXT(stored=True, analyzer=customAnalyzer), 
                        release_date=STORED, developer=TEXT(stored=True, analyzer=CustomWhooshAnalyzer), 
                        publisher=TEXT(stored=True, analyzer=CustomWhooshAnalyzer), 
                        platforms=TEXT(stored=True, analyzer=CustomWhooshAnalyzer), categories=STORED, 
                        genres=STORED, tags=STORED, cgt=TEXT(analyzer=CustomWhooshAnalyzer), 
                        positive_ratings=STORED, negative_ratings=STORED, price=NUMERIC(stored=True), 
                        description=TEXT(stored=True, analyzer=CustomWhooshAnalyzer), header_img=STORED, 
                        minimum_requirements=STORED, recommended_requirements=STORED)
        ix = create_in(folder_index, schema)
        writer = ix.writer()
        
        # iterate through all the files in the folder
        for filename in os.listdir(folder_path):
            # join path folder with filename in order to get the relative path to the file
            file_path = os.path.join(folder_path, filename)
            
            # chek if the file is a JSON file
            if not (file_path.endswith(".json")):
                continue
                
            with open(file_path, 'r') as game_file:
                # load JSON data
                game_data = json.load(game_file)
                cgt = ';'.join(game_data["categories"]) + ';'.join(game_data["genres"]) + ';'.join(game_data["tags"])
                # add document to the index
                writer.add_document(app_id=game_data["app_id"], name=game_data["name"], 
                                    release_date=game_data["release_date"], 
                                    developer=';'.join(game_data["developer"]), 
                                    publisher=';'.join(game_data["publisher"]), 
                                    platforms=';'.join(game_data["platforms"]), 
                                    categories=';'.join(game_data["categories"]), 
                                    genres=';'.join(game_data["genres"]), 
                                    tags=';'.join(game_data["tags"]), 
                                    cgt=cgt, 
                                    positive_ratings=game_data["positive_ratings"], 
                                    negative_ratings=game_data["negative_ratings"], 
                                    price=game_data["price"], 
                                    description=game_data["description"], 
                                    header_img=game_data["header_img"], 
                                    minimum_requirements=game_data["minimum_requirements"], 
                                    recommended_requirements=game_data["recommended_requirements"])
        writer.commit()
        return ix