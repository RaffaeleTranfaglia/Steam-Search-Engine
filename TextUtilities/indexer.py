import json
import os, os.path
from whoosh.index import create_in, open_dir
from whoosh.fields import *
from TextUtilities.analyzer import CustomWhooshAnalyzer
import time
from transformers import pipeline

# Class that define methods to create an inverted index on the documents' collection.
class Indexer:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def openIndex(folder_path, folder_index):
        if os.path.exists(folder_index):
            # If it exists, open the index
            return open_dir(folder_index + "/main_index"), open_dir(folder_index + "/reviews_index")
        
        # Else, create the folder and the index inside it
        os.mkdir(folder_index)
        os.mkdir(folder_index + "/main_index")
        os.mkdir(folder_index + "/reviews_index")
        console.log(f"Index completed")
        return Indexer.indexing(folder_path, folder_index)
        

    @staticmethod
    def indexing(folder_path, folder_index):
        print("starting indexing")
        start_time = time.time()
        customAnalyzer = CustomWhooshAnalyzer()
        main_schema = Schema(app_id=ID(stored=True), name=TEXT(stored=True, analyzer=customAnalyzer),
                        release_date=STORED, developer=TEXT(stored=True, analyzer=customAnalyzer), 
                        publisher=TEXT(stored=True, analyzer=customAnalyzer), 
                        platforms=TEXT(stored=True, analyzer=customAnalyzer), categories=STORED, 
                        genres=STORED, tags=STORED, cgt=TEXT(analyzer=customAnalyzer), 
                        positive_ratings=STORED, negative_ratings=STORED, price=NUMERIC(stored=True), 
                        description=TEXT(stored=True, analyzer=customAnalyzer), header_img=STORED, 
                        minimum_requirements=STORED, recommended_requirements=STORED)
        
        main_idx = create_in(folder_index + "/main_index", main_schema)
        main_writer = main_idx.writer()

        reviews_schema = Schema(app_id=ID(stored=True),
                                review_text=TEXT(stored=True, analyzer=customAnalyzer),
                                review_score=STORED,
                                anger=NUMERIC(stored=True),
                                disgust=NUMERIC(stored=True),
                                fear=NUMERIC(stored=True),
                                joy=NUMERIC(stored=True),
                                neutral=NUMERIC(stored=True),
                                sadness=NUMERIC(stored=True),
                                surprise=NUMERIC(stored=True))
        reviews_idx = create_in(folder_index + "/reviews_index", reviews_schema)
        reviews_writer = reviews_idx.writer()

        classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

        # iterate through all the files in the folder
        for filename in os.listdir(folder_path):
            # join path folder with filename in order to get the relative path to the file
            file_path = os.path.join(folder_path, filename)
            
            # chek if the file is a JSON file
            if not (file_path.endswith(".json")):
                continue
                
            with open(file_path, 'r', encoding='utf-8') as game_file:
                # load JSON data
                game_data = json.load(game_file)
                cgt = ';'.join(game_data["categories"]) + ';'.join(game_data["genres"]) + ';'.join(game_data["tags"])
                appid = str(game_data["app_id"])
                # add document to the index
                main_writer.add_document(app_id=appid, name=game_data["name"],
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

                for r in game_data["reviews"]:
                    anger = 0
                    disgust = 0
                    fear = 0
                    joy = 0
                    neutral = 0
                    sadness = 0
                    surprise = 0
                    try:
                        sentiments = classifier(r["review_text"])
                        for sent in sentiments:
                            for s in sent:
                                s_type = s["label"]
                                if s_type == "anger":
                                    anger = s["score"]
                                elif s_type == "disgust":
                                    disgust = s["score"]
                                elif s_type == "fear":
                                    fear = s["score"]
                                elif s_type == "joy":
                                    joy = s["score"]
                                elif s_type == "neutral":
                                    neutral = s["score"]
                                elif s_type == "sadness":
                                    sadness = s["score"]
                                elif s_type == "surprise":
                                    surprise = s["score"]
                    except Exception as e:
                        print(str(type(e)) + " caused by review of " + appid + " : " + r["review_text"])

                    reviews_writer.add_document(app_id=appid,
                                                review_text=r["review_text"],
                                                review_score=r["review_score"],
                                                anger=anger,
                                                disgust=disgust,
                                                fear=fear,
                                                joy=joy,
                                                neutral=neutral,
                                                sadness=sadness,
                                                surprise=surprise)

        print("starting writing at " + str(time.time() - start_time) + "s")
        main_writer.commit()
        reviews_writer.commit()
        print("finished index after " + str(time.time() - start_time) + "s")
        return main_idx, reviews_idx
