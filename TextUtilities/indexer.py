import json
import os, os.path
from whoosh.index import create_in, open_dir
from whoosh.fields import *
from TextUtilities.analyzer import CustomWhooshAnalyzer
import time
from transformers import pipeline
import concurrent.futures
from typing import List


#a review which will be indexed
class ReviewDocument:
    def __init__(self, review_text, review_score):
        self.review_text = review_text
        self.review_score = review_score

    def doSentimentAnalysis(self, classifier):
        return classifier(self.review_text)[0]



#a game which will be indexed
class GameDocument:
    #inits from a json file
    def __init__(self, file_path, do_sentiment, classifier):
        with open(file_path, 'r', encoding='utf-8') as game_file:
            game_data = json.load(game_file)
            self.app_id = str(game_data["app_id"])
            self.name = game_data["name"]
            self.cgt = ';'.join(game_data["categories"]) + ';'.join(game_data["genres"]) + ';'.join(game_data["tags"])
            self.release_date = game_data["release_date"]
            self.developer = ';'.join(game_data["developer"])
            self.publisher = ';'.join(game_data["publisher"])
            self.platforms = ';'.join(game_data["platforms"])
            self.categories = ';'.join(game_data["categories"])
            self.genres = ';'.join(game_data["genres"])
            self.tags = ';'.join(game_data["tags"])
            self.positive_ratings = game_data["positive_ratings"]
            self.negative_ratings = game_data["negative_ratings"]
            self.price = game_data["price"]
            self.description = game_data["description"]
            self.header_img = game_data["header_img"]
            self.minimum_requirements = game_data["minimum_requirements"]
            self.recommended_requirements = game_data["recommended_requirements"]
            self.reviews = []
            self.anger = -1
            self.disgust = -1
            self.fear = -1
            self.joy = -1
            self.neutral = -1
            self.sadness = -1
            self.surprise = -1
            for r in game_data["reviews"]:
                self.reviews.append(ReviewDocument(r["review_text"], r["review_score"]))

            if do_sentiment and classifier is not None:
                self.anger = 0
                self.disgust = 0
                self.fear = 0
                self.joy = 0
                self.neutral = 0
                self.sadness = 0
                self.surprise = 0
                for r in self.reviews:
                    for s in r.doSentimentAnalysis(classifier):
                        s_type = s["label"]
                        if s_type == "anger":
                            self.anger += s["score"]
                        elif s_type == "disgust":
                            self.disgust += s["score"]
                        elif s_type == "fear":
                            self.fear += s["score"]
                        elif s_type == "joy":
                            self.joy += s["score"]
                        elif s_type == "neutral":
                            self.neutral += s["score"]
                        elif s_type == "sadness":
                            self.sadness += s["score"]
                        elif s_type == "surprise":
                            self.surprise += s["score"]
                n_reviews = len(self.reviews) if len(self.reviews) > 0 else 1
                self.anger /= n_reviews
                self.disgust /= n_reviews
                self.fear /= n_reviews
                self.joy /= n_reviews
                self.neutral /= n_reviews
                self.sadness /= n_reviews
                self.surprise /= n_reviews

        print(f"finished loading {self.app_id}")


def createGameDocument(file_path, do_sentiment=False, classifier=None):
    return GameDocument(file_path, do_sentiment, classifier)


# Class that define methods to create an inverted index on the documents' collection.
class Indexer:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def openIndex(folder_path, folder_index, console, do_sentiment=False, worker_threads=4):
        if os.path.exists(folder_index):
            # If it exists, open the index
            return open_dir(folder_index + "/main_index"), open_dir(folder_index + "/reviews_index")
        
        # Else, create the folder and the index inside it
        os.mkdir(folder_index)
        os.mkdir(folder_index + "/main_index")
        os.mkdir(folder_index + "/reviews_index")
        console.log(f"Index completed")
        return Indexer.indexing(folder_path, folder_index, do_sentiment, worker_threads)
        

    @staticmethod
    def indexing(folder_path, folder_index, do_sentiment, worker_threads):
        print("starting indexing")
        start_time = time.time()
        customAnalyzer = CustomWhooshAnalyzer()
        main_schema = Schema(app_id=ID(stored=True),
                        name=TEXT(stored=True, analyzer=customAnalyzer),
                        release_date=STORED,
                        developer=TEXT(stored=True, analyzer=customAnalyzer),
                        publisher=TEXT(stored=True, analyzer=customAnalyzer), 
                        platforms=TEXT(stored=True, analyzer=customAnalyzer),
                        categories=STORED,
                        genres=STORED, tags=STORED,
                        cgt=TEXT(analyzer=customAnalyzer),
                        positive_ratings=STORED, negative_ratings=STORED,
                        price=NUMERIC(stored=True),
                        description=TEXT(stored=True, analyzer=customAnalyzer),
                        header_img=STORED,
                        minimum_requirements=STORED, recommended_requirements=STORED,
                        anger=NUMERIC(stored=True),
                        disgust=NUMERIC(stored=True),
                        fear=NUMERIC(stored=True),
                        joy=NUMERIC(stored=True),
                        neutral=NUMERIC(stored=True),
                        sadness=NUMERIC(stored=True),
                        surprise=NUMERIC(stored=True)
                        )
        
        main_idx = create_in(folder_index + "/main_index", main_schema)
        main_writer = main_idx.writer()

        reviews_schema = Schema(app_id=ID(stored=True),
                                review_text=TEXT(stored=True, analyzer=customAnalyzer),
                                review_score=STORED)
        reviews_idx = create_in(folder_index + "/reviews_index", reviews_schema)
        reviews_writer = reviews_idx.writer()

        classifier = None
        if do_sentiment:
            classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

        executor = concurrent.futures.ThreadPoolExecutor(worker_threads)
        games: List[concurrent.futures.Future] = []

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if not (file_path.endswith(".json")):
                continue

            games.append(executor.submit(createGameDocument, file_path, do_sentiment, classifier))

        print("starting adding at " + str(time.time() - start_time) + "s")
        for future_game in games:
            g = future_game.result(timeout=None)
            main_writer.add_document(app_id=g.app_id,
                                     name=g.name,
                                     release_date=g.release_date,
                                     developer=g.developer,
                                     publisher=g.publisher,
                                     platforms=g.platforms,
                                     categories=g.categories,
                                     genres=g.genres,
                                     tags=g.tags,
                                     cgt=g.cgt,
                                     positive_ratings=g.positive_ratings,
                                     negative_ratings=g.negative_ratings,
                                     price=g.price,
                                     description=g.description,
                                     header_img=g.header_img,
                                     minimum_requirements=g.minimum_requirements,
                                     recommended_requirements=g.recommended_requirements,
                                     anger=g.anger,
                                     disgust=g.disgust,
                                     fear=g.fear,
                                     joy=g.joy,
                                     neutral=g.neutral,
                                     sadness=g.sadness,
                                     surprise=g.surprise)

            for r in g.reviews:
                reviews_writer.add_document(app_id=g.app_id,
                                            review_text=r.review_text,
                                            review_score=r.review_score)

        print("starting writing at " + str(time.time() - start_time) + "s")
        main_writer.commit()
        reviews_writer.commit()
        print("finished index after " + str(time.time() - start_time) + "s")
        return main_idx, reviews_idx
