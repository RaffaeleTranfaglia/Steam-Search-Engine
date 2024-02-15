import json
import os, os.path
from whoosh.index import create_in, open_dir
from whoosh.fields import *
from TextUtilities.analyzer import CustomWhooshAnalyzer
import time
from transformers import pipeline
import concurrent.futures
from typing import List


'''
    A review which will be indexed.
'''
class ReviewDocument:
    def __init__(self, review_text, review_score):
        self.review_text = review_text
        self.review_score = review_score

    '''
        Compute sentiment values for the current review.
    '''
    def doSentimentAnalysis(self, classifier):
        return classifier(self.review_text)[0]



'''
    A game which will be indexed.
'''
class GameDocument:
    #inits from a json file
    def __init__(self, file_path, do_sentiment, classifier):
        with open(file_path, 'r', encoding='utf-8') as game_file:
            game_data = json.load(game_file)
            self.app_id = str(game_data["app_id"])
            self.name = game_data["name"]
            self.cgt = ';'.join(game_data["categories"]) + ";" + ';'.join(game_data["genres"]) + ";" + ';'.join(game_data["tags"])
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
            self.av_anger = -1
            self.av_disgust = -1
            self.av_fear = -1
            self.av_joy = -1
            self.av_neutral = -1
            self.av_sadness = -1
            self.av_surprise = -1
            self.inav_anger = -1
            self.inav_disgust = -1
            self.inav_fear = -1
            self.inav_joy = -1
            self.inav_neutral = -1
            self.inav_sadness = -1
            self.inav_surprise = -1
            for r in game_data["reviews"]:
                self.reviews.append(ReviewDocument(r["review_text"], r["review_score"]))

            '''
                Compute sentiment values for a game.
            '''
            if do_sentiment and classifier is not None:
                self.av_anger = 0
                self.av_disgust = 0
                self.av_fear = 0
                self.av_joy = 0
                self.av_neutral = 0
                self.av_sadness = 0
                self.av_surprise = 0
                self.inav_anger = 0
                self.inav_disgust = 0
                self.inav_fear = 0
                self.inav_joy = 0
                self.inav_neutral = 0
                self.inav_sadness = 0
                self.inav_surprise = 0
                inav_weights_sum = 0
                for r in self.reviews:
                    neutral = 0
                    anger = 0
                    disgust = 0
                    fear = 0
                    joy = 0
                    sadness = 0
                    surprise = 0
                    for s in r.doSentimentAnalysis(classifier):
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

                    self.av_anger += anger
                    self.av_disgust += disgust
                    self.av_fear += fear
                    self.av_joy += joy
                    self.av_neutral += neutral
                    self.av_sadness += sadness
                    self.av_surprise += surprise

                    #to ensure to not divide by 0
                    if neutral == 0:
                        neutral = 0.0000000001

                    self.inav_anger += (1 / neutral) * anger
                    self.inav_disgust += (1 / neutral) * disgust
                    self.inav_fear += (1 / neutral) * fear
                    self.inav_joy += (1 / neutral) * joy
                    self.inav_neutral += (1 / neutral) * neutral
                    self.inav_sadness += (1 / neutral) * sadness
                    self.inav_surprise += (1 / neutral) * surprise
                    inav_weights_sum += (1 / neutral)

                n_reviews = len(self.reviews) if len(self.reviews) > 0 else 1
                self.av_anger /= n_reviews
                self.av_disgust /= n_reviews
                self.av_fear /= n_reviews
                self.av_joy /= n_reviews
                self.av_neutral /= n_reviews
                self.av_sadness /= n_reviews
                self.av_surprise /= n_reviews
                inav_weights_sum = inav_weights_sum if inav_weights_sum > 0 else 1
                self.inav_anger /= inav_weights_sum
                self.inav_disgust /= inav_weights_sum
                self.inav_fear /= inav_weights_sum
                self.inav_joy /= inav_weights_sum
                self.inav_neutral /= inav_weights_sum
                self.inav_sadness /= inav_weights_sum
                self.inav_surprise /= inav_weights_sum
        print(f"finished loading Dataset/{self.app_id}.json")

    '''
        Print for testing purposes.
    '''
    def printSentiments(self):
        print(f"av_anger : {self.av_anger}")
        print(f"av_disgust : {self.av_disgust}")
        print(f"av_fear : {self.av_fear}")
        print(f"av_joy : {self.av_joy}")
        print(f"av_neutral : {self.av_neutral}")
        print(f"av_sadness : {self.av_sadness}")
        print(f"av_surprise : {self.av_surprise}")
        print("-----------------------------------")
        print(f"inav_anger : {self.inav_anger}")
        print(f"inav_disgust : {self.inav_disgust}")
        print(f"inav_fear : {self.inav_fear}")
        print(f"inav_joy : {self.inav_joy}")
        print(f"inav_neutral : {self.inav_neutral}")
        print(f"inav_sadness : {self.inav_sadness}")
        print(f"inav_surprise : {self.inav_surprise}")

'''
    Used by threads to call GameDocument constructor.
'''
def createGameDocument(file_path, do_sentiment=False, classifier=None):
    return GameDocument(file_path, do_sentiment, classifier)


'''
    Class that define methods to create an inverted index on the documents' collection.
'''
class Indexer:
    def __init__(self) -> None:
        pass
    
    '''
        Open the chose index if it exists, otherwise create the main index and the reviews' one.
    '''
    @staticmethod
    def openIndex(folder_path, folder_index, console, do_sentiment=False, worker_threads=4):
        if os.path.exists(folder_index):
            # If it exists, open the index
            return open_dir(folder_index + "/main_index"), open_dir(folder_index + "/reviews_index")
        
        # Else, create the folder and the index inside it
        os.mkdir(folder_index)
        os.mkdir(folder_index + "/main_index")
        os.mkdir(folder_index + "/reviews_index")
        mainix, reviewix = Indexer.indexing(folder_path, folder_index, do_sentiment, worker_threads)
        console.log(f"Index completed")
        return mainix, reviewix
        
    '''
        Create the indexes using the number of woriking threads specified by the user.
    '''
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
                        cgt=TEXT(stored=True, analyzer=customAnalyzer),
                        positive_ratings=STORED, negative_ratings=STORED,
                        price=NUMERIC(stored=True),
                        description=TEXT(stored=True, analyzer=customAnalyzer),
                        header_img=STORED,
                        minimum_requirements=STORED, recommended_requirements=STORED,
                        av_anger=NUMERIC(stored=True),
                        av_disgust=NUMERIC(stored=True),
                        av_fear=NUMERIC(stored=True),
                        av_joy=NUMERIC(stored=True),
                        av_neutral=NUMERIC(stored=True),
                        av_sadness=NUMERIC(stored=True),
                        av_surprise=NUMERIC(stored=True),
                        inav_anger=NUMERIC(stored=True),
                        inav_disgust=NUMERIC(stored=True),
                        inav_fear=NUMERIC(stored=True),
                        inav_joy=NUMERIC(stored=True),
                        inav_neutral=NUMERIC(stored=True),
                        inav_sadness=NUMERIC(stored=True),
                        inav_surprise=NUMERIC(stored=True))
        
        main_idx = create_in(folder_index + "/main_index", main_schema)
        main_writer = main_idx.writer()

        reviews_schema = Schema(app_id=ID(stored=True),
                                review_text=TEXT(stored=True, analyzer=customAnalyzer),
                                review_score=STORED)
        reviews_idx = create_in(folder_index + "/reviews_index", reviews_schema)
        reviews_writer = reviews_idx.writer()

        classifier = None
        # retirive AI model for sentiment analysis
        if do_sentiment:
            classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

        # instanciate and handle all the woriking threads
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
                                     av_anger=g.av_anger,
                                     av_disgust=g.av_disgust,
                                     av_fear=g.av_fear,
                                     av_joy=g.av_joy,
                                     av_neutral=g.av_neutral,
                                     av_sadness=g.av_sadness,
                                     av_surprise=g.av_surprise,
                                     inav_anger=g.inav_anger,
                                     inav_disgust=g.inav_disgust,
                                     inav_fear=g.inav_fear,
                                     inav_joy=g.inav_joy,
                                     inav_neutral=g.inav_neutral,
                                     inav_sadness=g.inav_sadness,
                                     inav_surprise=g.inav_surprise)

            for r in g.reviews:
                reviews_writer.add_document(app_id=g.app_id,
                                            review_text=r.review_text,
                                            review_score=r.review_score)

        print("starting writing at " + str(time.time() - start_time) + "s")
        main_writer.commit()
        reviews_writer.commit()
        print("finished index after " + str(time.time() - start_time) + "s")
        return main_idx, reviews_idx
