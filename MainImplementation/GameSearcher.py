import math
from whoosh.qparser import MultifieldParser, QueryParser
from transformers import pipeline
import re
from typing import List, Tuple
from whoosh.searching import Hit

'''
    Compute cosine similatiry between two vectors.
'''
def cosineSimilarity(sv1, sv2):
    dot_p = 0
    mod_sv1 = 0
    mod_sv2 = 0
    for i in range(0, 7):
        dot_p += sv1[i] * sv2[i]
        mod_sv1 += sv1[i] ** 2
        mod_sv2 += sv2[i] ** 2

    return dot_p / (math.sqrt(mod_sv1) * math.sqrt(mod_sv2)) if mod_sv1 != 0 and mod_sv2 != 0 else 0

'''
    Handle the searches.
'''
class GameSearcher:
    def __init__(self, main_idx, reviews_idx, do_sentiment=False, sentiment_version=None):
        self.main_idx = main_idx
        self.reviews_idx = reviews_idx
        self.do_sentiment = do_sentiment
        self.sentiment_version = None
        if self.do_sentiment:
            self.sentiment_version = sentiment_version
            self.classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

    '''
        Search execution.
    '''
    def search(self, queryText: str, fields, limit=20):
        searcher = self.main_idx.searcher()
        parser = MultifieldParser(fields, self.main_idx.schema)

        match = re.search(r'sentiment\[(.*?)\]', queryText)
        qsentv = []
        sent_prefix = ""
        doing_sentiment_query = False
        if match:
            sentiment_string = match.group(1)
            queryText = re.sub(r'\\sentiment\[(.*?)\]', '', queryText)

            if self.do_sentiment:
                doing_sentiment_query = True
                sentiments = self.classifier(sentiment_string)[0]
                anger = 0
                disgust = 0
                fear = 0
                joy = 0
                neutral = 0
                sadness = 0
                surprise = 0
                for s in sentiments:
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

                qsentv = [anger, disgust, fear, joy, neutral, sadness, surprise]

                sent_prefix = "inav_"
                if self.sentiment_version == "av":
                    sent_prefix = "av_"

        query = parser.parse(queryText)

        if not doing_sentiment_query:
            return searcher.search(query, limit=limit)
        else:
            # Hit is a dictionary returned by the indexer, its structure is equivalent to the schema
            result_order: List[Tuple[Hit, float]] = []
            result = searcher.search(query, limit=None)
            for g in result:
                gsentv = [g[sent_prefix + 'anger'], g[sent_prefix + 'disgust'], g[sent_prefix + 'fear'], g[sent_prefix + 'joy'], g[sent_prefix + 'neutral'], g[sent_prefix + 'sadness'], g[sent_prefix + 'surprise']]

                # weight funcuntion
                revs = g["positive_ratings"] + g["negative_ratings"]
                if revs <= 1:
                    revs = 2

                k = 400
                result_order.append((g, cosineSimilarity(qsentv, gsentv) * math.exp(-k/(revs*math.log(revs,10)))))

            result_order.sort(key=lambda r: r[1], reverse=True)
            result: List[Hit] = []
            for r in result_order:
                result.append(r[0])

            return result[:limit]

    '''
        Return a reviews list given the app_id.
    '''
    def getGameReviews(self, app_id, limit=None):
        searcher = self.reviews_idx.searcher()
        parser = QueryParser("app_id", schema=self.reviews_idx.schema)
        query = parser.parse(app_id)
        return searcher.search(query, limit=limit)
