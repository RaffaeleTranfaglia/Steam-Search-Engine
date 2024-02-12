import json
import math
import os.path

from whoosh.qparser import MultifieldParser, QueryParser
from transformers import pipeline
import re
from typing import List, Tuple
from whoosh.searching import Hit
import gensim
from gensim.models.doc2vec import Doc2Vec
from TextUtilities.analyzer import TokenAnalyzer


def cosineSimilarity(sv1, sv2):
    dot_p = 0
    mod_sv1 = 0
    mod_sv2 = 0
    for i in range(0, 7):
        dot_p += sv1[i] * sv2[i]
        mod_sv1 += sv1[i] ** 2
        mod_sv2 += sv2[i] ** 2

    return dot_p / (math.sqrt(mod_sv1) * math.sqrt(mod_sv2)) if mod_sv1 != 0 and mod_sv2 != 0 else 0


class GameSearcher:
    def __init__(self, main_idx, reviews_idx, do_sentiment=False, sentiment_version=None, d2v=False, d2v_models=None, d2v_i_to_fp=None, dataset_path=None):
        self.main_idx = main_idx
        self.reviews_idx = reviews_idx
        self.do_sentiment = do_sentiment
        self.sentiment_version = None
        if self.do_sentiment:
            self.sentiment_version = sentiment_version
            self.classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
        self.d2v = d2v
        self.d2v_models = d2v_models
        self.d2v_i_to_fp = d2v_i_to_fp
        self.dataset_path = dataset_path

    def search(self, queryText: str, fields, limit=10):
        if self.d2v:
            return self.searchD2V(queryText, fields, limit)
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
            result_order: List[Tuple[Hit, float]] = []
            result = searcher.search(query, limit=None)
            for g in result:
                gsentv = [g[sent_prefix + 'anger'], g[sent_prefix + 'disgust'], g[sent_prefix + 'fear'], g[sent_prefix + 'joy'], g[sent_prefix + 'neutral'], g[sent_prefix + 'sadness'], g[sent_prefix + 'surprise']]

                revs = g["positive_ratings"] + g["negative_ratings"]
                if revs <= 1:
                    revs = 2

                k = 400
                result_order.append((g, cosineSimilarity(qsentv, gsentv) * math.exp(-k/(revs*math.log(revs, 10)))))

            result_order.sort(key=lambda r: r[1], reverse=True)
            result: List[Hit] = []
            for r in result_order:
                result.append(r[0])

            return result[:limit]

    def getGameReviews(self, app_id, limit=None):
        if self.d2v:
            reviews = []
            with open(os.path.join(self.dataset_path, str(app_id) + ".json"), 'r', encoding='utf-8') as game_file:
                game_data = json.load(game_file)
                for r in game_data["reviews"]:
                    reviews.append({"app_id": app_id, "review_text": r["review_text"], "review_score": r["review_score"]})
            return reviews
        searcher = self.reviews_idx.searcher()
        parser = QueryParser("app_id", schema=self.reviews_idx.schema)
        query = parser.parse(app_id)
        return searcher.search(query, limit=limit)


    def searchD2VSingleField(self, q, n, field):
        qv = self.d2v_models[field].infer_vector(q)
        sims = self.d2v_models[field].dv.most_similar([qv], topn=n)
        r = []
        for s in sims:
            r.append((self.d2v_i_to_fp[s[0]], s[1]))
        return r

    def searchD2V(self, queryText: str, fields, limit=10):
        match = re.search(r'sentiment\[(.*?)\]', queryText)
        if match:
            queryText = re.sub(r'\\sentiment\[(.*?)\]', '', queryText)
        qv = TokenAnalyzer.preprocessing(queryText)
        singular_results = []
        for f in fields:
            rs = self.searchD2VSingleField(qv, limit, f)
            for r in rs:
                singular_results.append({"app_id": r[0], "sim": r[1]})

        appids = set([r["app_id"] for r in singular_results])
        appids_sims = []
        for a in appids:
            s = 0
            for r in singular_results:
                if r["app_id"] == a:
                    s += r["sim"]
            s /= len(fields)
            appids_sims.append((a, s))
        appids_sims.sort(key=lambda a: a[1], reverse=True)
        result = []
        for f in appids_sims[:limit]:
            with open(f[0], 'r', encoding='utf-8') as game_file:
                game_data = json.load(game_file)
                del game_data["reviews"]
                game_data["app_id"] = str(game_data["app_id"])
                game_data["developer"] = ";".join(game_data["developer"])
                game_data["publisher"] = ";".join(game_data["publisher"])
                game_data["platforms"] = ";".join(game_data["platforms"])
                game_data["categories"] = ";".join(game_data["categories"])
                game_data["genres"] = ";".join(game_data["genres"])
                game_data["tags"] = ";".join(game_data["tags"])
                if game_data["recommended_requirements"] is None:
                    del game_data["recommended_requirements"]
                if game_data["minimum_requirements"] is None:
                    del game_data["minimum_requirements"]
                result.append(game_data)
        return result