import json
import time
import gensim
from TextUtilities.analyzer import TokenAnalyzer
import os
from gensim.models.doc2vec import Doc2Vec


class D2V:
    def __init__(self) -> None:
        pass

    @staticmethod
    def load_model(ds_folder_path, models_folder_path, worker_threads=4):
        if os.path.exists(models_folder_path):
            models = {
                "name": Doc2Vec.load(os.path.join(models_folder_path, "d2v_name.model")),
                "description": Doc2Vec.load(os.path.join(models_folder_path, "d2v_description.model")),
                "developer": Doc2Vec.load(os.path.join(models_folder_path, "d2v_developer.model")),
                "publisher": Doc2Vec.load(os.path.join(models_folder_path, "d2v_publisher.model")),
                "platforms": Doc2Vec.load(os.path.join(models_folder_path, "d2v_platforms.model")),
                "cgt": Doc2Vec.load(os.path.join(models_folder_path, "d2v_cgt.model"))
            }
            games = os.listdir(ds_folder_path)
            games.sort()
            i_to_fp = {}
            for i, g in enumerate(games):
                if not g.endswith(".json"):
                    continue
                i_to_fp[i] = os.path.join(ds_folder_path, g)
            return models, i_to_fp

        os.mkdir(models_folder_path)
        models, i_to_fp = D2V.train(ds_folder_path, worker_threads)
        for key in models:
            models[key].save(os.path.join(models_folder_path, "d2v_" + key + ".model"))
        return models, i_to_fp

    @staticmethod
    def train(ds_folder_path, worker_threads):
        start_time = time.time()
        vector_size = 55
        min_count = 2
        epochs = 1000
        window = 2
        models = {
            "name": Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs, seed=1, workers=worker_threads, window=window),
            "description": Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs, seed=1, workers=worker_threads, window=window),
            "developer": Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs, seed=1, workers=worker_threads, window=window),
            "publisher": Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs, seed=1, workers=worker_threads, window=window),
            "platforms": Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs, seed=1, workers=worker_threads, window=window),
            "cgt": Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs, seed=1, workers=worker_threads, window=window),
        }

        corpus, i_to_fp = D2V.load_corpus(ds_folder_path)
        print(f"loaded corpus at {time.time() - start_time}s")

        for key in models:
            D2V.build_and_train(models[key], corpus[key])
            print(f"finished building and training {key} at {time.time() - start_time}s")

        return models, i_to_fp


    @staticmethod
    def build_and_train(model, corpus):
        model.build_vocab(corpus)
        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

    @staticmethod
    def load_corpus(ds_folder_path):
        games = os.listdir(ds_folder_path)
        games.sort()
        corpus = {
            "name": [],
            "description": [],
            "developer": [],
            "publisher": [],
            "platforms": [],
            "cgt": []
        }
        i_to_fp = {}
        for i, f in enumerate(games):
            if not f.endswith(".json"):
                continue

            fp = os.path.join(ds_folder_path, f)
            with open(fp, 'r', encoding='utf-8') as game_file:
                raw_data = json.load(game_file)
                game_data = {
                    "app_id": raw_data["app_id"],
                    "name": raw_data["name"],
                    "description": raw_data["description"],
                    "developer": " ".join(raw_data["developer"]),
                    "publisher": " ".join(raw_data["publisher"]),
                    "platforms": " ".join(raw_data["platforms"]),
                    "cgt": " ".join(raw_data["categories"]) + " " + " ".join(raw_data["genres"]) + " " + " ".join(raw_data["tags"])
                }
                for key in corpus:
                    corpus[key].append(gensim.models.doc2vec.TaggedDocument(words=TokenAnalyzer.preprocessing(game_data[key]), tags=[i]))

                i_to_fp[i] = fp
        return corpus, i_to_fp

