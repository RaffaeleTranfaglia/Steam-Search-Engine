import nltk
from nltk.corpus import stopwords
import string

'''
import json
import os

folder_path = "Dataset/"

# iterate through all the files in the folder
for filename in os.listdir(folder_path):
    # join path folder with filename in order to get the relative path to the file
    file_path = os.path.join(folder_path, filename)
    
    # chek if the file is a JSON file
    if (file_path.endswith(".json")):
        with open(file_path, 'r') as game_file:
            # load JSON data
            game_data = json.load(game_file)
            
            # text preprocessing
            tokens = nltk.word_tokenize(game_data["name"])
            print(tokens)
'''

class analyzer:
    def _init_(self):
        pass
    
    @staticmethod
    def preprocessing(str):
        # tokenization
        tokens = nltk.word_tokenize(str)
        tokens1 = []
        for t in tokens:
            tokens1.extend(t.split('/'))
        
        # remove stopwords and convert to lowercase
        tokens2 = []
        # Get the default list of stopwords
        stop_words = set(stopwords.words('english'))
        # Add punctuation symbols to the list of stopwords
        stop_words.update(set(string.punctuation))
        for t in tokens1:
            if t not in stop_words:
                tokens2.append(t.lower())
        
        # handle tokens based on genres acronyms and lemmatize
        genres = {
            "fps" : ["first", "person", "shooter"],
            "tps" : ["third", "person", "shooter"],
            "mmo" : ["massive", "multiplayer", "online"],
            "jrpg" : ["japaese", "role", "playing", "game"],
            "rpg" : ["role", "playing", "game"],
            "moba" : ["multiplayer", "online", "battle", "arena"],
            "sim" : ["simulation"],
            "rts" : ["real", "time", "strategy"],
            "tbs" : ["turn", "based", "strategy"],
            "co-op" : ["cooperation"],
            "coop" : ["cooperation"],
            "sci-fi" : ["science", "fiction"],
            "scifi" : ["science", "fiction"]
        }
        tokens3 = []
        wnl = nltk.WordNetLemmatizer()
        for t in tokens2:
            if genres.get(t) != None:
                for tag in genres.get(t):
                    tokens3.append(wnl.lemmatize(tag))
            else:
                tokens3.append(wnl.lemmatize(t))
                
        # split on '-' and remove potential empty strings
        tokens4 = []
        for t in tokens3:
            filtered_list = [item for item in t.split('-') if item != ""]
            tokens4.extend(filtered_list)
                
        return tokens4
    
str = "Counter-Strike, Sci-fi CO-OP -c MMO/rpg games with weapons!"
res = analyzer.preprocessing(str)
print(res)
    