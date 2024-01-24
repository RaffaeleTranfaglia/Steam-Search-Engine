import nltk
from nltk.corpus import stopwords
import string
from whoosh.analysis import Analyzer, Token

# Class that set whoosh tokenizer to the custom tokenizer.
class CustomWhooshAnalyzer(Analyzer):
    def __init__(self):
        super().__init__()
        
    def __call__(self, text, positions=False, chars=False, keeporiginal=False, removestops=True, 
                 start_pos=0, start_char=0, mode='', **kwargs):
        # Call the custom tokenizer
        words = TokenAnalyzer.preprocessing(text)

        for position, word in enumerate(words, start=start_pos):
            yield Token(text=word, pos=position)

# Class that implements all the methods to proccess a natural language text to create a list of tokens.
class TokenAnalyzer:
    def __init__(self) -> None:
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
                if t.endswith('™'):
                    t = t[:-1]
                tokens2.append(t.lower())
        
        # handle tokens based on genres acronyms and lemmatize
        genres = {
            "fps" : ["first", "person", "shooter"],
            "tps" : ["third", "person", "shooter"],
            "mmo" : ["massive", "multiplayer", "online"],
            "jrpg" : ["japaese", "role", "playing", "game"],
            "crpg" : ["classic", "role", "playing", "game"],
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