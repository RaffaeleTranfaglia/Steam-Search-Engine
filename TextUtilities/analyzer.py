import nltk
from nltk.corpus import stopwords, wordnet
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
            yield Token(text=word, pos=position, positions=True, chars=True, startchar=ord(word[0]),
                        endchar=ord(word[-1]))


# Class that implements all the methods to proccess a natural language text to create a list of tokens.
class TokenAnalyzer:
    def __init__(self) -> None:
        pass

    '''
        Preprocesses text and returns a list of tokens
        
        @param text: text to preprocess 
    '''
    @staticmethod
    def preprocessing(text):
        text = text.lower()
        # tokenization
        tokens = nltk.word_tokenize(text)
        tokens1 = []
        for t in tokens:
            tokens1.extend(t.split('/'))

        # handle tokens based on genres acronyms and lemmatize
        genres = {
            "fps": ["first", "person", "shooter"],
            "tps": ["third", "person", "shooter"],
            "mmo": ["massive", "multiplayer", "online"],
            "jrpg": ["japaese", "role", "playing", "game"],
            "crpg": ["classic", "role", "playing", "game"],
            "rpg": ["role", "playing", "game"],
            "moba": ["multiplayer", "online", "battle", "arena"],
            "sim": ["simulation"],
            "rts": ["real", "time", "strategy"],
            "tbs": ["turn", "based", "strategy"],
            "co-op": ["cooperation"],
            "coop": ["cooperation"],
            "sci-fi": ["science", "fiction"],
            "scifi": ["science", "fiction"]
        }
        tokens2 = []
        for t in tokens1:
            if genres.get(t) is not None:
                for tag in genres.get(t):
                    tokens2.append(tag)
            else:
                tokens2.append(t)

        # split on '-' and remove potential empty strings
        tokens3 = []
        for t in tokens2:
            filtered_list = [item for item in t.split('-') if item != ""]
            tokens3.extend(filtered_list)

        tokens3 = nltk.pos_tag(tokens3)

        # remove stopwords
        tokens4 = []
        # Get the default list of stopwords
        stop_words = set(stopwords.words('english'))
        # Add punctuation symbols to the list of stopwords
        stop_words.update(set(string.punctuation))
        for t in tokens3:
            if t[0] not in stop_words:
                if t[0].endswith('™') or t[0].endswith('®') or t[0].endswith('©'):
                    t = (t[0][:-1], t[1])
                if t[0] != "":
                    tokens4.append(t)

        wnl = nltk.WordNetLemmatizer()
        tokens4 = [(t[0], TokenAnalyzer.nltktags_to_wdntags(t[1])) for t in tokens4]
        result = []
        for tt in tokens4:
            result.append(wnl.lemmatize(word=tt[0], pos=tt[1]))

        return result

    '''
        Converts a nltk POS tag to a wordnet POS tag
        
        @param tag: tag to be converted
    '''
    @staticmethod
    def nltktags_to_wdntags(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
