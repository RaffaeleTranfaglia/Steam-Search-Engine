from whoosh.qparser import MultifieldParser


class GameSearcher:
    def __init__(self, ix):
        self.ix = ix

    def search(self, queryText, fields, limit=10):
        searcher = self.ix.searcher()
        parser = MultifieldParser(fields, self.ix.schema)
        query = parser.parse(queryText)
        return searcher.search(query, limit=limit)
