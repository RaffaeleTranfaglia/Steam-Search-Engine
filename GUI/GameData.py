class GameData(object):
    def __init__(self, game_data):
        self.app_id = game_data['app_id']
        self.name = game_data['name']
        self.release_date = game_data['release_date']
        self.developer = game_data['developer'].replace(";", ", ")
        self.publisher = game_data['publisher'].replace(";", ", ")
        self.platforms = game_data['platforms'].replace(";", ", ")
        self.categories = game_data['categories'].replace(";", ", ")
        self.genres = game_data['genres'].replace(";", ", ")
        self.tags = game_data['tags'].replace(";", ", ")
        self.positive_ratings = game_data['positive_ratings']
        self.negative_ratings = game_data['negative_ratings']
        self.price = game_data['price']
        self.description = game_data['description']
        self.header_img = game_data['header_img']
        if 'minimum_requirements' in game_data.keys():
            self.minimum_requirements = game_data['minimum_requirements']
        else:
            self.minimum_requirements = ' '
        if 'recommended_requirements' in game_data.keys():
            self.recommended_requirements = game_data['recommended_requirements']
        else:
            self.recommended_requirements = 'Not provided'


class ReviewData:
    def __init__(self, review_data):
        self.app_id = review_data['app_id']
        self.review_text = review_data['review_text']
        self.review_score = review_data['review_score']
