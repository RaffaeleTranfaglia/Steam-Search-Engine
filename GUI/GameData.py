class GameData(object):
    def __init__(self, game_data):
        self.app_id = game_data['app_id']
        self.name = game_data['name']
        self.release_date = game_data['release_date']
        self.developer = game_data['developer']
        self.publisher = game_data['publisher']
        self.platforms = game_data['platforms']
        self.categories = game_data['categories']
        self.genres = game_data['genres']
        self.tags = game_data['tags']
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
            self.recommended_requirements = ' '
