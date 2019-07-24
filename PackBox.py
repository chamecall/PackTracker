

class Packbox:
    def __init__(self, coords):
        self.coords = coords
        self.statuses = {'Unknown': 0, 'Empty': 1, 'PartsIn': 2, 'Closed': 3}
        self.status = self.statuses['Unknown']
