from tinydb import TinyDB, where
import logging

class DataBase:
    def __init__(self):
        self.db_name = 'db.json'
        self.db = TinyDB(self.db_name)
        self.file_name = 'db.txt'
        self.encoding = 'utf-8'
        self.file = None
        self.sort_field = None
        self.load_file_in_db()

    def load_file_in_db(self):
        try:
            with open(self.file_name, encoding=self.encoding) as self.file:
                for line in self.file:
                    data = line.split("\t")
                    self.db.insert({'№Pack': data[0], 'Date': data[1], 'WorkerName': data[2], 'Products': data[3],
                                    'ProductsCount': data[4], 'PlacesCount': data[5], 'Multiplicity': data[6],
                                    'Weight': data[7], 'GrossWeight': data[8], 'Volume': data[9], 'Length': data[10], 'Height': data[11],
                                    'Depth': data[12]})
                self.file.close()
        except IOError:
            print(f'File {self.file_name} not found.')

    def print(self):
        for item in self.db.all():
            print(item)

    def clear(self):
        self.db.purge()

    # return needed list from db
    def get_by_name(self, name):
        data = self.db.search(where('WorkerName') == name)
        return data

    def get_by_time(self, cur_time):
        data = self.db.search(where('Date') > cur_time)
        return data

    def get_by_time_by_name(self, cur_time, name):
        data = []
        try:
            next_pack_row = self.db.get((where('WorkerName') == name) & (where('Date') > cur_time))
            if (next_pack_row != None):
                data = self.db.search((where('Date') >= next_pack_row['Date']) & (where('WorkerName') == name))
        except:
            print("Search Error")
        return data