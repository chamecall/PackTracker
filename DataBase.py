from tinydb import TinyDB, where
import logging
from os import path

class DataBase:
    def __init__(self):
        self.db_name = 'db.json'
        is_db_exist = path.exists(self.db_name)
        self.db = TinyDB(self.db_name)
        self.file_name = 'db.txt'
        self.encoding = 'utf-8'
        self.file = None
        self.sort_field = None
        if not is_db_exist:
            self.load_file_in_db()

    def load_file_in_db(self):
        try:
            with open(self.file_name, encoding=self.encoding) as self.file:
                for line in self.file:
                    data = line.split("\t")
                    # print(f"№Pack: {data[0]},")
                    # print(f" Date:{data[1]},")
                    # print(f" 'WorkerName': {data[2]},")
                    # print(f"'Products': {data[3]},")

                    # print(f"ProductsCount: {data[4]}, 'PlacesCount': {data[5]}, 'Multiplicity':{data[6]},")
                    # print(f"'Weight': {data[7]}, 'GrossWeight': {data[8]}, 'Volume': {data[9]}, 'Length':{data[10]}, 'Height': {data[11]},")
                    # print(f"'Depth': {data[12]}")
                    if len(data) == 13:
                        self.db.insert({'№Pack': data[0], 'Date': data[1], 'WorkerName': data[2], 'Products': data[3],
                                        'ProductsCount': data[4], 'PlacesCount': data[5], 'Multiplicity': data[6],
                                        'Weight': data[7], 'GrossWeight': data[8], 'Volume': data[9],
                                        'Length': data[10], 'Height': data[11],
                                        'Depth': data[12].split('\n')[0], 'Class': None})
                    else:
                        self.db.insert({'№Pack': data[0], 'Date': data[1], 'WorkerName': data[2], 'Products': data[3],
                                        'ProductsCount': data[4], 'PlacesCount': data[5], 'Multiplicity': data[6],
                                        'Weight': data[7], 'GrossWeight': data[8], 'Volume': data[9],
                                        'Length': data[10], 'Height': data[11],
                                        'Depth': data[12], 'Class': data[13].split('\n')[0]})
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
        data = self.db.search(where('Date') >= cur_time)
        return data

    def get_by_time_by_name(self, cur_time, name):
        data = []
        try:
            next_pack_row = self.db.get((where('WorkerName') == name) & (where('Date') >= cur_time))
            if next_pack_row != None:
                data = self.db.search((where('Date') >= next_pack_row['Date']) & (where('WorkerName') == name))
        except:
            print("Search Error")
        return data