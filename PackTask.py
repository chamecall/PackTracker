import DataBase
from Part import Part


class PackTask:
    def __init__(self, part: Part, amount):
        self.part = part
        self.amount = amount

    @staticmethod
    def get_pack_tasks(db: DataBase, cur_time, packWorker):
        data = db.get_by_time_by_name(cur_time, packWorker)
        pack_tasks = []
        next_task_time = data[0]['Date']
        for item in data:
            if item['Date'] != next_task_time:
                break
            part = Part(item['Products'].split(',')[1], int(item['Length']), int(item['Depth']), int(item['Height']), int(item['ProductsCount']))
            pack_tasks.append(PackTask(part, int(item['ProductsCount'])))

        return pack_tasks
