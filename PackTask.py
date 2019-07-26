import DataBase
from Part import Part
import cv2


class PackTask:
    statuses = {'Detected': 0, 'Not detected': 1}

    def __init__(self, part: Part, amount):
        self.status = PackTask.statuses['Not detected']
        self.part = part
        self.amount = amount

    def is_detected(self):
        return self.status == PackTask.statuses['Detected']

    def set_status_as_detected(self):
        self.status = PackTask.statuses['Detected']

    def set_status_as_not_detected(self):
        self.status = PackTask.statuses['Not detected']

    @staticmethod
    def get_pack_tasks(db: DataBase, cur_time, packWorker):

        data = db.get_by_time_by_name(cur_time, packWorker)
        pack_tasks = []
        cur_task_time, next_task_time = None, None
        if data:
            cur_task_time = data[0]['Date']

        for item in data:
            if item['Date'] != cur_task_time:
                next_task_time = item['Date']
                break
            part = Part(item['Products'].split(',')[1], int(item['Length']), int(item['Depth']), int(item['Height']),
                        int(item['ProductsCount']))
            pack_tasks.append(PackTask(part, int(item['ProductsCount'])))

        return pack_tasks, f'{next_task_time}:00'
