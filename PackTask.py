import DataBase
import Part

class PackTasks:
    def __init__(self):
        self.tasks = []




    @staticmethod
    def get_pack_tasks(db: DataBase, cur_time, packWorker):
        data = db.get_by_time_by_name(cur_time, packWorker)
        PackTasks = []
        next_task_time = data[0]['Date']
        for item in data:
            if item['Date'] != next_task_time:
                break
            PackTasks.append((item['Products'], item['ProductsCount'], (item['Length'], item['Depth'].split('\n')[1], item['Height'])))

        return PackTasks