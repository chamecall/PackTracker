import DataBase
class PackTasks():
    @staticmethod
    def get_pack_tasks(db: DataBase, cur_time, packWorker):
        data = db.get_by_time_by_name(cur_time, packWorker)
        PackTasks = []
        next_task_time = data[0]['Date']
        for item in data:
            print(item['Date'])
            if item['Date'] != next_task_time:
                break
            print(item)
            PackTasks.append((item['Products'], item['ProductsCount'], (item['Length'], item['Height'], item['Depth'])))

        return PackTasks