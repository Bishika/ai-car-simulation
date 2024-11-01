#Wrapper for genome
# By Shaun S

class genome:
    def __init__(self, genome, generations, trained_map, lap_time, id):
        self.genome = genome
        self.generations = generations
        self.map = trained_map
        self.lap_time = lap_time
        self.id = id
        
    def report_stats(self):
        print(f'ID: {self.id} Generations: {self.generations} Map: {self.map} Lap Time: {self.lap_time/1000: .2f}s')