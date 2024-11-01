# Used for running evaluations for my report generates graphs from training statistics
# By Shaun Searle

import neatloader
import os
import copy

# requires pip
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":

    # Get maps from folder, returns set of maps excluding finish line example
    def get_maps():
        maps = os.listdir("./maps")
        maps.remove("map_finish_line.png")
        return maps

    # First evaluation that train from defaults
    def evaluation1():

        results = []

        maps = get_maps()
        for m in maps:
            map_path = "./maps/" + m

            loader.control_results = []

            for i in range(1, 10):

                loader.reset_defaults()  # Discards existing training data
                loader.train(20, True, False, map_path)

            results.append(loader.control_results)

        averaged_results = average_eval(results)

        maps = get_maps()
    
        gens = []
        lap_times = []
        
        for g, l in averaged_results:
            gens.append(g)
            lap_times.append(l)
        
        print(gens)
        print(lap_times)
        
        df = pd.DataFrame({"generations": gens, "lap_time(s)": lap_times}, index=maps)
        x = df.plot.bar(rot=0)
        
        x.set_title('Averaged new training per map')
        x.set_xlabel("Maps")
        x.set_ylabel("Generations & Lap time")
        
        for container in x.containers:
            x.bar_label(container)
        
        plt.show()

    # Second evaluation with seeded start
    def evaluation2():

        results = []

        loader.reset_defaults()
        loader.train(20, True, False, "./maps/map.png")
        check = copy.deepcopy(loader.Game.population)
        control_gens = check.generation
        
        maps = get_maps()
        for m in maps:
            map_path = "./maps/" + m

            loader.control_results = []

            for i in range(1, 10):

                loader.Game.population = copy.deepcopy(check)
                loader.train(20, True, False, map_path)
                
                # if loader.Game.population.generation - control_gens == 0: # incase control gens is zero
                #     break # if the network completes the map first try it is a waste of time to do 10 runs

            results.append(loader.control_results)

        averaged_results = average_eval(results, control_gens)

        maps = get_maps()
    
        gens = []
        lap_times = []
        
        for g, l in averaged_results:
            gens.append(g)
            lap_times.append(l)
        
        print(gens)
        print(lap_times)
        
        gens[0] += control_gens # We run run the first map that was trained on this is the ensure the generation count is correct
        
        df = pd.DataFrame({"generations": gens, "lap_time(s)": lap_times}, index=maps)
        x = df.plot.bar(rot=0)
        
        x.set_title('Averaged seeded training per map')
        x.set_xlabel("Maps")
        x.set_ylabel("Generations & Lap time")
        
        for container in x.containers:
            x.bar_label(container)
        
        plt.show()

    # Code duplication but is designed just for evauluation
    def average_eval(results_set, control_gens=0):
        
        averages_by_map = []
        average_gens = 0
        
        for m in results_set:
            gens = 0
            lap_time = 0

            for g, l in m:
                gens += g - control_gens
                lap_time += l
                
            if gens == 0:
                average_gens = 0
            else:
                average_gens = gens / len(m)
            averages_by_map.append([average_gens, lap_time / len(m)])
        return averages_by_map

    loader = neatloader.neatloader()
    
    # First evaluation that train from defaults
    evaluation1()
    # Second evaluation with seeded start
    evaluation2()
