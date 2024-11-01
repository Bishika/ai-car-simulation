# Main program - By Shaun S
# Handles menu loading and handling interactions with NeatGame
# Further Code refactoring required, designed for demo use and data collection

# Needs Pip Install
import neatgame

# Included / Default modules
import pickle
import os
import genome

# Handler for other modules - Manages Set of genomes
class neatloader:

    def __init__(self):
        self.control_map = "./maps/map.png"

        # Manually specified for report data gathering
        self.maps = [
            "./maps/map1.png",
            "./maps/map2.png",
            "./maps/map3.png",
            "./maps/map4.png",
            "./maps/map5.png",
        ]

        # Used to hold and sum various generations and lap times
        self.control_results = []
        self.control_results_set = []

        self.control_generations = 0  # Generations it takes control network to train
        self.Game = neatgame.neatgame(10, True, self.control_map)

        # All genomes are loaded into and saved from this
        self.genome_set = [genome.genome]

    # Save Set of top genomes
    def save_genomes(self):
        with open("./save/genomes_all", "wb") as output:
            pickle.dump(self.genome_set, output, 1)
            print("All genomes exported to ./saves/genomes_all")

    # Load Set of all top genomes

    # Load saved genomes from file
    def load_genomes(self):
        try:
            with open("./save/genomes_all", "rb") as f:
                self.genome_set = pickle.load(f)

            print("Startup - Genomes Loaded")
        except OSError as e:
            # Do nothing first run/File missing
            empty = []

    # Iterates genomes set and reports their statistics ordered by lap times ascending
    def report_genomes(self):
        self.genome_set.sort(key=lambda ge: ge.lap_time, reverse=True)
        for g in self.genome_set:
            g.report_stats()

    # Resets the current neatgame instance to discard training data
    def reset_defaults(self):
        self.Game = neatgame.neatgame(10)  # Ensure Fresh Game

    # Train network
    def train(
        self,
        generations_to_train=10,
        solve_for_solution=True,
        save_solution=False,
        map=None,
    ):

        if map == None:
            map = self.control_map

        self.Game.cur_map = map

        self.Game.solve_for_solution = solve_for_solution
        # Train control model
        self.Game.population.run(self.Game.run_simulation, generations_to_train)
        # Append training generations and lap time in seconds
        self.control_results.append(
            [self.Game.population.generation, self.Game.best_time / 1000]
        )
        self.control_generations = self.Game.population.generation

        if save_solution:
            # Save Control Model
            self.Game.save_checkpoint()
            temp = self.Game.save_best_genome(
                len(self.genome_set) + 1
            )  # Hacky way to set ID and ensure it is unique for now
            self.genome_set.append(temp)

    # Averages and returns a set of training data
    def average_control(self, control_results):
        # Tally control
        temp_gen = 0
        temp_lap_time = 0
        for g, l in control_results:
            temp_gen += g
            temp_lap_time += l

        print(
            "Average Generations to train control: "
            + str(temp_gen / len(control_results))
        )
        print(
            "Average lap time during control training: "
            + str(temp_lap_time / len(control_results))
        )
        print(
            "Total times trained: "
            + str(len(control_results))
            + " Overall Generations trained: "
            + str(temp_gen)
        )

        return [temp_gen / len(control_results), temp_lap_time / len(control_results)]


# useful functions foor menu and operation
if __name__ == "__main__":

    # Get maps from folder, returns set of maps excluding finish line example
    def get_maps():
        maps = os.listdir("./maps")
        maps.remove("map_finish_line.png")
        return maps

    # Gets all checkpoint names from folder
    def get_checkpoints():
        checkpoint = os.listdir("./checkpoints")
        return checkpoint

    # Presents and handles user input for loading checkpoints
    def checkpoint_load():

        print(
            "\nAll Saved Checkpoints - Select a checkpoint by number then use train network with keep. c to cancel\n\n"
        )

        checkpoints = get_checkpoints()
        for c in range(1, len(checkpoints)):
            print(f"{c}. {checkpoints[c]}")

        valid = False
        while valid == False:
            try:
                inp = input("\nSelect Checkpoint ID to Load: ")

                if inp == "c":
                    valid = True
                    break

                loader.Game.population = loader.Game.cur_checkpoint.restore_checkpoint(
                    "./checkpoints/" + checkpoints[int(inp)]
                )  # Adjust to zero index
                loader.Game.reattatch_reporters()
                print(
                    f"Loaded: {checkpoints[int(inp)]} - Use keep in train network to resume training\n"
                )
                valid = True
            except ValueError as e:
                print("Invalid Input: Try again!")

    # Handles selection for single genome replay
    def genome_replay():
        if len(loader.genome_set) == 0:
            print("No genomes saved! Train a new network and select save.")
        else:
            print("All Genomes sorted by best lap time - Not sorted by map")
            loader.report_genomes()
            valid = False
            while valid == False:
                try:
                    inp = input("Select Genome ID to replay: ")

                    for g in loader.genome_set:
                        if g.id == int(inp):
                            loader.Game.cur_map = g.map
                            loader.Game.replay_genome(g.genome, g.lap_time)
                    valid = True
                except ValueError as e:
                    print("Invalid Input: Try again!")

    # Prints maps on one line
    def print_maps():
        print("\n\nAvailable maps: ")
        msg = ""
        for m in maps:
            msg += m + ", "
        print(msg)

    # Training related user input
    def train_menu():
        print_maps()
        correct = False
        while not correct:
            print(
                "\nUsage: map | number of generations(Max 1000) | train or solve | save or no | new or keep"
            )
            print(
                'Example: map1.png 10 solve save new \n "train" trains for the number of generations specified, "solve" stops when a lap is completed, save stores checkpoint and best genome\nNew starts with a new network, keep retains the current network for further training\n'
            )

            try:
                tar_map, gens, pick, save, retain_inp = input("Enter: ").split(" ")
            except:
                print("Invalid Input: Try again!")
                break

            train = None
            save_choice = None
            retain = None

            if pick.lower() == "train":
                train = False

            elif pick.lower() == "solve":
                train = True

            if save.lower() == "save":
                save_choice = True

            elif save.lower() == "no":
                save_choice = False

            if retain_inp.lower() == "keep":
                retain = True

            elif retain_inp.lower() == "new":
                retain = False
                loader.reset_defaults()

            if train == None or save_choice == None or retain == None:
                print("Invalid Input: Try again!")
            else:
                try:
                    map_path = "./maps/" + tar_map
                    loader.train(int(gens), train, save_choice, map_path)
                    correct = True
                except OSError as e:
                    print(e)
                    print("Invalid Input: Try again!")

    # Evaluation Section
    
    # Main menu loop
    def menu():
        Menu = True

        while Menu == True:

            print("Select Option")
            print("1. Train network")
            print("2. Resume from checkpoint")
            print("3. Replay genomes by map")
            print("4. Export genomes/Save")
            print("5. Exit")

            try:
                key = input("Select Option: ")

                match int(key):

                    case 1:
                        train_menu()

                    case 2:
                        checkpoint_load()
                    case 3:
                        genome_replay()

                    case 4:
                        loader.save_genomes()

                    case 5:
                        Menu = False
                        print("Thank you: Exiting")

            except ValueError as e:
                print("Invalid Input: Try again!\n")

    # Load maps from folder
    maps = get_maps()
    loader = neatloader()

    # Loads genomes set
    loader.load_genomes()

    # Main menu loop
    menu()
    