# This Code is Heavily Inspired By The YouTuber: Cheesy AI
# Code Changed, Optimized And Commented By: NeuralNine (Florian Dedov)
# Code Modified for University assignment by Shaun Searle - Checkpointing, Replays, Code structure, Lap Detection, Lap Times, Data collection

# Needs Pip Install
import neat
import pygame

#Included
import pickle
import genome
import os

from pathlib import Path

import Car

WIDTH = 1920
HEIGHT = 1080


class neatgame:

    current_generation = (
        0  # Generation counter #Checkpoint also keeps track of generation
    )

    def __init__(
        self,
        generations_to_train,
        solve_for_solution=True,
        cur_map="./maps/map.png",
        config_path="./config.txt",
    ):
        self.generations_to_train = generations_to_train
        self.config_path = config_path

        self.config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )

        # Create Population And Add Reporters
        self.population = neat.Population(self.config)
        
        self.stats = neat.StatisticsReporter()
        self.cur_checkpoint = neat.Checkpointer(
            None, None, "./checkpoints/neat-checkpoint-" + Path(cur_map).stem + "-"
        )  # Prevent AutoSave, Start CheckPoint
        self.cur_map = cur_map
        
        self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(self.stats)
        self.population.add_reporter(self.cur_checkpoint)

        self.cur_checkpoint.start_generation
        self.best_time = 0
        self.solve_for_solution = solve_for_solution

    # Loading checkpoint removes reporters - use to reattach
    def reattatch_reporters(self):
        self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(self.stats)
        self.population.add_reporter(self.cur_checkpoint)
    
    def update_checkpoint_prefix(self):
        self.cur_checkpoint.filename_prefix = (
            "./checkpoints/neat-checkpoint-" + Path(self.cur_map).stem + "-"
        )

    def save_checkpoint(self):
        self.update_checkpoint_prefix()
        self.cur_checkpoint.save_checkpoint(
            self.config,
            self.population.population,
            self.population.species,
            self.population.generation,
        )

    def load_genome(self, genome_file):
        with open(genome_file, "rb") as input:
            return pickle.load(input)

    def replay_genome(self, genome, lap_time):

        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_path,
        )
        
        # genomes = list(iteritems(population.population))
        self.run_simulation([(1, genome)], config, True, lap_time)
        
    def save_best_genome(self, id):   
        gen = genome.genome(self.population.best_genome, self.population.generation, self.cur_map, self.best_time, id)
        return gen
        

    def run_simulation(self, genomes, config, replay=False, lap_time=None):

        # Empty Collections For Nets and Cars
        nets = []
        cars = []

        # Initialize PyGame And The Display
        pygame.init()
        screen = pygame.display.set_mode((WIDTH,HEIGHT),pygame.SCALED | pygame.FULLSCREEN)
        # os.environ['SDL_VIDEO_CENTERED'] = '1'

        # For All Genomes Passed Create A New Neural Network
        for i, g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g, config)
            nets.append(net)
            g.fitness = 0

            cars.append(Car.Car())

        # Clock Settings
        # Font Settings & Loading Map
        clock = pygame.time.Clock()
        generation_font = pygame.font.SysFont("Arial", 30)
        alive_font = pygame.font.SysFont("Arial", 20)
        game_map = pygame.image.load(self.cur_map).convert()  # Convert Speeds Up A Lot

        self.current_generation += 1

        # Simple Counter To Roughly Limit Time (Not Good Practice)
        counter = 0
        lap_start = pygame.time.get_ticks()

        running = True
        
        while running == True:
            # Exit On Quit Event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    break

            # For Each Car Get The Acton It Takes
            for i, car in enumerate(cars):
                output = nets[i].activate(car.get_data())
                choice = output.index(max(output))
                if choice == 0:
                    car.angle += 10  # Left
                elif choice == 1:
                    car.angle -= 10  # Right
                elif choice == 2:
                    if car.speed - 2 >= 12:
                        car.speed -= 2  # Slow Down
                else:
                    car.speed += 2  # Speed Up

            # Check If Car Is Still Alive
            # Increase Fitness If Yes And Break Loop If Not
            still_alive = 0
            Lap = False
            for i, car in enumerate(cars):
                if car.is_alive():
                    still_alive += 1
                    car.update(game_map)
                    genomes[i][1].fitness += car.get_reward()
                    if not Lap:
                        if car.Lap:
                            # print("Lap Complete")
                            Lap = True
                            # Check if training for performance or completion
                            if self.solve_for_solution:
                                genomes[i][1].fitness = +100000001

                            self.best_time = (
                                pygame.time.get_ticks() - lap_start
                            )  # Time Lap
                            break

            if Lap:
                if replay == False:
                    self.population.reporters.end_generation(
                        self.population.config,
                        self.population.population,
                        self.population.species,
                    )  # Lap Complete end generation
                pygame.quit()
                break
            
            if still_alive == 0:
                break

            counter += 1
            if counter == 30 * 40:  # Stop After About 20 Seconds
                break

            # Draw Map And All Cars That Are Alive
            screen.blit(game_map, (0, 0))
            for car in cars:
                if car.is_alive():
                    car.draw(screen)

            # Display Info / Change text if replay
            if replay == True:
                text = generation_font.render(f'Best Genome Replay - Lap Time: {lap_time/1000: .2f}s', True, (255, 0, 0))
                text_rect = text.get_rect()
                text_rect.center = (250, 50)
                screen.blit(text, text_rect)
            else:
                text = generation_font.render(
                    "Generation: " + str(self.population.generation), True, (0, 0, 0)
                )
                text_rect = text.get_rect()
                text_rect.center = (100, 50)
                screen.blit(text, text_rect)

                text = alive_font.render(
                    "Still Alive: " + str(still_alive), True, (0, 0, 0)
                )
                text_rect = text.get_rect()
                text_rect.center = (100, 90)
                screen.blit(text, text_rect)

            pygame.display.flip()
            clock.tick(60)  # 60 FPS
