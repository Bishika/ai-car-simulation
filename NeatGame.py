# This Code is Heavily Inspired By The YouTuber: Cheesy AI
# Code Changed, Optimized And Commented By: NeuralNine (Florian Dedov)
# Code Modified for University assignment by Shaun Searle - Checkpointing, Replays


import sys
import neat
import pygame

from neat.six_util import iteritems
from pathlib import Path

import Car

WIDTH = 1920
HEIGHT = 1080

class NeatGame:
    
    current_generation = 0 # Generation counter #Checkpoint also keeps track of generation
    
    def __init__(self, generations_to_train, cur_map='./maps/map2_2.png', config_path = "./config.txt"):
        self.generations_to_train = generations_to_train
        self.config_path = config_path
        
        self.config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

        # Create Population And Add Reporters
        self.population = neat.Population(self.config)
        self.population.add_reporter(neat.StdOutReporter(True))
        self.stats = neat.StatisticsReporter()
        self.cur_checkpoint = neat.Checkpointer(None, None,"neat-checkpoint-" + Path(cur_map).stem + "-") # Prevent AutoSave, Start CheckPoint
        self.cur_map = cur_map

        self.population.add_reporter(self.stats)
        self.population.add_reporter(self.cur_checkpoint)

        self.cur_checkpoint.start_generation
        


    def replay_genome(self, genomes):
        
        # genomes = list(iteritems(population.population))
        print(genomes)
        self.run_simulation([(1, genomes)], self.population.config, True)
        


    def run_simulation(self, genomes, config, replay=False):
        
        # Empty Collections For Nets and Cars
        nets = []
        cars = []

        # Initialize PyGame And The Display
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)

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
        game_map = pygame.image.load(self.cur_map).convert() # Convert Speeds Up A Lot

        global current_generation
        self.current_generation += 1

        # Simple Counter To Roughly Limit Time (Not Good Practice)
        counter = 0

        while True:
            # Exit On Quit Event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit(0)

            # For Each Car Get The Acton It Takes
            for i, car in enumerate(cars):
                output = nets[i].activate(car.get_data())
                choice = output.index(max(output))
                if choice == 0:
                    car.angle += 10 # Left
                elif choice == 1:
                    car.angle -= 10 # Right
                elif choice == 2:
                    if(car.speed - 2 >= 12):
                        car.speed -= 2 # Slow Down
                else:
                    car.speed += 2 # Speed Up
            
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
                            print("Lap Complete")
                            Lap = True
                            genomes[i][1].fitness =+ 100000001

            if Lap:
                self.population.reporters.end_generation(self.population.config, self.population.population, self.population.species)
                break
            if still_alive == 0:
                break

            counter += 1
            if counter == 30 * 40: # Stop After About 20 Seconds
                break

            # Draw Map And All Cars That Are Alive
            screen.blit(game_map, (0, 0))
            for car in cars:
                if car.is_alive():
                    car.draw(screen)
                
            # Display Info / Change text if replay
            if(replay == True) :
                text = generation_font.render("Best Genome Replay", True, (255,0,0))
                text_rect = text.get_rect()
                text_rect.center = (150, 50)
                screen.blit(text, text_rect)
            else:
                text = generation_font.render("Generation: " + str(self.current_generation), True, (0,0,0))
                text_rect = text.get_rect()
                text_rect.center = (100, 50)
                screen.blit(text, text_rect)

                text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
                text_rect = text.get_rect()
                text_rect.center = (100, 90)
                screen.blit(text, text_rect)
            

            pygame.display.flip()
            clock.tick(60) # 60 FPS
            











