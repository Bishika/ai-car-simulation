# This Code is Heavily Inspired By The YouTuber: Cheesy AI
# Code Changed, Optimized And Commented By: NeuralNine (Florian Dedov)
# Code Modified for University assignment by Shaun Searle - Checkpointing, Replays


import sys
import pickle

import neat
import pygame

import Car

WIDTH = 1920
HEIGHT = 1080

current_generation = 0 # Generation counter

generations_to_train = 10 # How many generations to train
genome_file = "winner_10Gen.pkl" # Specifically for replay and not training

def replay_genome(config_path, genome_path):
    # Load requried NEAT config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Unpickle saved winner
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    # Convert loaded genome into required data structure
    genomes = [(1, genome)]

    # Call game with only the loaded genome
    run_simulation(genomes, config)


def run_simulation(genomes, config, cur_map='map2_2.png'):
    
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

        cars.append(Car())

    # Clock Settings
    # Font Settings & Loading Map
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    game_map = pygame.image.load(cur_map).convert() # Convert Speeds Up A Lot

    global current_generation
    current_generation += 1

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
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()

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
        if(generations_to_train < current_generation) :
            text = generation_font.render("Best Genome Replay", True, (255,0,0))
            text_rect = text.get_rect()
            text_rect.center = (150, 50)
            screen.blit(text, text_rect)
        else:
            text = generation_font.render("Generation: " + str(current_generation), True, (0,0,0))
            text_rect = text.get_rect()
            text_rect.center = (100, 50)
            screen.blit(text, text_rect)

            text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
            text_rect = text.get_rect()
            text_rect.center = (100, 90)
            screen.blit(text, text_rect)
        

        pygame.display.flip()
        clock.tick(60) # 60 FPS

if __name__ == "__main__":
    
    # Load Config
    config_path = "./config.txt"
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    # Create Population And Add Reporters
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    cur_checkpoint = neat.Checkpointer(generations_to_train + 1,) # Prevent AutoSave, Start CheckPoint
    cur_checkpoint.start_generation

    # Run Simulation For A Maximum of 1000 Generations
    population.run(run_simulation, 20)

    cur_checkpoint.end_generation(population.config, population, population.species) # Save Checkpoint


    # winner = population.run(run_simulation(genomes, config), generations_to_train)
    # with open(genome_file, "wb") as f:
    #     pickle.dump(winner, f)
    # f.close()

    # replay_genome(config_path, genome_file)




