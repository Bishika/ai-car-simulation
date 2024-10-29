import NeatGame
    

map = './maps/map5.png'
Game = NeatGame.NeatGame(10, map)

Game.population = Game.cur_checkpoint.restore_checkpoint('neat-checkpoint-map-12')

maps = ['./maps/map1.png', './maps/map2.png', './maps/map3.png', './maps/map4.png', './maps/map5.png']
results = []


#Train control model

#Save Control Model

#Iterate through maps training until the solution has been found, load checkpoint from control each time, keep generations required and lap time in an array

#Iterate through maps training until the solution has been found, Training from scratch, keep generations required and lap time in an array

# Run until solution found
Game.population.run(Game.run_simulation)

# Game.replay_genome(Game.population.best_genome)
Game.cur_checkpoint.save_checkpoint(Game.config, Game.population.population, Game.population.species, Game.population.generation)
