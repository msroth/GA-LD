"""
(C) 2020-2024 MSRoth
"""
import Levenshtein
import random
import uuid
import shortuuid
import sys
import time


# defaults
DEFAULT_PHRASE = 'Scarlett O\'Hara was not beautiful,'
DEFAULT_GENERATIONS = 50000         # max number of generations to run
DEFAULT_POPULATION = 50             # number of individuals per generation
DEFAULT_NUMBER_OF_PARENTS = 20      # number of parents to populate next generation

# constants
PHRASE_KEY = 'phrase'
GENERATIONS_KEY = 'generations'
POPULATION_KEY = 'population'
PARENTS_KEY = 'number_of_parents'
LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
           'v', 'w', 'x', 'y', 'z', ' ']


class Individual:
    """
    A class representing a single individual in a population
    """
    def __init__(self, phrase_length: int, born: int, parents: list = None):
        self.fitness = 10000
        self.born = born   # generation this individual was born
        self.parents = []  # list of parents
        #self.id = uuid.uuid4().hex  # my id
        self.id = shortuuid.ShortUUID().random(length=18)
        self.phrase = ''.join(random.choices(LETTERS, k=phrase_length))  # this individual's starting phrase
        if parents is not None:
            self.parents = parents

    def calculate_fitness(self, target_phrase):
        """
        Calculate this individual's fitness using the LD edit distance
        """
        self.fitness = Levenshtein.distance(target_phrase, self.phrase)

    def print(self):
        print('Id:              {}'.format(self.id))
        if len(self.parents) > 0:
            print('  Parents:       {}, {}'.format(str(self.parents[0].id), str(self.parents[1].id)))
        else:
            print('  Parents:       None')
        print('  Phrase:        {}'.format(self.phrase))
        print('  Fitness score: {}'.format(self.fitness))


class Population:
    """
    A class representing a generation's population.
    """
    def __init__(self, size, generation: int = 0, target_phrase: str = ''):
        self.members = []  # list of Individuals in generation
        self.size = size
        self.target_phrase = target_phrase
        self.generation = generation

    def print(self):
        """
        Default print method for the class.
        """
        self.update_fitness_scores()
        print('Population Summary:')
        print('  Generation:       {}'.format(self.generation))
        print('  Size:             {}'.format(self.size))
        print('  Best individual:  {}'.format(self.return_fittest().id))
        print('  Best score:       {}'.format(self.return_fittest().fitness))
        print('  Best phrase:      {}'.format(self.return_fittest().phrase))
        print('Individuals:')
        for individual in self.members:
            individual.print()

    def create_initial_population(self):
        """
        Create the initial members to start the algorithm.
        """
        for _ in range(self.size):
            self.members.append(Individual(len(self.target_phrase), 0))
        self.update_fitness_scores()

    def update_fitness_scores(self):
        """
        Update the fitness score for each member of the population
        """
        for individual in self.members:
            individual.calculate_fitness(self.target_phrase)

    def return_fittest(self) -> Individual:
        """
        Return the best individual in the population
        """
        return self.return_top_fittest(1)[0]

    def return_top_fittest(self, number_to_return) -> list:
        """
        Return the best number_to_return individuals in the population
        """
        best_individuals = []

        # make sure fitness scores are current
        self.update_fitness_scores()

        # the best individuals have the lowest fitness scores, so sort them
        sorted_population = self.__sort_population_by_fitness()

        # return requested number of members
        for i in range(number_to_return):
            best_individuals.append(sorted_population[i])
        return best_individuals

    def __sort_population_by_fitness(self):
        """
        Sort by lowest fitness score
        """
        sorted_list = sorted(self.members, key=lambda x: x.fitness, reverse=False)
        return sorted_list

    def add_individuals_to_population(self, individuals: list):
        """
        Add an Individual to the population's membership
        """
        self.members.extend(individuals)
        if len(self.members) > self.size:
            print('WARNING:  population size is {} > {}'.format(len(self.members), self.size))


class GeneticAlgorithm:
    """
    The Genetic Algorithm class
    """
    def __init__(self, inputs_values):
        self.target_phrase = inputs_values[PHRASE_KEY]
        self.max_generations = int(inputs_values[GENERATIONS_KEY])
        self.number_of_parents = int(inputs_values[PARENTS_KEY])
        self.population_size = int(inputs_values[POPULATION_KEY])
        self.mutation_rate = 1.0/len(self.target_phrase)  # suggested by Internet sources
        self.current_generation = 0
        self.all_generations_data = []
        self.current_population = Population(self.population_size, 0, self.target_phrase)
        self.best_individual = None

    def run(self):
        """
        Main entry point to run the Genetic Algorithm.
        """
        # create initial, random generation
        self.current_population.create_initial_population()

        # append current population to generation list for tracking
        self.all_generations_data.append(self.current_population)

        # logic to keep track of the best individual
        self.best_individual = self.current_population.return_fittest()

        print('*' * 15)
        print('Generation: {}, Fittest {}'.format(self.current_generation, self.best_individual.fitness))
        self.current_population.print()
        begin = input('Ready to begin [Y/N]: ')
        if len(begin) > 0 and begin.upper()[0] == 'N':
            sys.exit(-1)

        # main loop
        while self.best_individual.fitness > 0 and self.current_generation < self.max_generations - 1:

            # get pool of parents
            parents = self.current_population.return_top_fittest(self.number_of_parents)

            # create a new population for next generation
            self.current_generation += 1
            self.current_population = Population(self.population_size, self.current_generation, self.target_phrase)

            # create next generation with crossovers
            next_generation_individuals = self.create_crossover_children(parents, self.population_size)
            self.current_population.add_individuals_to_population(next_generation_individuals)

            # mutate an individual already in the population
            for i in range(self.current_population.size):
                probabilty = random.random()
                if probabilty <= self.mutation_rate:
                    self.current_population.members[i] = self.return_mutated_child(self.current_population.members[i])

            # update fitness scores
            self.current_population.update_fitness_scores()

            # keep track of the best score
            generation_best_individual = self.current_population.return_fittest()
            if generation_best_individual.fitness < self.best_individual.fitness:
                self.best_individual = generation_best_individual

            # append current population to generation list for tracking
            self.all_generations_data.append(self.current_population)

            print('*' * 15)
            print('Generation: {}, Fittest {}'.format(self.current_generation, generation_best_individual.fitness))
            self.current_population.print()
            print('Next Gen Parents:')
            for parent in self.current_population.return_top_fittest(self.number_of_parents):
                print('  Id: {}, Phrase: {}, Fitness score: {}'.format(parent.id, parent.phrase, parent.fitness))

            # DEBUG - pause for each generation
            #_ = input('Press [ENTER] to continue')

    def create_crossover_children(self, parents: list, number_individuals: int) -> list:
        """
        Create children by mating two parents and crossing their genes at a particular cross point.

        - the crossover point is a random number between 1 and the length of the phrase
        - the child is an assembly of the left portion of one parent and the right portion of the other

        For example:
        - crosspoint = random(1, 32) = 18
        - parent 1 = id: xhxxbwkqkblgqpwtlxl <-> phrase: bjsdmfyzjwvxqq
        - parent 2 = id: bftmeswlsyzrjjjmoar <-> phrase: hvqhc hf roejt
        - if LEFT1:
            - child = xhxxbwkqkblgqpwtlxl + hvqhc hf roejt
        -if LEFT2:
            - child = bftmeswlsyzrjjjmoar + bjsdmfyzjwvxqq
        """
        LEFT1 = 'LEFT1'
        LEFT2 = 'LEFT2'
        next_generation = []

        for _ in range(number_individuals):
            # select two parents from parents list
            selected_parents = random.sample(parents, k=2)
            parent1 = selected_parents[0]
            parent2 = selected_parents[1]

            # create new child
            child = Individual(len(parent1.phrase), self.current_generation, [parent1, parent2])

            # find the cross point -- not extreme ends
            cross_point = random.randint(1, len(self.target_phrase) - 2)

            # chop up the parent phrases into left and right components at the cross point
            parent1_left = parent1.phrase[:cross_point]
            parent1_right = parent1.phrase[cross_point:]
            parent2_left = parent2.phrase[:cross_point]
            parent2_right = parent2.phrase[cross_point:]

            # choose to cross the left or right side of the cross point
            left_segment = random.choice([LEFT1, LEFT2])

            # assemble left and right components to make new child phrase
            if left_segment == LEFT1:
                child.phrase = parent1_left + parent2_right
            else:
                child.phrase = parent2_left + parent1_right

            if child.phrase == parent1.phrase or child.phrase == parent2.phrase:
                print('Adding mutation because child phrase = parent phrase...')
                child = self.return_mutated_child(child)

            # add new individuals to return list
            next_generation.append(child)

            print('*CROSSOVER*')
            print('parent1 id:        {}'.format(parent1.id))
            print('parent2 id:        {}'.format(parent2.id))
            print('crossover point:   {}'.format(cross_point))
            print('parent1 phrase:   |{} <-> {}| ({})'.format(parent1_left, parent1_right, len(parent1.phrase)))
            print('parent2 phrase:   |{} <-> {}| ({})'.format(parent2_left, parent2_right, len(parent2.phrase)))
            print('assembly sequence: {}'.format(left_segment))
            print('new child id:      {}'.format(child.id))
            print('child phrase:     |{}| ({})'.format(child.phrase, len(child.phrase)))

        return next_generation

    @staticmethod
    def return_mutated_child(child) -> Individual:
        """
        Randomly select a gene (letter) and replace it with another letter from the LETTERS list.
        """
        # randomly select a gene (letter) to mutate
        gene_to_mutate = random.randint(0, len(child.phrase) -1)

        # randomly select a new letter
        mutation = random.sample(LETTERS, k=1)[0]

        print('*MUTATE*')
        print('individual id:      {}'.format(child.id))
        print('mutate gene:        {}'.format(gene_to_mutate))
        print('mutation:           {}'.format(mutation))
        print('phrase before:     |{}| ({})'.format(child.phrase, len(child.phrase)))
        print('                    {}*'.format(' ' * (gene_to_mutate)))

        # replace the randomly selected gene with a new letter
        left_str = child.phrase[:gene_to_mutate]
        right_str = child.phrase[gene_to_mutate + 1:]
        child.phrase = left_str + mutation + right_str
        print('new phrase:        |{}| ({})'.format(child.phrase, len(child.phrase)))

        # Debug
        # _ = input('Press [ENTER] to continue')
        return child

    def print(self):
        """
        Print final results with option to write historic data to file.
        """
        print('**********')
        print('Number generations run:    {}'.format(self.current_generation))
        print('Genetic Algorithm parameters:')
        print('  Phrase to find:          {}'.format(self.target_phrase))
        print('  Max generations:         {}'.format(self.max_generations))
        print('  Population/generation:   {}'.format(self.population_size))
        print('  Number of parents:       {}'.format(self.number_of_parents))
        print('  Probability of mutation: {:.4f}'.format(self.mutation_rate))
        print('Best individual:        {}'.format(self.best_individual.id))
        print('  Phrase:               {}'.format(self.best_individual.phrase))
        print('  Score:                {}'.format(self.best_individual.fitness))
        print('  Born:                 {}'.format(self.best_individual.born))
        if len(self.best_individual.parents) > 0:
            print('  Parents:              {}, {}'.format(str(self.best_individual.parents[0].id),
                                                          str(self.best_individual.parents[1].id)))
        else:
            print('  Parents:              None')
        print()

        # write data to file
        save = input('Write data to file? [Y/N]: ')
        if len(save) > 0 and save.upper()[0] == 'Y':
            print('Writing data to GA-LD.csv...')
            f = open('GA-LD.csv', 'w')

            # write best of each generation
            f.write('BEST OF EACH GENERATION\n')
            f.write('generation, id, score, phrase, parent1_id, parent1_phrase, parent2_id, parent2_phrase\n')
            for i in range(len(self.all_generations_data)):
                population = self.all_generations_data[i]
                best_individual = population.return_fittest()
                f.write('{}, {}, {}, {}'.format(i, best_individual.id, best_individual.fitness, best_individual.phrase))
                if len(best_individual.parents) > 0:
                    f.write(', {}, {}, {}, {}\n'.format(best_individual.parents[0].id,
                                                        best_individual.parents[0].phrase,
                                                        best_individual.parents[1].id,
                                                        best_individual.parents[1].phrase))
                else:
                    f.write('\n')

            # write all data
            f.write('\nALL DATA\n')
            f.write('generation, id, score, phrase, parent1, parent2\n')
            for i in range(len(self.all_generations_data)):
                population = self.all_generations_data[i]
                for j in range(len(population.members)):
                    individual = population.members[j]
                    f.write('{}, {}, {}, {}'.format(i, individual.id, individual.fitness, individual.phrase))
                    if len(individual.parents) > 0:
                        f.write(', {}, {}\n'.format(individual.parents[0].id, individual.parents[1].id))
                    else:
                        f.write('\n')
            f.close()


def get_user_inputs() -> dict:
    """
    Get inputs from the user.  Use defaults if no values are received.
    """

    # set defaults
    input_values = {PHRASE_KEY: str(DEFAULT_PHRASE).lower(),
                    GENERATIONS_KEY: DEFAULT_GENERATIONS,
                    POPULATION_KEY: DEFAULT_POPULATION,
                    PARENTS_KEY: DEFAULT_NUMBER_OF_PARENTS
                    }

    print('\n\n')
    input_phrase = input('Enter a phrase for the GA to find [{}]: '.format(DEFAULT_PHRASE)).strip().lower()
    if len(input_phrase) > 0:
        input_values[PHRASE_KEY] = remove_punctuation(input_phrase)
    else:
        input_values[PHRASE_KEY] = remove_punctuation(input_values[PHRASE_KEY])

    input_generations = input('Enter the maximum number of generations for the GA to run [{}]:'.
                              format(DEFAULT_GENERATIONS)).strip()
    if len(input_generations) > 0:
        if int(input_generations) > 1:
            input_values[GENERATIONS_KEY] = int(input_generations)
        else:
            print('WARNING:  max generations must be > 1.  Using default value {} instead.'.format(DEFAULT_GENERATIONS))

    input_population = input('Enter the population size for each generation [{}]:'.format(DEFAULT_POPULATION)).strip()
    if len(input_population) > 0:
        if int(input_population) > 1:
            input_values[POPULATION_KEY] = int(input_population)
        else:
            print('WARNING:  population size must be > 1.  Using default value {} instead.'.format(DEFAULT_POPULATION))

    input_parents = input('Enter the number of parents to populate the next generation [{}]:'.
                          format(DEFAULT_NUMBER_OF_PARENTS)).strip()
    if len(input_parents) > 0:
        if int(input_parents) > 1:
            input_values[PARENTS_KEY] = int(input_parents)
        else:
            print('WARNING:  parents must be > 1.  Using default value {} instead.'.format(DEFAULT_NUMBER_OF_PARENTS))

    print('Mutation rate will be {:.4f} (1/length of phrase)'.format(1/len(input_values[PHRASE_KEY])))
    _ = input('Press [ENTER] to continue')
    return input_values


def remove_punctuation(string_in: str) -> str:
    """
    Remove any character not in the LETTERS list (punctuation and numbers), and collapse multiple spaces
    created sometimes by removing punctuation.
    """
    clean_string = ''
    for letter in string_in.lower():
        if letter in LETTERS:
            clean_string += letter
    return clean_string.strip()


def print_intro():
    print('\n(C) 2020-2024 MSRoth')
    print('*' * 68)
    print('This experiment combines the power of a Genetic Algorithm and the')
    print('Levenshtein Distance calculation to \'find\' a target phrase')
    print('entered by the user.\n')
    print('The input parameters are:')
    print('* Input Phrase: the target phrase for the GA to find.  The default')
    print('   is the first clause from _Gone With the Wind_.')
    print('* Max Generations:  the maximum number of generations the GA will')
    print('   use to find the target phrase.  A generation includes creating')
    print('   offspring by crossing the genes of the \'best\' parents\' from')
    print('   the previous generation and some random mutations.')
    print('* Population Size:  the number of individuals in a generation.  This')
    print('   number does not change from generation to generation.')
    print('* Number of Parents:  the number of \'best\' individuals from the')
    print('   previous generation to sire all of the offspring in the next')
    print('   generation.')
    print('*' * 68)
    input('Press any [ENTER] to continue.')
    print()


if __name__ == '__main__':
    # random.seed(1111)  # used for testing
    print_intro()
    inputs = get_user_inputs()
    ga = GeneticAlgorithm(inputs)
    ga.run()
    ga.print()

# <SDG><
