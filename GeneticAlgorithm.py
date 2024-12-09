import random
import time
import matplotlib.pyplot as plt

random.seed(42)

cityList = ["A", "B", "C", "D", "E", "F"]
# Read the city distance matrix from a file or define it here
cityMatrix = [
#   A   B   C   D   E   F   
   [0,  29, 20, 21, 16, 31],  # A
   [29, 0,  15, 17, 28, 23],  # B
   [20, 15, 0,  30, 26, 40],  # C  
   [21, 17, 30, 0,  23, 27],  # D
   [16, 28, 26, 23, 0,  19],  # E
   [31, 23, 40, 27, 19, 0],   # F
]

# Function to display the city distance matrix
def displayMatrix(matrix):
    print(f"{"_":3}{"A":4}{"B":4}{"C":4}{"D":4}{"E":4}{"F":4}")
    for i, x in enumerate(matrix):
        print(cityList[i], end="")
        for y in x:
            print(f"{y:3} ", end="")
        print()

displayMatrix(cityMatrix)

def main():
    gLength = len(cityMatrix)  # Number of cities
    popSize = 8  # Population size (must be a multiple of 4)
    iterCount = 10  # Number of iterations

    bestFit = []  # Store best fitness values per iteration
    worstFit = []  # Store worst fitness values per iteration

    # Generate initial population (genomes)
    genomeS = []
    for i in range(popSize):
        genomeS.append(random.sample(range(gLength), gLength))

    for i in range(iterCount - 1):
        # Evaluate genomes and get fitness values
        evaluation = evaluate_genomes(genomeS)

        # Select parents using tournament selection
        parentGens = tournament_selection(popSize, evaluation)

        # Store best and worst fitness values
        bestFit.append(min(evaluation)[0])
        worstFit.append(max(evaluation)[0])

        # Generate new population through mutation
        genomeS = populate_with_mutation(gLength, parentGens)

    # Final evaluation after last iteration
    evaluation = evaluate_genomes(genomeS)
    
    bestFit.append(min(evaluation)[0])
    worstFit.append(max(evaluation)[0])

    return min(evaluation), bestFit, worstFit


def populate_with_mutation(gLength, parentGens):
    genomeS = []

    for parent in parentGens:
        genomeS.append(parent[1])  # Save parents

        for i in range(3):  # Number of new genomes generated per parent
            newGenome = parent[1][:]
            change = random.sample(range(gLength), 2)  # Change two gene positions
            tmp = newGenome[change[0]]
            newGenome[change[0]] = newGenome[change[1]]
            newGenome[change[1]] = tmp
            genomeS.append(newGenome[:])  # Add new genome to the population
    return genomeS

def tournament_selection(popSize, evaluation):
    gCount = popSize / 4  # Number of tournaments
    parentGens = []

    for offset in range(int(gCount)):
        best = min(evaluation[offset * 4:offset * 4 + 4])  # Select the best genome in each tournament
        parentGens.append(best)
    return parentGens

def evaluate_genomes(genomeS):
    evaluation = []
    
    for genome in genomeS:
        fitness = 0
        past = genome[-1]

        # Calculate the fitness (total distance) for each genome
        for i in genome:
            fitness += cityMatrix[past][i]
            past = i
        evaluation.append([fitness, genome])
    return evaluation

start_time = time.time()  # Start measuring execution time
final, best, worst = main()
end_time = time.time()  # End measuring execution time
execution_time = end_time - start_time
print(f"\nExecution time: {execution_time * 1000:0.4f} ms")

# Generate the path from the final genome
path_str = "->".join([cityList[i] for i in final[1]])

# Print the formatted output: final path and total distance
print(f"Path       : {path_str}")
print(f"Distance   : {final[0]}")

# Plot the best and worst fitness values over iterations
x = range(len(best))
plt.plot(x, best, color='blue', label='best-fit')
plt.plot(x, worst, color='red', label='worst-fit')

plt.xlabel('Iteration')
plt.ylabel('Fitness')

plt.legend()
plt.show()
