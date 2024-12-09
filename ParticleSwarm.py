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

# Evaluate particles and calculate their fitness
def evaluate_particles(particles):
    evaluation = []
    for particle in particles:
        fitness = 0
        past = particle[-1]
        # Calculate the total distance for the route
        for i in particle:
            fitness += cityMatrix[past][i]
            past = i
        evaluation.append([fitness, particle])
    return evaluation

# Apply velocity (swap operations) to a route
def apply_velocity(route, velocity):
    for swap in velocity:
        i, j = swap
        route[i], route[j] = route[j], route[i]
    return route

# Generate a random velocity (list of swap operations)
def generate_random_velocity(num_cities):
    return [tuple(random.sample(range(num_cities), 2))]

# Update velocity based on the current position, pBest, and gBest
def update_velocity(current_velocity, current_position, pBest_position, gBest_position, num_cities):
    new_velocity = []
    # Approach pBest by making only one swap operation a b c d /0-2/ c b a d
    for i in range(num_cities-1,-1,-1):
        if current_position[i] != pBest_position[i]:
            j = current_position.index(pBest_position[i])
            new_velocity.append((i, j))
            break  # Stop after performing only one swap

    # Approach gBest by making only one swap operation a b c d /1-2/ a c b d // c a b d //
    for i in range(num_cities):
        if current_position[i] != gBest_position[i]:
            j = current_position.index(gBest_position[i])
            new_velocity.append((i, j))
            break  # Stop after performing only one swap
    return new_velocity

# PSO (Particle Swarm Optimization) algorithm
def main(cityMatrix, num_particles=8, num_iterations=10):
    num_cities = len(cityMatrix)

    # Initialize particles and their velocities
    particles = []
    for _ in range(num_particles):
        particles.append(random.sample(range(num_cities), num_cities))

    velocities = []
    for _ in range(num_particles):
        velocities.append(generate_random_velocity(num_cities))

    # Evaluate initial particles
    evaluation = evaluate_particles(particles)
    gBest = min(evaluation)
    pBest = evaluation[:]

    # Track best and worst fitness values for each iteration
    best_fitness_per_iteration = []
    worst_fitness_per_iteration = []

    # Main PSO loop for specified number of iterations
    for _ in range(num_iterations):
        for i in range(num_particles):
            particles[i] = apply_velocity(particles[i][:], velocities[i])

        evaluation = evaluate_particles(particles)

        # Update pBest and gBest
        for i in range(num_particles):
            if evaluation[i][0] < pBest[i][0]:
                pBest[i] = evaluation[i][:]

        if min(evaluation)[0] < gBest[0]:
            gBest = min(evaluation)
        
        # Record the best and worst fitness for this iteration
        best_fitness_per_iteration.append(min(evaluation)[0])
        worst_fitness_per_iteration.append(max(evaluation)[0])

        # Update velocities based on the current position, pBest, and gBest
        for i in range(num_particles):
            velocities[i] = update_velocity(
                velocities[i], particles[i], pBest[i][1], gBest[1], num_cities
            )

    return gBest, best_fitness_per_iteration, worst_fitness_per_iteration

# Main program
start_time = time.time()  # Start measuring execution time
final, best, worst = main(cityMatrix)
end_time = time.time()  # End measuring execution time

execution_time = (end_time - start_time) * 1000  # Convert to milliseconds

# Output the results
print(f"\nExecution Time: {execution_time:.4f} ms")
path_str = "->".join([cityList[i] for i in final[1]])
# Print the formatted output
print(f"Path       : {path_str}")
print(f"Distance   : {final[0]}")

# Create the plot for best and worst fitness over iterations
x = range(len(best))
plt.plot(x, best, color='blue', label='best-fit')
plt.plot(x, worst, color='red', label='worst-fit')

plt.xlabel('Iteration')
plt.ylabel('Fitness')

plt.legend()
plt.show()
