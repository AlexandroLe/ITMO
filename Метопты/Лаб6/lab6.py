import random
# Матрица расстояний
matrix = [
    [0, 1, 7, 2, 8],
    [2, 0, 10, 3, 1],
    [7, 10, 0, 2, 6],
    [2, 3, 2, 0, 4],
    [8, 1, 6, 4, 0]
]

n = 5  # количество городов
population_size = 4
mutation_rate = 0.05
max_generations = 6

# Расчёт стоимости маршрута (с возвратом в начальный город)
def calculate_cost(route):
    cost = 0
    for i in range(len(route) - 1):
        cost += matrix[route[i] - 1][route[i + 1] - 1]
    cost += matrix[route[-1] - 1][route[0] - 1]  # возвращаемся в начало
    return cost

# Генерация случайной перестановки городов
def generate_individual():
    individual = list(range(1, n + 1))
    random.shuffle(individual)
    return individual

# Двухточечный оператор скрещивания
# Новый двухточечный оператор скрещивания
def crossover(parent1, parent2):
    size = len(parent1)
    p1, p2 = sorted(random.sample(range(1, size), 2))

    # Вырезаем промежутки
    fragment1 = parent1[p1:p2]
    fragment2 = parent2[p1:p2]

    child1 = [None] * size
    child2 = [None] * size

    # Вставляем фрагменты
    child1[p1:p2] = fragment2
    child2[p1:p2] = fragment1

    # Функция заполнения потомка
    def fill(child, donor, fragment):
        size = len(child)
        start_idx = (p2) % size
        donor_idx = (p2) % size
        while None in child:
            city = donor[donor_idx % size]
            if city not in child:
                child[start_idx % size] = city
                start_idx += 1
            donor_idx += 1
        return child

    child1 = fill(child1, parent1, fragment2)
    child2 = fill(child2, parent2, fragment1)

    return child1, child2


# Мутация: обмен двух случайных городов
def mutate(individual):
    if random.random() < mutation_rate:

        print(f"Произошла мутация для {individual}")
        i, j = random.sample(range(n), 2)
        individual[i], individual[j] = individual[j], individual[i]
        print(f"Он стал {individual}")


def select_pairs(population):
    population_size = len(population)
    costs = [calculate_cost(ind) for ind in population]
    fitness = [1 / cost for cost in costs]
    total_fitness = sum(fitness)
    probabilities = [f / total_fitness for f in fitness]

    pairs = set()

    while len(pairs) < (population_size // 2):
        p1 = random.choices(population, weights=probabilities, k=1)[0]
        idx = population.index(p1)
        population_without_p1 = population[:idx] + population[idx + 1:]
        probabilities_without_p1 = probabilities[:idx] + probabilities[idx + 1:]

        if not population_without_p1:
            break

        p2 = random.choices(population_without_p1, weights=probabilities_without_p1, k=1)[0]

        p1_h = tuple(p1)
        p2_h = tuple(p2)

        # Сортируем кортежи по содержимому для уникальности пар
        sorted_pair = tuple(sorted([p1_h, p2_h]))

        if sorted_pair not in pairs:
            pairs.add(sorted_pair)

    return list(pairs)

# Создание начальной популяции
population = [generate_individual() for _ in range(population_size)]

# Эволюция одного поколения
def evolve(population):
    # Оценка
    population = sorted(population, key=calculate_cost)
    # Случайный выбор пар для скрещивания
    pairs = select_pairs(population)
    children = []
    print("\n")
    for p1, p2 in pairs:
        c1, c2 = crossover(p1, p2)
        mutate(c1)
        mutate(c2)
        children.extend([c1, c2])

    # Новая популяция: выбираем 4 лучших из родителей и детей
    new_population = population + children
    new_population = sorted(new_population, key=calculate_cost)[:population_size]

    print("\n Generation")

    for ind in new_population:
        print(ind, "Cost:", calculate_cost(ind))

    return new_population

# Вывод начальной популяции
print("Initial population:")
for ind in population:
    print(ind, "Cost:", calculate_cost(ind))

def calculate_avg_cost(population):
    costs = [calculate_cost(ind) for ind in population]
    return sum(costs) / len(costs)

initial_avg = calculate_avg_cost(population)
print(f"Средняя стоимость в начальной популяции: {initial_avg:.2f}\n")

print("Начало алгоритма")

# Эволюция нескольких поколений
for generation in range(max_generations):
    population = evolve(population)

# Вывод финальной популяции
print("\nPopulation after", max_generations, "generations:")
for ind in population:
    print(ind, "Cost:", calculate_cost(ind))

final_avg = calculate_avg_cost(population)
print(f"\nСредняя стоимость в финальной популяции: {final_avg:.2f}")
