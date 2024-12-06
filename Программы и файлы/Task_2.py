#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math
import random
from collections import deque


class KruskalMaze:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.nodes, self.edges = self.create_graph()
        self.maze = self.generate_maze()
        self.grid = self.create_grid()
        self.carve_paths()

    def get_random_edge_weights(self):
        edge_weights = [(random.randint(1, 4), x, y) for (x, y) in self.edges]
        return edge_weights

    def create_graph(self):
        nodes = set()
        edges = set()
        for i in range(self.width):
            for j in range(self.height):
                nodes.add((i, j))
                if i > 0:
                    edges.add(((i, j), (i - 1, j)))
                if i < self.width - 1:
                    edges.add(((i, j), (i + 1, j)))
                if j > 0:
                    edges.add(((i, j), (i, j - 1)))
                if j < self.height - 1:
                    edges.add(((i, j), (i, j + 1)))
        return nodes, edges

    def generate_maze(self):
        edge_weights = self.get_random_edge_weights()
        clusters = {n: n for n in self.nodes}
        ranks = {n: 0 for n in self.nodes}
        solution = set()

        def find(u):
            if clusters[u] != u:
                clusters[u] = find(clusters[u])
            return clusters[u]

        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if ranks[root_x] > ranks[root_y]:
                clusters[root_y] = root_x
            else:
                clusters[root_x] = root_y
            if ranks[root_x] == ranks[root_y]:
                ranks[root_y] += 1

        for _, x, y in sorted(edge_weights):
            if find(x) != find(y):
                solution.add((x, y))
                union(x, y)

        return solution

    def create_grid(self):
        """Creates a grid filled with walls (0s)."""
        return [[0 for _ in range(self.width)] for _ in range(self.height)]

    def carve_paths(self):
        """Marks paths in the grid (1s) based on the generated maze."""
        for (x, y) in self.nodes:
            self.grid[y][x] = 1  # Mark all nodes as paths initially

        for (start, end) in self.edges:
            if (start, end) not in self.maze and (end, start) not in self.maze:
                x1, y1 = start
                x2, y2 = end
                self.grid[y1][x1] = 0
                self.grid[y2][x2] = 0

    def get_maze(self):
        """Returns the grid as the final maze structure."""
        return self.grid


# Базовый класс, описывающий задачу поиска пути в лабиринте. Составлен согласно методическим указаниям.
class Problem:
    def __init__(self, initial=None, goal=None, **kwds):
        # Инициализация начального и конечного состояния
        self.__dict__.update(initial=initial, goal=goal, **kwds)

    def actions(self, state):
        # Метод для получения доступных действий
        raise NotImplementedError

    def result(self, state, action):
        # Метод для получения результата действия
        raise NotImplementedError

    def is_goal(self, state):
        # Проверка, является ли состояние нужным
        return state == self.goal

    def action_cost(self):
        # Стоимость действия (по умолчанию - 1)
        return 1

    @staticmethod
    def h():
        # Эвристика (по умолчанию - 0)
        return 0


# Класс Node описывает узел в графе, включая состояние, родителя и действие
class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        # Инициализация параметров узла
        self.__dict__.update(state=state, parent=parent, action=action, path_cost=path_cost)

    def __len__(self):
        # Определение длины пути от корневого узла
        return 0 if self.parent is None else (1 + len(self.parent))


# Поиск в ширину для нахождения пути в лабиринте
def width_first_search(problem):
    node = Node(problem.initial)
    if problem.is_goal(problem.initial):
        return node
    frontier = deque([node])  # Очередь для хранения узлов
    reached = {problem.initial}  # Множество посещенных состояний
    while frontier:
        node = frontier.popleft()  # Извлеченме из начала, так как это очередь
        for child in expand(problem, node):
            s = child.state
            if problem.is_goal(s):
                return child
            if s not in reached:
                reached.add(s)
                frontier.append(child)  # Добавление в конец очереди
    return Node('failure', path_cost=math.inf)


# Генерация возможных переходов между состояниями
def expand(problem, node):
    s = node.state
    for action in problem.actions(s):
        s1 = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action, s1)
        yield Node(s1, node, action, cost)


# Определение длины пути до точки
def path_length(node):
    return len(path_states(node)) - 1 if node else math.inf


# Получение списка состояний для найденного пути
def path_states(node):
    if node.parent is None:
        return [node.state]
    return path_states(node.parent) + [node.state]


# Подкласс Problem для конкретного лабиринта
class MazeProblem(Problem):
    def __init__(self, maze, initial, goal):
        super().__init__(initial=initial, goal=goal)
        self.maze = maze
        self.height = len(maze)
        self.width = len(maze[0]) if self.height > 0 else 0

    def actions(self, state):
        # Доступные перемещения в соседние клетки
        x, y = state
        possible_moves = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.height and 0 <= ny < self.width and self.maze[nx][ny] == 1:
                possible_moves.append((dx, dy))
        return possible_moves

    def result(self, state, action):
        # Получение нового состояния после действия
        x, y = state
        dx, dy = action
        return x + dx, y + dy


def print_maze(maze, path=None):
    for i, row in enumerate(maze):
        for j, cell in enumerate(row):
            # Чёрный - стены, жёлтый - проход, красный - кратчайший путьы
            if cell == 0:
                print("\033[40m  \033[0m", end="")  # Черный фон
            elif path and (i, j) in path:
                print("\033[41m  \033[0m", end="")  # Красный фон
            else:
                print("\033[43m  \033[0m", end="")  # Желтый фон
        print()
    print("\033[0m")  # Сброс цвета


if __name__ == '__main__':
    # Пример 1: Лабиринт 8x12
    maze_1 = [
        [1, 0, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 0, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 1, 1],
        [1, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ]
    initial_1 = (0, 0)
    goal_1 = (8, 2)  # Конечная точка
    problem_1 = MazeProblem(maze_1, initial_1, goal_1)
    print("Пример 1 (8x12) - Исходный лабиринт:")
    print_maze(maze_1)  # Лабиринт
    # Поиск кратчайшего пути
    solution_1 = width_first_search(problem_1)
    if solution_1.state == "failure":
        print("Нет пути до выхода.")
    else:
        path_1 = path_states(solution_1)
        print("Кратчайший путь:", path_1, "\nДлина пути:", path_length(solution_1))
        print("Лабиринт (8x12) с кратчайшим путем:")
        print_maze(maze_1, path_1)

    # Лабиринт 15x7
    maze_2 = [
        [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
        [1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
        [1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
    initial_2 = (0, 0)
    goal_2 = (6, 14)
    problem_2 = MazeProblem(maze_2, initial_2, goal_2)
    print("\nПример 2 (15x7) - Исходный лабиринт:")
    print_maze(maze_2)  # Лабиринт
    # Поиск кратчайшего пути
    solution_2 = width_first_search(problem_2)
    if solution_2.state == "failure":
        print("Нет пути до выхода.")
    else:
        path_2 = path_states(solution_2)
        print("Кратчайший путь:", path_2, "\nДлина пути:", path_length(solution_2))
        print("Лабиринт (15x7) с кратчайшим путем:")
        print_maze(maze_2, path_2)

    # random_maze = KruskalMaze(int(input("Введите ширину лабиринта:")),
    # int(input("Введите длину лабиринта:"))).get_maze()
    random_maze = []
    random_width = int(input("Введите ширину лабиринта:"))
    random_height = int(input("Введите длину лабиринта:"))
    for _ in range(0, random_height):
        random_maze.append([])
        for j in range(0, random_width):
            random_maze[_].append(random.randint(0, 1))
    print("\nПример 3 - Рандомный лабиринт:")
    print_maze(random_maze)  # Лабиринт
    # Поиск кратчайшего пути
    random_initial = (int(input("Введите начальный y:")) - 1, int(input("Введите начальный x:")) - 1)
    random_goal = (int(input("Введите конечный y:")) - 1, int(input("Введите конечный x:")) - 1)
    random_problem = MazeProblem(random_maze, random_initial, random_goal)
    random_solution = width_first_search(random_problem)
    if random_solution.state == "failure":
        print("Нет пути до выхода.")
    else:
        random_path = path_states(random_solution)
        print("Кратчайший путь:", random_path, "\nДлина пути:", path_length(random_solution))
        print("Лабиринт с кратчайшим путем:")
        print_maze(random_maze, random_path)
