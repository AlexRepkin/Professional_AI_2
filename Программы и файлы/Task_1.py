#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import random
from collections import deque


class IslandCounter:
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    """ Подсчёт количества островов в бинарной матрице"""
    def count_islands(self):
        # Матрица для отслеживания посещённых клеток
        visited = [[False for _ in range(self.cols)] for _ in range(self.rows)]
        island_count = 0
        # Проход по всем клеткам матрицы
        for row in range(self.rows):
            for col in range(self.cols):
                # Если найдена непосещённая клетка суши, происходит подсчёт островов
                if self.grid[row][col] == 1 and not visited[row][col]:
                    self.width_search(row, col, visited)
                    island_count += 1
        return island_count

    """Поиск в ширину для текущего острова, помечая все клетки суши как посещённые"""
    def width_search(self, row, col, visited):
        # Все возможные направления для перемещения (Включая диагональные)
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        # Для поиска в ширину используется очередь. Для упрощения кода решено использовать очередь,
        # которую можно редактировать как слева, так и справа - deque
        queue = deque([(row, col)])
        visited[row][col] = True
        # Пока очередь не пуста - проверка
        while queue:
            current_row, current_col = queue.popleft()
            # Проверка всех 9 клеток вокруг
            for step_for_row, step_for_col in directions:
                changed_row = current_row + step_for_row
                changed_col = current_col + step_for_col
                # Проверка границ матрицы и состояния клетки
                if (0 <= changed_row < self.rows and 0 <= changed_col < self.cols and
                        self.grid[changed_row][changed_col] == 1 and not visited[changed_row][changed_col]):
                    visited[changed_row][changed_col] = True  # Клетка отмечена как посещённая
                    queue.append((changed_row, changed_col))  # Клетка добавлена в очередь для последующей проверки


def print_map(map):
    for i, row in enumerate(map):
        for j, cell in enumerate(row):
            # Зелёный - суша, Синий - вода.
            if cell == 0:
                print("\033[44m  \033[0m", end="")
            else:
                print("\033[42m  \033[0m", end="")
        print()
    print("\033[0m")  # Сброс цвета


if __name__ == '__main__':
    print("Пример №1:")
    map = [
        [1, 0, 0, 0, 1],
        [1, 1, 0, 1, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 1, 0, 1]
    ]
    print_map(map)
    island_counter = IslandCounter(map)
    print("Общее количество островов:", island_counter.count_islands())

    print("Пример №2:")
    random_map = []
    random_width = int(input("Введите ширину карты:"))
    random_height = int(input("Введите длину карты:"))
    for _ in range(0, random_width):
        random_map.append([])
        for j in range(0, random_height):
            random_map[_].append(random.randint(0, 1))
    print_map(random_map)
    print("Общее количество островов:", IslandCounter(random_map).count_islands())
