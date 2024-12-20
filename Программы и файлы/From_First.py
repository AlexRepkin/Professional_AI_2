#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from collections import deque


class Node:

    def __init__(self, state, parent=None, action=None, path_cost=0.0):
        self.__dict__.update(state=state, parent=parent, action=action,
                             path_cost=path_cost)

    def __repr__(self):
        return '<{}>'.format(self.state)

    def __len__(self):
        return 0 if self.parent is None else (1 + len(self.parent))

    def __lt__(self, other):
        return self.path_cost < other.path_cost


def expand(matrix, tree, node):
    # Раскрытие узла, создавая дочерние узлы
    s = node.state
    last_node = node.action
    lst_nodes = tree[int(last_node)]
    result = []
    for item in lst_nodes:
        temp = s.copy()
        if item not in s:  # ИЧтобы не зациклилось
            temp.append(item)
            dist = round(matrix[temp[-2] - 1][temp[-1] - 1] + node.path_cost, 6)
            result.append(Node(temp, node, item, dist))
    return result


tree = {1: [2, 3], 2: [1, 5], 3: [1, 4], 4: [3, 5, 6, 7], 5: [2, 4, 9], 6: [4, 8, 9],
        7: [4, 8], 8: [6, 7, 9], 9: [5, 6, 8, 10, 12, 15, 22], 10: [9, 11],
        11: [10], 12: [9, 13, 14], 13: [12], 14: [12], 15: [9, 16, 17],
        16: [15, 21], 17: [15, 18], 18: [17, 19], 19: [18, 20], 20: [19], 21: [16], 22: [9]}

matrix = [[0.0, 71.0, 72.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [71.0, 0.0, 0.0, 0.0, 48.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [72.0, 0.0, 0.0, 43.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 43.0, 0.0, 87, 38, 110, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 48.0, 0.0, 87.0, 0.0, 0.0, 0.0, 0.0, 108.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 38.0, 0.0, 0.0, 0.0, 109.0, 152.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 110.0, 0.0, 0.0, 0.0, 129.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 109.0, 129.0, 0.0, 45.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 108.0, 152.0, 0.0, 45.0, 0.0, 32.0, 0.0, 131.0, 0.0, 0.0, 63.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 117.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 32.0, 0.0, 170.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 170.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 131.0, 0.0, 0.0, 0.0, 75.0, 124.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 75.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 124.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 63.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 47.0, 86.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 47.0, 0.0, 0.0, 0.0, 0.0, 0.0, 156.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 86.0, 0.0, 0.0, 32.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 32.0, 0.0, 43.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 43.0, 0.0, 115.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 115.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 156.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 117.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
          ]

cities = {1: "Астана", 2: "Аршалы", 3: "Егинди", 4: "Нура", 5: "Осакаровка",
          6: "Кертинды", 7: "Кайнар", 8: "Шахтинск", 9: "Караганда",
          10: "Абай", 11: "Жанаарка", 12: "Аксу-Аюлы", 13: "пос. С. Сейфуллина",
          14: "Актогай", 15: "Ботакара", 16: "Керней", 17: "Матак",
          18: "Коктас", 19: "Каркаралинск", 20: "Егиндыбулак", 21: "Жанатилёк", 22: "Молодёжный"}

if __name__ == '__main__':
    start = 4
    finish = 14
    first = Node([start], None, '1', 0)
    paths = []
    q = deque()
    q.append(first)
    # Основной цикл поиска
    while q:
        temp = expand(matrix, tree, q.popleft())
        for i in temp:
            if i.state[-1] != finish:  # Если не достигли цели - в очередь
                q.append(i)
            else:
                paths.append(i)  # Добавить путь в список возможных маршрутов
    # Поиск минимального пути
    mn = [float('inf'), []]
    for i in paths:
        if i.path_cost < mn[0]:
            mn = [i.path_cost, i.state]

    print(f"Количество возможных путей из города {cities[start]} в город {cities[finish]}: {len(paths)}")
    print(f"Маршрут минимальной длины ({mn[0]} км):")
    for id in mn[1]:
        if id != finish:
            print(cities[id], end=" - ")
        else:
            print(cities[id])
