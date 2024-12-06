#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
from collections import deque

'''Льющиеся кувшины. Fill - наполнить полностью кувшин;
Dump - полностью вылить воду из кувшина;
Pour - перелить воду из кувшина в кувшин.'''


class JarProblem:
    def __init__(self, initial, goal, sizes):
        self.initial = initial  # Объёмы воды в кувшинах, по условию
        self.goal = goal  # Сколько нужно воды
        self.sizes = sizes  # Объёмы кувшинов

    def actions(self, state):
        # state - текущая заполненность кувшинов
        actions = []
        for i in range(len(state)):
            if state[i] < self.sizes[i]:  # Заполнить кувшин
                actions.append(('Fill', i))
            if state[i] > 0:  # Вылить воду из кувшина
                actions.append(('Dump', i))
            for j in range(len(state)):
                if i != j and state[i] > 0 and state[j] < self.sizes[j]:  # Перелить воду
                    actions.append(('Pour', i, j))
        return actions

    # Новое состояние после выполнения actions.
    def result(self, state, action):
        state = list(state)
        if action[0] == 'Fill':  # Наполнить кувшин
            state[action[1]] = self.sizes[action[1]]
        elif action[0] == 'Dump':  # Опустошить кувшин
            state[action[1]] = 0
        elif action[0] == 'Pour':  # Перелить воду из одного кувшина в другой
            i, j = action[1], action[2]
            pour_amount = min(state[i], self.sizes[j] - state[j])
            state[i] -= pour_amount
            state[j] += pour_amount
        return tuple(state)

    def check_water(self, state):
        return self.goal in state


def bfs(problem):
    node = (problem.initial, [])  # Узел, у которого указано состояние и последовательность действий
    if problem.check_water(node[0]):  # Проверка, если в кувшине уже нужный объём
        return node[1]
    frontier = deque([node])  # Очередь для анализа узлов
    explored = set()  # Множество для хранения посещенных состояний кувшинов
    while frontier:
        state, path = frontier.popleft()  # Извлечение состояния из очереди
        explored.add(state)  # Пометка состояния как посещенного
        for action in problem.actions(state):  # Перебор всех доступных действий
            child_state = problem.result(state, action)  # Новое состояние
            if child_state not in explored:  # Если состояние еще не посещено
                if problem.check_water(child_state):  # Проверка, достигнута ли цель
                    return path + [action]
                frontier.append((child_state, path + [action]))  # Добавление в очередь
                explored.add(child_state)  # Помечено как посещённое
    return None


if __name__ == '__main__':
    initial = (0, 0)  # Начальные уровни воды в кувшинах
    goal = 4  # Нужный объем воды
    sizes = (5, 3)  # Размеры кувшинов
    if goal > max(sizes):  # Если нужный объем > максимального объема кувшинов
        print(f"Невозможно получить {goal} , так как нет кувшина достаточного объёма.")
    else:
        solution = bfs(JarProblem(initial, goal, sizes))
        if solution:
            print("Полученная последовательность:", solution)  # Вывод последовательности действий
        else:
            print("Не удалось найти последовательность.")

    random_amount = random.randint(2,6)
    random_sizes = list()
    random_initial = list()
    random_sizes.append(random.randint(1, 15))
    random_initial.append(random.randint(0, random_sizes[0]))
    for i in range(1, random_amount):
        random_sizes.append(random.randint(1, 15))
        random_initial.append(random.randint(0,random_sizes[i]))
    random_goal = random.randint(1, max(random_sizes))
    print(f"\nРандомные:\nРазмеры:{random_sizes},\nНачальные значения: {random_initial},\nНужный объём: {random_goal}.")
    solution = bfs(JarProblem(tuple(random_initial), random_goal, random_sizes))
    if solution:
        print("Полученная последовательность:", solution)  # Вывод последовательности действий
    else:
        print("Не удалось найти последовательность.")
