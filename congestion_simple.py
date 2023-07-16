import numpy as np


COSTS_2P = {"AB": [3, 6], "BD": [5, 5], "AC": [7, 7], "BC": [0, 0], "CD": [2, 4]}

COSTS_3P = {"AB": [2, 3, 5], "BD": [2, 3, 6], "AC": [4, 6, 7], "BC": [1, 2, 8],
            "CD": [1, 5, 6]}


class SimpleCongestionGame:
    def __init__(self, num_players, with_information=False):
        if num_players < 2 or num_players >= 4:
            raise ValueError("Simple Congestion Game is only defined for 2 or 3 "
                             "player games.")

        self.num_players = int(num_players)
        self.with_information = with_information

        self.player = 0
        self.scheduled_actions = [""] * num_players

        self.player_positions = ["A"] * num_players
        self.action_history = []
        self.player_history = []
        self.actions_made = []

        self.max_potential = -np.inf

    def reset(self, start_state=None):
        self.player_positions = start_state if start_state \
                                 else ["A"] * self.num_players
        self.scheduled_actions = [""] * self.num_players

        self.player = 0
        self.action_history = []
        self.actions_made = []

    def get_num_players(self):
        return self.num_players

    def get_accompanying(self):
        return sum(pos == self.player_positions
                   for pos in self.player_positions) - 1

    def get_infostate(self):
        return self.player_positions[self.player]

    def get_all_start_states(self):
        return [["A"] * self.num_players]

    def get_valid_actions(self):
        costs = COSTS_3P if self.num_players == 3 else COSTS_2P
        return [c for c in costs if c[0] == self.player_positions[self.player]]

    def get_next_player(self):
        return self.player

    def get_remaining_players(self):
        return [i for i, pos in enumerate(self.scheduled_actions) if not pos
                and self.player_positions[i] != "D"]

    def is_terminal(self):
        return all(pos == "D" for pos in self.player_positions)

    def make_moves(self):
        actions_made = 0
        for i, a in enumerate(self.scheduled_actions):
            if a:
                self.player_positions[i] = a[-1]
                self.action_history += [a]
                self.player_history += [i]
                actions_made += 1
        self.scheduled_actions = [""] * self.num_players
        self.actions_made += [actions_made]
        remaining_players = self.get_remaining_players()
        self.player = remaining_players[0] if remaining_players else -1

    def take_action(self, action):
        if action in self.get_valid_actions():
            self.scheduled_actions[self.player] = action

            remaining_players = self.get_remaining_players()
            if remaining_players:
                self.player = remaining_players[0]
            else:
                self.make_moves()

    def get_scheduled_actions(self):
        return self.scheduled_actions

    def undo_action(self):
        if self.actions_made and all(not a for a in self.scheduled_actions):
            players = self.player_history[-self.actions_made[-1]:]
            actions = self.action_history[-self.actions_made[-1]:]

            for i in range(self.actions_made[-1]):
                self.player_positions[players[i]] = actions[i][0]
                self.scheduled_actions[players[i]] = actions[i]
            self.player = players[-1]
            self.scheduled_actions[players[-1]] = ""

            self.player_history = self.player_history[:-self.actions_made[-1]]
            self.action_history = self.action_history[:-self.actions_made[-1]]
            self.actions_made.pop(-1)

        elif any(self.scheduled_actions):
            self.player = max(i for i, a in enumerate(self.scheduled_actions) if a)
            self.scheduled_actions[self.player] = ""

    def get_utility(self):
        if self.is_terminal():
            utility = [0] * self.num_players
            costs = COSTS_3P if self.num_players == 3 else COSTS_2P
            congestion = {c: sum(a == c for a in self.action_history) for c in costs}
            for i, a in zip(self.player_history, self.action_history):
                utility[i] -= costs[a][congestion[a]-1]
            average_utility = sum(utility) / self.num_players
            zerosum_utility = [u - average_utility for u in utility]
            return zerosum_utility

    def get_potential(self):
        if self.is_terminal():
            potential = 0
            costs = COSTS_3P if self.num_players == 3 else COSTS_2P
            congestion = {c: sum(a == c for a in self.action_history) for c in costs}
            for move in congestion:
                potential -= sum(costs[move][:congestion[move]])
            if potential > self.max_potential:
                self.max_potential = potential
            result = potential - self.max_potential
            return result
