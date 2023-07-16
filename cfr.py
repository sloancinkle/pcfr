import numpy as np
import pandas as pd

from random import choices


class RegretMinimizer:
    def __init__(self, actions, num_players):
        self.num_actions = len(actions)

        self.regret_sum = {a: 0 for a in actions}
        self.strategy_sum = {a: 0 for a in actions}
        self.reach_sum = 0

        self.action_util: dict[list[float]] = {}
        self.infostate_util = [0] * num_players

    def reset_utilities(self):
        self.action_util: dict[list[float]] = {}
        self.infostate_util = [0] * len(self.infostate_util)

    def get_next_strategy(self):
        positive_regrets = {a: max(self.regret_sum[a], 0) for a in self.regret_sum}
        total = sum(positive_regrets.values())
        return {a: (positive_regrets[a] / total)
                if total > 0 else (1 / self.num_actions)
                for a in self.regret_sum}

    def update_regret(self, player, reach):
        for a in self.regret_sum:
            action_utility = self.action_util[a][player]
            imm_regret = (action_utility - self.infostate_util[player]) * reach
            self.regret_sum[a] += imm_regret
            pass

    def update_strategy_sum(self, strategy, reach, t):
        for a in self.strategy_sum:
            self.strategy_sum[a] += strategy[a] * reach * t
        self.reach_sum += reach * t
        pass

    def get_average_strategy(self):
        return {a: (self.strategy_sum[a] / self.reach_sum)
                if self.reach_sum > 0 else (1 / self.num_actions)
                for a in self.strategy_sum}

    def get_average_regret(self, t):
        return max(self.regret_sum.values()) / t if self.num_actions > 0 else 0


class CFR:
    def __init__(self, game):
        self.game = game
        self.regret_minimizers = {}
        self.t = 1
        self.regret_table = []
        self.strategy_list = []
        self.all_strategies = [1] * game.get_num_players()
        self.last_time = [""] * game.get_num_players()

    def walk_trees(self, reach=1, info_reach=1):
        if self.game.is_terminal():
            u = self.game.get_utility()
            return u, reach

        player = self.game.get_next_player()
        infostate = self.game.get_infostate()
        actions = self.game.get_valid_actions()

        if (player, infostate) not in self.regret_minimizers:
            self.regret_minimizers[(player, infostate)] = \
                RegretMinimizer(actions, self.game.get_num_players())
        rm = self.regret_minimizers[(player, infostate)]
        strategy = rm.get_next_strategy()
        rm.update_strategy_sum(strategy, info_reach, self.t)

        rm.reset_utilities()
        if not any(self.game.get_scheduled_actions()):
            all_strategies = self.all_strategies.copy()
            self.all_strategies = [1] * self.game.get_num_players()

        for a in actions:
            new_reach = reach * strategy[a]

            self.all_strategies[player] = strategy[a]
            self.last_time[player] = a

            self.game.take_action(a)

            if self.game.get_next_player() <= player:
                rm.action_util[a], action_reach = self.walk_trees(new_reach, new_reach)
            else:
                rm.action_util[a], action_reach = self.walk_trees(new_reach, info_reach)

            self.game.undo_action()

            other_reach = np.prod([s for i, s in enumerate(self.all_strategies) if i != player])
            action_util = [u_i * strategy[a] * other_reach for u_i in rm.action_util[a]]
            rm.infostate_util = np.add(rm.infostate_util, action_util)
            pass

        rm.update_regret(player, action_reach)

        if not any(self.game.get_scheduled_actions()):
            self.all_strategies = all_strategies
            action_reach = reach

        return rm.infostate_util, action_reach

    def get_average_strategies(self):
        num_players = self.game.get_num_players()
        avg_strategies = {}
        for i in self.regret_minimizers:
            rm = self.regret_minimizers[i]
            avg_strategy = rm.get_average_strategy()

            if i[1] not in avg_strategies:
                avg_strategies[i[1]] = {a: avg_strategy[a] / num_players for a in avg_strategy}
            else:
                for j in avg_strategies[i[1]]:
                    avg_strategies[i[1]][j] += avg_strategy[j] / num_players

        return avg_strategies

    def get_overall_regret(self):
        overall_regret = [0] * self.game.get_num_players()
        for i in self.regret_minimizers:
            rm = self.regret_minimizers[i]
            regret = rm.get_average_regret(self.t)
            overall_regret[i[0]] += max(0, regret)
        return overall_regret

    def get_regret_table(self):
        colnames = ["Iteration", "Player 1", "Player 2"]
        if self.game.num_players == 3:
            colnames += ["Player 3"]
        return pd.DataFrame(self.regret_table, columns=colnames)

    def get_strategy_list(self):
        return self.strategy_list

    def training_iteration(self, update_rate):
        self.t += 1
        for state in self.game.get_all_start_states():
            self.game.reset(start_state=state)
            self.walk_trees()
        regret = self.get_overall_regret()
        if self.t % update_rate == 0:
            self.regret_table += [[self.t] + regret]
            self.strategy_list += [self.get_average_strategies()]
        return regret

    def train(self, iterations=1, epsilon=np.inf, update_rate=1):
        self.t = 0
        num_players = self.game.get_num_players()

        regret = self.training_iteration(update_rate)
        equilibrium = [r < epsilon / num_players for r in regret]

        while self.t < iterations or not all(equilibrium):
            regret = self.training_iteration(update_rate)
            equilibrium = [r < epsilon / num_players for r in regret]

        return self.t



