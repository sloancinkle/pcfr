import numpy as np
import pandas as pd

from random import choices


class PotentialRegretMinimizer:
    def __init__(self, actions):
        self.num_actions = len(actions)
        self.regret_sum = {a: 1 for a in actions}

        self.strategy_sum = {a: 0 for a in actions}
        self.reach_sum = 0

        self.action_util: dict[float] = {}
        self.infostate_util = 0

    def reset_utilities(self):
        self.action_util: dict[float] = {}
        self.infostate_util = 0

    def get_next_strategy(self):
        positive_regrets = {a: max(self.regret_sum[a], 0)
                            for a in self.regret_sum}
        total = sum(positive_regrets.values())
        return {a: (positive_regrets[a] / total)
                if total > 0 else (1 / self.num_actions)
                for a in self.regret_sum}

    def update_regrets(self, action, reach):
        imm_regret = (self.action_util[action] - self.infostate_util)
        self.regret_sum[action] += imm_regret * reach

    def update_strategy_sum(self, strategy, reach, t):
        for a in strategy:
            self.strategy_sum[a] += strategy[a] * reach * t
        self.reach_sum += reach * t

    def get_average_strategy(self):
        return {a: (self.strategy_sum[a] / self.reach_sum)
                if self.reach_sum > 0 else (1 / self.num_actions)
                for a in self.strategy_sum}

    def get_average_regret(self, t):
        return max(self.regret_sum.values()) / t if self.num_actions > 0 else 0


class PotentialCFR:
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
            return self.game.get_potential(), reach

        infostate = self.game.get_infostate()
        actions = self.game.get_valid_actions()

        player = self.game.get_next_player()
        if (player, infostate) not in self.regret_minimizers:
            self.regret_minimizers[(player, infostate)] = PotentialRegretMinimizer(actions)
        rm = self.regret_minimizers[(player, infostate)]
        strategy = rm.get_next_strategy()
        rm.update_strategy_sum(strategy, info_reach, self.t)

        rm.reset_utilities()
        action = choices(list(strategy.keys()), weights=list(strategy.values()))[0]

        all_strategies = []
        if not any(self.game.get_scheduled_actions()):
            all_strategies = self.all_strategies.copy()
            self.all_strategies = [1] * self.game.get_num_players()
        new_reach = reach * strategy[action]
        self.all_strategies[player] = strategy[action]
        self.last_time[player] = action

        self.game.take_action(action)
        if self.game.get_next_player() <= player:
            rm.action_util[action], action_reach = self.walk_trees(new_reach, new_reach)
        else:
            rm.action_util[action], action_reach = self.walk_trees(new_reach, info_reach)
        self.game.undo_action()

        other_reach = np.prod([s for i, s in enumerate(self.all_strategies) if i != player])
        rm.infostate_util += rm.action_util[action] * strategy[action] * other_reach
        rm.update_regrets(action, action_reach)

        if not any(self.game.get_scheduled_actions()):
            self.all_strategies = all_strategies
            new_reach = reach
            utility = rm.infostate_util
        else:
            new_reach = action_reach
            utility = rm.action_util[action]

        return utility, new_reach

    def get_average_strategies(self):
        avg_strategies = {}
        info_hits = {}
        for i in self.regret_minimizers:
            rm = self.regret_minimizers[i]
            avg_strategy = rm.get_average_strategy()

            if i[1][0] not in avg_strategies:
                avg_strategies[i[1][0]] = {a: avg_strategy[a] for a in avg_strategy}
                info_hits[i[1][0]] = 1
            else:
                for j in avg_strategies[i[1][0]]:
                    avg_strategies[i[1][0]][j] = (info_hits[i[1][0]] * avg_strategies[i[1][0]][j] + avg_strategy[j]) / \
                                                 (info_hits[i[1][0]] + 1)
                info_hits[i[1][0]] += 1
        return avg_strategies

    def get_overall_regret(self):
        overall_regret = 0
        for i in self.regret_minimizers:
            rm = self.regret_minimizers[i]
            regret = rm.get_average_regret(self.t)
            overall_regret += max(0, regret)
        return overall_regret

    def get_regret_table(self):
        colnames = ["Iteration", "Regret"]
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
            self.regret_table += [[self.t, regret]]
            self.strategy_list += [self.get_average_strategies()]
        return regret

    def train(self, iterations=1, epsilon=np.inf, update_rate=1):
        self.t = 0
        num_players = self.game.get_num_players()

        regret = self.training_iteration(update_rate)
        while self.t < iterations or regret > epsilon:
            regret = self.training_iteration(update_rate)

        return self.t


