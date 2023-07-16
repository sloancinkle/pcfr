import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

from cfr_original import OriginalCFR
from cfr_potential import PotentialCFR

from congestion_simple import SimpleCongestionGame
from congestion_complex import ComplexCongestionGame

LINE_WIDTH = 1.5
LINE_STYLES = ['-', '--', '-.', (0, (3, 1, 1, 1, 1, 1)), ':']
LINE_COLORS = ['#47a', '#e67', '#283', '#cb4', '#a37']


def plot_regrets(zerosum, potential, gamename, out_file=""):
    fig, ax = plt.subplots()

    zerosum['Sum'] = zerosum[['Player 1', 'Player 2']].sum(axis=1)
    ax.plot(zerosum['Iteration'], zerosum['Sum'],
            label="Zero-sum",
            linewidth=LINE_WIDTH,
            linestyle=LINE_STYLES[0],
            color=LINE_COLORS[0])

    ax.plot(potential['Iteration'], potential['Regret'],
            label="Potential",
            linewidth=LINE_WIDTH,
            linestyle=LINE_STYLES[1],
            color=LINE_COLORS[1])

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Overall Regret')
    ax.set_title('Average Overall Regret on ' + gamename)
    ax.legend()

    if not out_file:
        plt.show()
    else:
        plt.savefig(out_file)

    difference = zerosum['Sum'] - potential['Regret']
    fig, ax = plt.subplots()
    ax.plot(potential['Iteration'], difference, label='Zero-sum - Potential',
            linewidth=LINE_WIDTH, linestyle=LINE_STYLES[0],
            color=LINE_COLORS[0])

    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    ax.set_xlabel('Iteration')
    ax.set_title('Regret Difference on ' + gamename +
                 (" with Information" if with_information else ""))
    ax.legend()

    if not out_file:
        plt.show()
    else:
        plt.savefig(out_file + "_difference")


def kl_divergence(p, q):
    return sum(p[i] * np.log(p[i] / q[i]) for i in p)


def js_divergence(p, q):
    m = {a: (p[a] + q[a]) / 2 for a in p}
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


def plot_strategies(zerosum, potential, with_information, gamename, out_file=""):
    divergence_avg = []
    for t in range(len(zerosum)):
        divergence_sum = 0
        for i in zerosum[t]:
            divergence_sum += js_divergence(zerosum[t][i], potential[t][i])
        divergence_avg += [divergence_sum / len(zerosum[t])]

    iterations = range(1, len(zerosum) + 1)
    fig, ax = plt.subplots()
    ax.plot(iterations, divergence_avg,
            linewidth=LINE_WIDTH, linestyle=LINE_STYLES[0],
            color=LINE_COLORS[0])

    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    ax.set_xlabel('Iteration')
    ax.set_title('Strategy Divergence between '
                 'Zero-sum and Potential CFR' +
                 (" with Information" if with_information else ""))

    if not out_file:
        plt.show()
    else:
        plt.savefig(out_file)


def plot_exploitability(game, gamename, out_file=""):
    pass


def main(args, out_folder):
    num_players = 2
    epsilon = .0001

    for arg in args:
        if arg == '-3p':
            num_players = 3
        try:
            epsilon = -float(arg)
        except ValueError:
            pass

    if '-complex' in args:
        game = ComplexCongestionGame(num_players)
        gamename = str(num_players) + "-player Complex Congestion Game"
        out_folder += '/complex_' + str(num_players) + "p"
    else:
        game = SimpleCongestionGame(num_players)
        gamename = str(num_players) + "-player Simple Congestion Game"
        out_folder += '/simple_' + str(num_players) + "p"

    zerosum = OriginalCFR(game)
    iterations = zerosum.train(iterations=20000)
    potential = PotentialCFR(game)
    potential.train(iterations=iterations)

    zerosum_regrets = zerosum.get_regret_table()
    potential_regrets = potential.get_regret_table()
    plot_regrets(zerosum_regrets, potential_regrets,
                 gamename, out_folder + "_regret")

    zerosum_strategy = zerosum.get_average_strategies()
    potential_strategy = potential.get_average_strategies()
    print(zerosum_strategy)
    print(potential_strategy)

    # zerosum_strategies = zerosum.get_strategy_list()
    # potential_strategies = potential.get_strategy_list()
    # plot_strategies(zerosum_strategies, potential_strategies, with_information,
    #                 gamename, out_folder + "_strategies")
    #
    # plot_exploitability(game, gamename, out_folder + "_exploitability")


if __name__ == "__main__":
    main(sys.argv, os.path.dirname(os.path.realpath(__file__)) + "/output")
