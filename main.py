from BayesNet import BayesNet
from BNReasoner import BNReasoner
import copy
import networkx as nx
import pandas as pd
from utils import powerset
import matplotlib.pyplot as plt
import timeit

testing_path=[ "testing/dog_problem.BIFXML",
               "testing/lecture_example.BIFXML",
               "testing/lecture_example2.BIFXML",
               "testing/task3.BIFXML"]

exp_path=["bifxml_files/large_networks/win95pts.bifxml",
          "bifxml_files/medium_networks/medium.BIFXML",
          "bifxml_files/small_networks/asia.bifxml",
          "bifxml_files/small_networks/cancer.bifxml"]

def test_prune():
    new_BN = BNReasoner(testing_path[1])
    # new_BN.bn.draw_structure()

    # new_BN.d_sep(["Sprinkler?"],["Slippery Road?"],["Rain?"])

    d = {'Winter?': True, 'Rain?': False}
    ev = pd.Series(data=d, index=['Winter?', 'Rain?'])

    query_vars = {"Wet Grass?"}
    new_BN.prune(query_vars, ev)

    new_BN.bn.draw_structure()
    print(new_BN.bn.get_cpt("Sprinkler?"))
    print("------------")
    print(new_BN.bn.get_cpt("Rain?"))
    print("------------")
    print(new_BN.bn.get_cpt("Wet Grass?"))

def plot():
    #running time MPE 312
    random_rt = [0.2325157,0.2938359,0.25270,0.30720,2.46622]
    minFill_rt = [0.166,0.14474,0.12757,0.18353,3.43713]
    minDegree_rt = [0.1382,0.14358,0.151,0.19926,2.691]

    no_variables = [5,5,5,8,10]

    fig, ax = plt.subplots()
    ax.plot(no_variables, random_rt, label="random")
    ax.plot(no_variables, minDegree_rt, label="minDegree")
    ax.plot(no_variables, minFill_rt, label="minFill")
    ax.legend()

    plt.show()

def main():



    BN = BNReasoner(exp_path[2])
    # #BN.bn.draw_structure()

    d = {'dysp': True }
    evidence = pd.Series(data=d, index=['dysp'])

    start = timeit.default_timer()
    #print(BN.marginal_distribution({'Paralysis'}, evidence, 2))
    print(BN.MAP_MPE({}, evidence, 1))
    stop = timeit.default_timer()

    print('Time: ', stop - start)
    #plot()

if __name__ == '__main__':
    main()


