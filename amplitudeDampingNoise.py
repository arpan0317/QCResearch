#!/usr/bin/env python

from pgmpy.models import BayesianNetwork
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference import VariableElimination
from math import sqrt

q0m0 = TabularCPD (
    variable = 'q0m0',
    variable_card = 4,
    values = [[1/2], [-1/2], [0], [0]],
    state_names={'q0m0': ['I', 'X', 'Y', 'Z']}
)

rv = TabularCPD (
    variable = 'rv',
    variable_card = 2,
    values = [[(1+sqrt(1-9/25))/2], [(1-sqrt(1-9/25))/2]],
    state_names={'rv': ['I', 'AD']}
)
rv0 = TabularCPD (
    variable = 'rv0',
    variable_card = 2,
    values = [[1, 0], [0, 1]],
    evidence = [ 'rv' ],
    evidence_card = [2],
    state_names={'rv': ['I', 'AD'], 'rv0': ['I', 'AD']}
)
rv1 = TabularCPD (
    variable = 'rv1',
    variable_card = 2,
    values = [[1, 0], [0, 1]],
    evidence = [ 'rv' ],
    evidence_card = [2],
    state_names={'rv': ['I', 'AD'], 'rv1': ['I', 'AD']}
)

# print(rv0)

q0m1 = TabularCPD (
    variable='q0m1',
    variable_card = 4,
    values = [
        [ 1,0,0,0, 1,0,0,0 ],
        [ 0,1,0,0, 0,-1,0,0 ],
        [ 0,0,1,0, 0,0,-1,0 ],
        [ -1,0,0,1, 1,0,0,-1 ],
    ],
    evidence = [ 'rv0', 'q0m0' ],
    evidence_card = [2,4],
    state_names={'rv0': ['I', 'AD'], 'q0m0': ['I', 'X', 'Y', 'Z'], 'q0m1': ['I', 'X', 'Y', 'Z']}
)

q0m2 = TabularCPD (
    variable='q0m2',
    variable_card = 4,
    values = [
        [ 1,0,0,0, 1,0,0,0 ],
        [ 0,1,0,0, 0,1,0,0 ],
        [ 0,0,1,0, 0,0,1,0 ],
        [ 1,0,0,1, 1,0,0,-1 ],
    ],
    evidence = [ 'rv1', 'q0m1' ],
    evidence_card = [2,4],
    state_names={'rv1': ['I', 'AD'], 'q0m1': ['I', 'X', 'Y', 'Z'], 'q0m2': ['I', 'X', 'Y', 'Z']}
)

# meas = TabularCPD (
#     variable='meas',
#     variable_card = 2,
#     values = [
#         [ 1,1,0,0 ],
#         [ 1,-1,0,0 ],
#     ],
#     evidence = ['q0m1'],
#     evidence_card = [4],
#     state_names={'q0m1': ['I', 'X', 'Y', 'Z'], 'meas': ['|+><+|', '|-><-|']}
# )

AmplitudeDamping = BayesianNetwork([
    ('q0m0', 'q0m1'),
    ('rv', 'rv0'),
    ('rv0', 'q0m1'),
    ('q0m1', 'q0m2'),
    ('rv', 'rv1'),
    ('rv1', 'q0m2'),
    # ('q0m2', 'meas'),
])

AmplitudeDamping.add_cpds(
    q0m0,
    rv,
    rv0,
    q0m1,
    rv1,
    q0m2,
    # meas
)
outcome = VariableElimination(AmplitudeDamping.to_markov_model()).query(['q0m2'])
print(type(outcome))
print(outcome)

exit()

AmplitudeDampingParam = BayesianNetwork([
    ('q0m0', 'q0m1'),
    ('rv', 'q0m1'),
    # ('q0m1', 'meas'),
])
AmplitudeDampingParam.add_cpds(q0m0,q0m1)
AmplitudeDampingParamMN = AmplitudeDampingParam.to_markov_model()
AmplitudeDampingParamMN.add_factors(outcome)
print(VariableElimination(AmplitudeDampingParamMN).query(['rv']))