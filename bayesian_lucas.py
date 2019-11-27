from pomegranate import DiscreteDistribution, ConditionalProbabilityTable, State, BayesianNetwork

Anxiety = DiscreteDistribution({"T": 0.64277, "F": 1 - 0.64277})
Peer_Pressure = DiscreteDistribution({"T": 0.32997, "F": 1 - 0.32997})
Genetics = DiscreteDistribution({"T": 0.15953, "F": 1 - 0.15953})
Born_an_Even_Day = DiscreteDistribution({"T": 0.5, "F": 0.5})
Allergy = DiscreteDistribution({"T": 0.32841, "F": 1 - 0.32841})

attention_genetics = [
    ["T", "T", 0.68706],
    ["T", "F", 0.28956],
    ["F", "T", 1 - 0.68706],
    ["F", "F", 1 - 0.28956],
]

fatigue_lung_cancer_coughing = [
    ["T", "F", "F", 0.35],
    ["T", "T", "F", 0.56],
    ["T", "F", "T", 0.80],
    ["T", "T", "T", 0.89],
    ["F", "F", "F", 1 - 0.35],
    ["F", "T", "F", 1 - 0.56],
    ["F", "F", "T", 1 - 0.80],
    ["F", "T", "T", 1 - 0.89],
]

car_accident_attention_fatigue = [
    ["T", "F", "F", 0.22],
    ["T", "T", "F", 0.77],
    ["T", "F", "T", 0.78],
    ["T", "T", "T", 0.97],
    ["F", "F", "F", 1 - 0.22],
    ["F", "T", "F", 1 - 0.77],
    ["F", "F", "T", 1 - 0.78],
    ["F", "T", "T", 1 - 0.97],
]

smoking_peer_pressure_anxiety = [
    ["T", "F", "F", 0.43],
    ["T", "T", "F", 0.74],
    ["T", "F", "T", 0.86],
    ["T", "T", "T", 0.91],
    ["F", "F", "F", 1 - 0.43],
    ["F", "T", "F", 1 - 0.74],
    ["F", "F", "T", 1 - 0.86],
    ["F", "T", "T", 1 - 0.91],
]

yellow_fingers_smoking = [
    ["T", "T", 0.95],
    ["T", "F", 0.23],
    ["F", "T", 1 - 0.95],
    ["F", "F", 1 - 23],
]

lung_cancer_genetics_smoking = [
    ["T", "F", "F", 0.23],
    ["T", "T", "F", 0.86],
    ["T", "F", "T", 0.83],
    ["T", "T", "T", 0.99],
    ["F", "F", "F", 1 - 0.23],
    ["F", "T", "F", 1 - 0.86],
    ["F", "F", "T", 1 - 0.83],
    ["F", "T", "T", 1 - 0.99],
]
# coughing_allergy = [
#     ["T", "T", 0.815],
#     ["T", "F", 0.445],
#     ["F", "T", 1 - 0.815],
#     ["F", "F", 1 - 0.445],
# ]
# fatigue_coughing = [
#     ["T", "T", 0.845],
#     ["T", "F", 0.45863],
#     ["F", "T", 1 - 0.845],
#     ["F", "F", 1 - 0.45863],
# ]
coughing_allergy_lung_cancer = [
    ["T", "F", "F", 0.13],
    ["T", "T", "F", 0.64],
    ["T", "F", "T", 0.76],
    ["T", "T", "T", 0.99],
    ["F", "F", "F", 1 - 0.13],
    ["F", "T", "F", 1 - 0.64],
    ["F", "F", "T", 1 - 0.76],
    ["F", "T", "T", 1 - 0.99],
]

Attention = ConditionalProbabilityTable(table=attention_genetics, parents=[Genetics])
Smoking = ConditionalProbabilityTable(table=smoking_peer_pressure_anxiety,
                                                            parents=[Peer_Pressure, Anxiety])
Lung_Cancer = ConditionalProbabilityTable(table=lung_cancer_genetics_smoking,
                                                           parents=[Genetics, Smoking])
Coughing = ConditionalProbabilityTable(table=coughing_allergy_lung_cancer,
                                                           parents=[Allergy, Lung_Cancer])
Yellow_Fingers = ConditionalProbabilityTable(table=yellow_fingers_smoking,
                                                             parents=[Smoking])
Fatigue = ConditionalProbabilityTable(table=fatigue_lung_cancer_coughing, parents=[Lung_Cancer,Coughing])
Car_Accident = ConditionalProbabilityTable(table=car_accident_attention_fatigue,
                                                             parents=[Attention, Fatigue])

states = {}
states['Anxiety'] = State(Anxiety, name="Anxiety")
states['Peer_Pressure'] = State(Peer_Pressure, name="Peer_Pressure")
states['Smoking'] = State(Smoking, name="Smoking")
states['Yellow_Fingers'] = State(Yellow_Fingers, name="Yellow_Fingers")
states['Genetics'] = State(Genetics, name="Genetics")
states['Lung_Cancer'] = State(Lung_Cancer, name="Lung_Cancer")
states['Attention'] = State(Attention, name="Attention")
states['Allergy'] = State(Allergy, name="Allergy")
states['Coughing'] = State(Coughing, name="Coughing")
states['Born_an_Even_Day'] = State(Born_an_Even_Day, name="Born_an_Even_Day")

states['Fatigue'] = State(Fatigue, name="Fatigue")
states['Car_Accident' ] = State(Car_Accident, name="Car_Accident")

network = BayesianNetwork("Monty hall problem")
network.add_states(*states.values())
network.add_edge(states["Peer_Pressure"],states["Smoking"])
network.add_edge(states["Anxiety"],states["Smoking"])
network.add_edge(states["Smoking"],states["Yellow_Fingers"])
network.add_edge(states["Genetics"],states["Lung_Cancer"])
network.add_edge(states["Smoking"],states["Lung_Cancer"])
network.add_edge(states["Genetics"],states["Attention"])
network.add_edge(states['Lung_Cancer'], states["Coughing"])
network.add_edge(states['Allergy'], states["Coughing"])
network.add_edge(states['Coughing'], states["Fatigue"])
network.add_edge(states['Lung_Cancer'], states["Fatigue"])
network.add_edge(states["Fatigue"], states["Car_Accident"])
network.add_edge(states["Attention"], states["Car_Accident"])
import ast
network.bake()
beliefs = network.predict_proba({"Genetics":"T"},max_iterations=100000)
# print(beliefs)
# beliefs = map(str, beliefs)
for state, belief in zip(network.states, beliefs):
    if hasattr(belief,"parameters"):
        print(state.name,belief.parameters)
# exit()
# network.add_edge(s1, s3)
# network.add_edge(s2, s3)
#
# beliefs = network.predict_proba({"guest": "A", "monty": "B"})
# beliefs = map(str, beliefs)
# for state, belief in zip(network.states, beliefs):
#     print(state.name, belief)
