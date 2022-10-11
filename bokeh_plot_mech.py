import cantera as ct
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
import matplotlib.pyplot as plt


mechanism_file = "/home/moon/autokmc/rmg2kmc/examples/co_oxidation/chem_annotated.cti"
# define starting conditions:
T = 274


gas = ct.Solution(mechanism_file, "gas")
surf = ct.Interface(mechanism_file, "surface1", [gas])

species_list = surf.species() + gas.species()
reaction_list_total = surf.reactions() + gas.reactions()

reaction_weights = [rxn.rate(T) for rxn in reaction_list_total]
# plt.hist(np.log(reaction_weights))
# plt.show()
# exit(0)
# kcut = np.mean(reaction_weights)
kcut = 0
reaction_list_trimmed = []
for i, rxn in enumerate(reaction_list_total):
    if reaction_weights[i] > kcut:
        reaction_list_trimmed.append(rxn)

reaction_list = reaction_list_trimmed
hide_inerts = True
inerts = set(['Ar', 'Ne', 'N2'])
overall_reactants = set(['O2(2)', 'CO(4)'])
overall_products = set(['CO2(3)'])

G = nx.Graph()

for i, species in enumerate(species_list):
    if hide_inerts:
        if species.name in inerts:
            continue

        G.add_node(species.name, name=species.name)
node_color_map = []
for node in G:
    if node in overall_reactants:
        node_color_map.append('red')
    elif node in overall_products:
        node_color_map.append('green')
    else:
        node_color_map.append('grey')


for reaction in reaction_list:
    print(f'{reaction.equation}:\t{reaction.rate(T)}')
    for product_name in reaction.products:
        for reactant_name in reaction.reactants:
            G.add_edge(product_name, reactant_name, color='black')

pos1 = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
nx.draw(G, pos=pos1, node_color=node_color_map, with_labels=True, font_weight='bold')
plt.show()


# for sp in species_list:
#     print(sp.name)
