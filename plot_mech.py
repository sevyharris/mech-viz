import cantera as ct
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt


mechanism_file = "/home/moon/autokmc/rmg2kmc/examples/co_oxidation/chem_annotated.cti"

gas = ct.Solution(mechanism_file, "gas")
surf = ct.Interface(mechanism_file, "surface1", [gas])

species_list = surf.species() + gas.species()
reaction_list = surf.reactions() + gas.reactions()


hide_inerts = True
inerts = set(['Ar', 'Ne', 'N2'])

G = nx.Graph()
for i, species in enumerate(species_list):
    if hide_inerts:
        if species.name in inerts:
            continue

    # color based on surface/gas or other property
    G.add_node(species.name, color='red', name=species.name, label=species.name)


for reaction in reaction_list:
    for product_name in reaction.products:
        # prod_indices = [species_list.index(sp) for sp in species_list if sp.name == product_name]
        # product_index = prod_indices[0]
        for reactant_name in reaction.reactants:
            # reactant_indices = [species_list.index(sp) for sp in species_list if sp.name == reactant_name]
            # reactant_index = reactant_indices[0]
            # G.add_edge(reactant_index, product_index, color='black')
            G.add_edge(product_name, reactant_name, color='black')

my_pos = nx.spring_layout(G, seed=400)
nx.draw(G, pos=my_pos, with_labels=True, font_weight='bold')
plt.show()

for sp in species_list:
    print(sp.name)
