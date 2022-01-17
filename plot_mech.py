# Use existing RMG tools and graphviz to plot the flux diagram

from rmgpy.tools import fluxdiagram

spec_dict = "/home/moon/rmg/my_examples/min_example/chemkin/chem_annotated.inp"
chemkin = "/home/moon/rmg/my_examples/min_example/chemkin/species_dictionary.txt"
input_file = "/home/moon/rmg/my_examples/min_example/input.py"
output_path = "/home/moon/rmg/mech_viz/flux_diagram/"
fluxdiagram.create_flux_diagram(input_file, chemkin, spec_dict, save_path=output_path)



