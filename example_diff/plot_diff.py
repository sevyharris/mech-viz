import os
import re
import numpy as np
import pydot
import rmgpy.tools.fluxdiagram
from rmgpy.solver.base import TerminationTime
# from rmgpy.solver.base import TerminationTime, TerminationConversion
from rmgpy.solver.liquid import LiquidReactor
from rmgpy.kinetics.diffusionLimited import diffusion_limiter


# Options controlling the individual flux diagram renderings:
program = 'dot'  # The program to use to lay out the nodes and edges
max_node_count = 50  # The maximum number of nodes to show in the diagram
max_edge_count = 50  # The maximum number of edges to show in the diagram
concentration_tol = 1e-6  # The lowest fractional concentration to show (values below this will appear as zero)
species_rate_tol = 1e-6  # The lowest fractional species rate to show (values below this will appear as zero)
max_node_pen_width = 10.0  # The thickness of the border around a node at maximum concentration
max_edge_pen_width = 10.0  # The thickness of the edge at maximum species rate
radius = 1  # The graph radius to plot around a central species
central_reaction_count = None  # The maximum number of reactions to draw from each central species (None draws all)
# If radius > 1, then this is the number of reactions from every species

# Options controlling the ODE simulations:
initial_time = 1e-12  # The time at which to initiate the simulation, in seconds
time_step = 10 ** 0.1  # The multiplicative factor to use between consecutive time points
abs_tol = 1e-16  # The absolute tolerance to use in the ODE simluations
rel_tol = 1e-8  # The relative tolerance to use in the ODE simulations

# Options controlling the generated movie:
video_fps = 6  # The number of frames per second in the generated movie
initial_padding = 5  # The number of seconds to display the initial fluxes at the start of the video
final_padding = 5  # The number of seconds to display the final fluxes at the end of the video


def generate_flux_diff_diagram(
    reaction_model,
    times,
    concentrations,
    reaction_rates,
    output_directory,
    central_species_list=None,
    superimpose=False,
    species_directory=None,
    settings=None
):
    """
    For a given `reaction_model` and simulation results stored as arrays of
    `times`, species `concentrations`, and `reaction_rates`, generate a series
    of flux diagrams as frames of an animation, then stitch them together into
    a movie. The individual frames and the final movie are saved on disk at
    `output_directory.`
    """
    global max_node_count, max_edge_count, concentration_tol, species_rate_tol, max_node_pen_width, max_edge_pen_width, radius, central_reaction_count
    # Allow user defined settings for flux diagram generation if given
    if settings:
        max_node_count = settings.get('max_node_count', max_node_count)
        max_edge_count = settings.get('max_edge_count', max_edge_count)
        concentration_tol = settings.get('concentration_tol', concentration_tol)
        species_rate_tol = settings.get('species_rate_tol', species_rate_tol)
        max_node_pen_width = settings.get('max_node_pen_width', max_node_pen_width)
        max_edge_pen_width = settings.get('max_edge_pen_width', max_edge_pen_width)
        radius = settings.get('radius', radius)
        central_reaction_count = settings.get('central_reaction_count', central_reaction_count)

    # Get the species and reactions corresponding to the provided concentrations and reaction rates
    species_list = reaction_model.core.species[:]
    num_species = len(species_list)
    reaction_list = reaction_model.core.reactions[:]

    # Search for indices of central species
    central_species_indices = []
    if central_species_list is not None:
        for centralSpecies in central_species_list:
            for i, species in enumerate(species_list):
                if species.index == centralSpecies:
                    central_species_indices.append(i)
                    break
            else:
                raise Exception("Central species '{}' could not be found in species list.".format(centralSpecies))

    # Compute the rates between each pair of species (big matrix warning!)
    species_rates = np.zeros((len(times), num_species, num_species), np.float64)
    for index, reaction in enumerate(reaction_list):
        rate = reaction_rates[:, index]
        if not reaction.pairs: reaction.generate_pairs()
        for reactant, product in reaction.pairs:
            reactant_index = species_list.index(reactant)
            product_index = species_list.index(product)
            species_rates[:, reactant_index, product_index] += rate
            species_rates[:, product_index, reactant_index] -= rate

    # Determine the maximum concentration for each species and the maximum overall concentration
    max_concentrations = np.max(np.abs(concentrations), axis=0)
    max_concentration = np.max(max_concentrations)

    # Determine the maximum reaction rates
    max_reaction_rates = np.max(np.abs(reaction_rates), axis=0)

    # Determine the maximum rate for each species-species pair and the maximum overall species-species rate
    max_species_rates = np.max(np.abs(species_rates), axis=0)
    max_species_rate = np.max(max_species_rates)
    species_index = max_species_rates.reshape((num_species * num_species)).argsort()

    # Determine the nodes and edges to keep
    nodes = []
    edges = []
    if not superimpose and central_species_list is not None:
        for central_species_index in central_species_indices:
            nodes.append(central_species_index)
            rmgpy.tools.fluxdiagram.add_adjacent_nodes(
                central_species_index,
                nodes,
                edges,
                species_list,
                reaction_list,
                max_reaction_rates,
                max_species_rates,
                reactionCount=central_reaction_count,
                rad=radius
            )
    else:
        for i in range(num_species * num_species):
            product_index, reactant_index = divmod(species_index[-i - 1], num_species)
            if reactant_index > product_index:
                # Both reactant -> product and product -> reactant are in this list,
                # so only keep one of them
                continue
            if max_species_rates[reactant_index, product_index] == 0:
                break
            if reactant_index not in nodes and len(nodes) < max_node_count: nodes.append(reactant_index)
            if product_index not in nodes and len(nodes) < max_node_count: nodes.append(product_index)
            if [reactant_index, product_index] not in edges and [product_index, reactant_index] not in edges:
                edges.append([reactant_index, product_index])
            if len(nodes) > max_node_count:
                break
            if len(edges) >= max_edge_count:
                break

        if superimpose and central_species_list is not None:
            nodes_copy = nodes[:]
            for central_species_index in central_species_indices:
                if central_species_index not in nodes:  # Only add central species if it doesn't already exist
                    nodes.append(central_species_index)
                    # Recursively add nodes until they connect with main graph
                    rmgpy.tools.fluxdiagram.add_adjacent_nodes(
                        central_species_index,
                        nodes,
                        edges,
                        species_list,
                        reaction_list,
                        max_reaction_rates,
                        max_species_rates,
                        reactionCount=central_reaction_count,
                        rad=-1,  # "-1" signifies that we add nodes until they connect to the main graph
                        mainNodes=nodes_copy
                    )

    # Create the master graph
    # First we're going to generate the coordinates for all of the nodes; for
    # this we use the thickest pen widths for all nodes and edges 
    graph = pydot.Dot('flux_diagram', graph_type='digraph', overlap="false")
    graph.set_rankdir('LR')
    graph.set_fontname('sans')
    graph.set_fontsize('10')

    # Add a node for each species
    for index in nodes:
        species = species_list[index]
        node = pydot.Node(name=str(species))
        node.set_penwidth(max_node_pen_width)
        graph.add_node(node)
        # Try to use an image instead of the label
        species_index = str(species) + '.png'
        image_path = ''
        if not species_directory or not os.path.exists(species_directory):
            continue
        for root, dirs, files in os.walk(species_directory):
            for f in files:
                if f.endswith(species_index):
                    image_path = os.path.join(root, f)
                    break
        if os.path.exists(image_path):
            node.set_image(image_path)
            node.set_label(" ")
    # Add an edge for each species-species rate
    for reactant_index, product_index in edges:
        if reactant_index in nodes and product_index in nodes:
            reactant = species_list[reactant_index]
            product = species_list[product_index]
            edge = pydot.Edge(str(reactant), str(product))
            edge.set_penwidth(max_edge_pen_width)
            graph.add_edge(edge)

    # Generate the coordinates for all of the nodes using the specified program
    graph = pydot.graph_from_dot_data(graph.create_dot(prog=program).decode('utf-8'))[0]

    # Now iterate over the time points, setting the pen widths appropriately
    # This should preserve the coordinates of the nodes from frame to frame
    frame_number = 1
    for t in range(len(times)):
        # Update the nodes
        slope = -max_node_pen_width / np.log10(concentration_tol)
        for index in nodes:
            species = species_list[index]
            if re.search(r'^[a-zA-Z0-9_]*$', str(species)) is not None:
                species_string = str(species)
            else:
                # species name contains special characters                
                species_string = '"{0}"'.format(str(species))

            node = graph.get_node(species_string)[0]
            concentration = concentrations[t, index] / max_concentration
            if concentration < concentration_tol:
                penwidth = 0.0
            else:
                penwidth = round(slope * np.log10(concentration) + max_node_pen_width, 3)
            node.set_penwidth(penwidth)
        # Update the edges
        slope = -max_edge_pen_width / np.log10(species_rate_tol)
        for index in range(len(edges)):
            reactant_index, product_index = edges[index]
            if reactant_index in nodes and product_index in nodes:
                reactant = species_list[reactant_index]
                product = species_list[product_index]

                if re.search(r'^[a-zA-Z0-9_]*$', str(reactant)) is not None:
                    reactant_string = str(reactant)
                else:
                    reactant_string = '"{0}"'.format(str(reactant))

                if re.search(r'^[a-zA-Z0-9_]*$', str(product)) is not None:
                    product_string = str(product)
                else:
                    product_string = '"{0}"'.format(str(product))

                edge = graph.get_edge(reactant_string, product_string)[0]
                # Determine direction of arrow based on sign of rate
                species_rate = species_rates[t, reactant_index, product_index] / max_species_rate
                if species_rate < 0:
                    edge.set_dir("back")
                    species_rate = -species_rate
                else:
                    edge.set_dir("forward")
                # Set the edge pen width
                if species_rate < species_rate_tol:
                    penwidth = 0.0
                    edge.set_dir("none")
                else:
                    penwidth = round(slope * np.log10(species_rate) + max_edge_pen_width, 3)
                edge.set_penwidth(penwidth)
        # Save the graph at this time to a dot file and a PNG image
        if times[t] == 0:
            label = 't = 0 s'
        else:
            label = 't = 10^{0:.1f} s'.format(np.log10(times[t]))
        graph.set_label(label)
        if t == 0:
            repeat = video_fps * initial_padding
        elif t == len(times) - 1:
            repeat = video_fps * final_padding
        else:
            repeat = 1
        for r in range(repeat):
            graph.write_dot(os.path.join(output_directory, 'flux_diagram_{0:04d}.dot'.format(frame_number)))
            graph.write_png(os.path.join(output_directory, 'flux_diagram_{0:04d}.png'.format(frame_number)))
            frame_number += 1

    # Use ffmpeg to stitch the PNG images together into a movie
    import subprocess

    command = ['ffmpeg',
               '-framerate', '{0:d}'.format(video_fps),  # Duration of each image
               '-i', 'flux_diagram_%04d.png',  # Input file format
               '-c:v', 'mpeg4',  # Encoder
               '-r', '30',  # Video framerate
               '-pix_fmt', 'yuv420p',  # Pixel format
               'flux_diagram.avi']  # Output filename

    subprocess.check_call(command, cwd=output_directory)



# define inputs
rmg_input_file = '/home/moon/rmg/mech_viz/example_diff/input.py'  # for conditions

mech_1_inp = '/home/moon/rmg/mech_viz/example_diff/chem_annotated.inp'
mech_1_dict = '/home/moon/rmg/mech_viz/example_diff/species_dictionary.txt'

mech_2_inp = '/home/moon/rmg/mech_viz/example_diff/sp85_with_rotors.inp'
mech_2_dict = '/home/moon/rmg/mech_viz/example_diff/species_dictionary.txt'


output_path = os.path.dirname(mech_1_inp)
save_path = output_path
# rmgpy.tools.fluxdiagram.create_flux_diagram(rmg_input_file, mech_1_inp, mech_1_dict, save_path=output_path)

# other settings I should probably delete
# save_path = None        # should be output_path
species_path = None
java = False            # always False
settings = None
chemkin_output = ''     # this will be generated automatically
central_species_list = None
superimpose = False
save_states = False
read_states = False     # fine to keep this always false and delete relevant code below
diffusion_limited = True
check_duplicates = True

if species_path is None:
    species_path = os.path.join(os.path.dirname(rmg_input_file), 'species')
    generate_images = True
else:
    generate_images = False


print('Loading RMG job...')
rmg_job1 = rmgpy.tools.fluxdiagram.load_rmg_job(
    rmg_input_file,
    mech_1_inp,
    mech_1_dict,
    generate_images=generate_images,
    check_duplicates=check_duplicates
)

print('Extracting species concentrations and calculating reaction rates from chemkin output...')
# Generate a flux diagram video for each reaction system
for index, reaction_system in enumerate(rmg_job1.reaction_systems):
    out_dir = os.path.join(save_path, '{0:d}'.format(index + 1))
    try:
        os.makedirs(out_dir)
    except OSError:
        # Fail silently on any OS errors
        pass

    # If there is no termination time, then add one to prevent jobs from
    # running forever
    if not any([isinstance(term, TerminationTime) for term in reaction_system.termination]):
        reaction_system.termination.append(TerminationTime((1e10, 's')))

    states_file = os.path.join(out_dir, 'states.npz')
    if read_states:
        print('Reading simulation states from file...')
        states = np.load(states_file)
        time = states['time']
        core_species_concentrations = states['core_species_concentrations']
        core_reaction_rates = states['core_reaction_rates']
    else:
        # Enable diffusion-limited rates
        if diffusion_limited and isinstance(reaction_system, LiquidReactor):
            rmg_job1.load_database()
            solvent_data = rmg_job1.database.solvation.get_solvent_data(rmg_job1.solvent)
            diffusion_limiter.enable(solvent_data, rmg_job1.database.solvation)

        print('Conducting simulation of reaction system {0:d}...'.format(index + 1))
        time, core_species_concentrations, core_reaction_rates = rmgpy.tools.fluxdiagram.simulate(
            rmg_job1.reaction_model,
            reaction_system,
            settings
        )

        if save_states:
            np.savez_compressed(
                states_file,
                time=time,
                core_species_concentrations=core_species_concentrations,
                core_reaction_rates=core_reaction_rates
            )

    print('Generating flux diagram for reaction system {0:d}...'.format(index + 1))
    generate_flux_diff_diagram(
        rmg_job1.reaction_model,
        time,
        core_species_concentrations,
        core_reaction_rates,
        out_dir,
        central_species_list=central_species_list,
        superimpose=superimpose,
        species_directory=species_path,
        settings=settings
    )
