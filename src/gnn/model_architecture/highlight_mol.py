import os
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import io
import cairosvg
import matplotlib.patches as mpatches
from PIL import Image
import torch

from model_architecture.tree_building import DGLMolTree
from model_architecture.unify_high_mol import unify_fragments

def colors_higlight(attention_scores_normalized, edge_index):
    """  
    Let's assume we already have edge_index, attention_weights and a molecule (mol)
    - edge_index: Tensor of shape (2, E)
    - attention_weights: Tensor of shape (E, H) -> we will use the mean over H 
    """

    # for each element in edge_index, if the value is greater than pred_edge_len, subtract pred_edge_len
    #edge_index = edge_index - pred_edge_len

    # Convert edge_index to a list of tuples (source, destination)
    edges = edge_index.t().tolist()[1::2]  # Get a list of pairs (source, destination)
    
    # Create a color map based on the attention scores
    # Here we use a simple linear mapping of colors (from white to red)
    colors = {}
    for i, (start, end) in enumerate(edges):
        score = attention_scores_normalized[i].item()
        
        # Assign the color based on the score value
        if score < 0.50:
            color = (1.0, 1.0, 1.0)
        elif 0.50 <= score <= 0.60:
            color = (1.0, 1.0, 0.0)  # Yellow
        elif 0.61 <= score <= 0.80:
            color = (1.0, 0.65, 0.0) # Orange
        elif 0.81 <= score <= 1.0:
            color = (1.0, 0.0, 0.0)  # Red
        else:
            color = (0.5, 0.5, 0.5)  # Gray for out-of-range values (optional)

        colors[(start, end)] = color  # Assign the color
    return edges, colors

def draw_molecule_with_attention(mol, edges, colors, attention_scores_normalized):
    drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
    opts = drawer.drawOptions()
    
    num_atoms = mol.GetNumAtoms()  # Get the total number of atoms in the molecule
    
    # Create a list for the indices of the bonds to highlight and a dictionary for the colors
    highlight_bonds = []
    highlight_bond_colors = {}
    bonds = mol.GetBonds()

    for i, (start, end) in enumerate(edges):
        # Ensure that start is always less than or equal to end
        if start > end:
            start, end = end, start
        
        # Check if the indices are valid
        if start >= num_atoms or end >= num_atoms:
            continue
        
        bond = mol.GetBondBetweenAtoms(start, end)
        
        # Check if the bond exists
        if bond is None:
            continue
        
        bond_idx = bond.GetIdx()
        
        # Add the bond to the highlights
        highlight_bonds.append(bond_idx)
        highlight_bond_colors[bond_idx] = tuple(colors.get((start, end), colors.get((end, start))))

    # Draw the molecule with highlighted bonds
    drawer.DrawMolecule(mol, highlightAtoms=[], highlightBonds=highlight_bonds, highlightBondColors=highlight_bond_colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg

def higlight_mol(attention_scores_normalized, edge_index, smiles, id, final=False):
    # Display the image
    mol = Chem.MolFromSmiles(smiles)
    edges, colors = colors_higlight(attention_scores_normalized, edge_index)
    svg = draw_molecule_with_attention(mol, edges, colors, attention_scores_normalized)
    
    # Convert SVG to PNG
    png_data = cairosvg.svg2png(bytestring=svg)
    
    # Create a PIL image from the PNG string
    image = Image.open(io.BytesIO(png_data))
    
    # Create the combined figure with matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(image)
    ax.axis('off')  # Remove the axes

    # Add the legend
    yellow_patch = mpatches.Patch(color=(1.0, 1.0, 0.0), label='0.50 - 0.60')
    orange_patch = mpatches.Patch(color=(1.0, 0.65, 0.0), label='0.61 - 0.80')
    red_patch = mpatches.Patch(color=(1.0, 0.0, 0.0), label='0.81 - 1.00')

    ax.legend(handles=[yellow_patch, orange_patch, red_patch], 
              bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Display the image with the legend
    if final == True:
        os.makedirs(f'../results/test_data/TCMBANK/images/mol_raw', exist_ok=True)
        plt.savefig(f'../results/test_data/TCMBANK/images/mol_raw/mol_{id}_final.png')
    else:
        os.makedirs(f'../results/test_data/images/mol_raw', exist_ok=True)
        plt.savefig(f'../results/test_data/images/mol_raw/mol_{id}.png')
    plt.close()

def map_tree_edges_to_molecule(tree_edge_index, tree_to_molecule_mapping):
    """
    Create a mapping between the tree edges and the molecule edges.
    For each edge in the tree (between cliques), obtain the set of corresponding edges in the molecule.
    """
    molecule_edge_mapping = {}
    
    for i, (start, end) in enumerate(tree_edge_index.t().tolist()):
       
        if len(tree_to_molecule_mapping) == 1:
            try:
                molecule_edge_mapping[(start, end)] = tree_to_molecule_mapping[start]
                break
            except Exception as e:
                print(f"Errore: {e}")
                break
            
        clique_start_bonds = tree_to_molecule_mapping[start]
        clique_end_bonds = tree_to_molecule_mapping[end]
        
        # Find all the possible molecular edges between the nodes of clique_start and clique_end
        # molecule_edges = [(a, b) for a in clique_start for b in clique_end]

        # Concatenate the lists of bonds simply by list concatenation
        molecule_edges = clique_start_bonds + clique_end_bonds
        molecule_edge_mapping[(start, end)] = molecule_edges

    return molecule_edge_mapping

def draw_molecule_with_edge_attention(mol, tree_edge_index, molecule_edge_mapping, tree_edge_attention_scores, alpha=1.0):
    """
    Draw the molecule highlighting the bonds based on the tree attention scores
    """
    drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
    opts = drawer.drawOptions()

    num_atoms = mol.GetNumAtoms()  # Get the total number of atoms in the molecule

    # Create a list for the indices of the bonds to highlight and a dictionary for the colors
    highlight_bonds = []
    highlight_bond_colors = {}

    edge_index = 0

    for (tree_start, tree_end), molecule_edges in molecule_edge_mapping.items():
        score = tree_edge_attention_scores[edge_index]
        
        if score < 0.50:
            color = (1.0, 1.0, 1.0, alpha)
        elif 0.50 <= score <= 0.60:
            color = (1.0, 1.0, 0.0, alpha)  # Yellow
        elif 0.61 <= score <= 0.80:
            color = (1.0, 0.65, 0.0, alpha)  # Orange
        elif 0.81 <= score <= 1.0:
            color = (1.0, 0.0, 0.0, alpha)  # Red
        else:
            color = (0.5, 0.5, 0.5, alpha)  # Gray for out-of-range values

        # Apply the color to each molecular bond associated with the tree edge
        for start, end in molecule_edges:
            if start > end:
                start, end = end, start

            # Check if the indices are valid
            if start >= num_atoms or end >= num_atoms:
                continue

            bond = mol.GetBondBetweenAtoms(start, end)
            assert bond is not None, f"Edge {start}-{end} does not exist in molecule"
            bond_idx = bond.GetIdx()

            highlight_bonds.append(bond_idx)
            highlight_bond_colors[bond_idx] = color
        
        edge_index += 1

    # Draw the molecule with highlighted bonds
    drawer.DrawMolecule(mol, highlightAtoms=[], highlightBonds=highlight_bonds, highlightBondColors=highlight_bond_colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg

def highlight_molecule_with_tree_edge_attention(smiles, tree_edge_index, attention_scores_mean_tree, tree_to_molecule_mapping, id, final=False):
    mol = Chem.MolFromSmiles(smiles)
    
    # Map the tree edges to the corresponding molecule edges
    molecule_edge_mapping = map_tree_edges_to_molecule(tree_edge_index, tree_to_molecule_mapping)

    # Draw the molecule highlighting the edges based on the attention weights on the tree edges
    svg = draw_molecule_with_edge_attention(mol, tree_edge_index, molecule_edge_mapping, attention_scores_mean_tree)

    # Convert SVG to PNG
    png_data = cairosvg.svg2png(bytestring=svg)
    image = Image.open(io.BytesIO(png_data))
    
    # Create the combined figure with matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(image)
    ax.axis('off')  # Remove the axes

    # Add the legend
    yellow_patch = mpatches.Patch(color=(1.0, 1.0, 0.0), label='0.50 - 0.60')
    orange_patch = mpatches.Patch(color=(1.0, 0.65, 0.0), label='0.61 - 0.80')
    red_patch = mpatches.Patch(color=(1.0, 0.0, 0.0), label='0.81 - 1.00')

    ax.legend(handles=[yellow_patch, orange_patch, red_patch], 
              bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Display the image with the legend
    if final == True:
        os.makedirs(f'../results/test_data/TCMBANK/images/tree_decomp', exist_ok=True)
        plt.savefig(f'../results/test_data/TCMBANK/images/tree_decomp/mol_{id}_final.png')
    else:
        os.makedirs(f'../results/test_data/images/tree_decomp', exist_ok=True)
        plt.savefig(f'../results/test_data/images/tree_decomp/mol_{id}.png')

    plt.close()

def highlight_molecule_with_high_fragments(smiles, high_score_fragments, medium_score_fragments, tree_to_molecule_mapping, id, final=False):
    mol = Chem.MolFromSmiles(smiles)

    high_score_atom_bonds = set()

    if len(tree_to_molecule_mapping) > 1:
        for f in high_score_fragments:
            high_score_atom_bonds.update(tree_to_molecule_mapping[f])

        medium_score_fragments = set()
        for f in medium_score_fragments:
            medium_score_fragments.update(tree_to_molecule_mapping[f])
    else:
        high_score_atom_bonds = tree_to_molecule_mapping[0]
        medium_score_fragments = []
        
    if len(high_score_atom_bonds) == 0 and len(medium_score_fragments) == 0:
        return False

    svg = draw_molecule_with_high_fragments(mol, high_score_atom_bonds, medium_score_fragments)

    # Convert SVG to PNG
    png_data = cairosvg.svg2png(bytestring=svg)
    image = Image.open(io.BytesIO(png_data))
    
    # Create the combined figure with matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(image)
    ax.axis('off')  # Remove the axes

    # Add the legend
    yellow_patch = mpatches.Patch(color=(1.0, 1.0, 0.0), label='0.50 - 0.60')
    red_patch = mpatches.Patch(color=(1.0, 0.0, 0.0), label='0.81 - 1.00')

    ax.legend(handles=[yellow_patch, red_patch], 
              bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Display the image with the legend
    if final == True:
        os.makedirs(f'../results/test_data/TCMBANK/images/tree_decomp', exist_ok=True)
        plt.savefig(f'../results/test_data/TCMBANK/images/tree_decomp/mol_{id}_final.png')
    else:
        os.makedirs(f'../results/test_data/images/tree_decomp', exist_ok=True)
        plt.savefig(f'../results/test_data/images/tree_decomp/mol_{id}.png')
    plt.close()
    return True

def colorize_bonds(mol, molecule_edges, num_atoms, highlight_bonds, highlight_bond_colors, color):
        (start, end) = molecule_edges
        if start > end:
            start, end = end, start

        # Check if the indices are valid
        assert not(start >= num_atoms or end >= num_atoms)
            
        
        bond = mol.GetBondBetweenAtoms(start, end)
        assert bond is not None, f"Edge {start}-{end} does not exist in molecule"
        bond_idx = bond.GetIdx()

        highlight_bonds.append(bond_idx)
        highlight_bond_colors[bond_idx] = color
                

def draw_molecule_with_high_fragments(mol, high_score_atom_bonds, medium_score_fragments, alpha=1.0):
    """
    Draw the molecule highlighting the bonds based on the extracted fragments and their score
    """
    drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
    opts = drawer.drawOptions()

    num_atoms = mol.GetNumAtoms()  # Get the total number of atoms in the molecule

    # Create a list for the indices of the bonds to highlight and a dictionary for the colors
    highlight_bonds = []
    highlight_bond_colors = {}

    for edge in high_score_atom_bonds:
        color = (1.0, 0.0, 0.0, alpha)  # Red
        # Apply the color to each molecular bond associated with the fragment
        colorize_bonds(mol, tuple(edge), num_atoms, highlight_bonds=highlight_bonds, highlight_bond_colors=highlight_bond_colors, color=color)
          
    for edge in medium_score_fragments:
        color = (1.0, 1.0, 0.0, alpha)  # Yellow
        colorize_bonds(mol, tuple(edge), num_atoms, highlight_bonds=highlight_bonds, highlight_bond_colors=highlight_bond_colors, color=color)

    # Draw the molecule with highlighted bonds
    drawer.DrawMolecule(mol, highlightAtoms=[], highlightBonds=highlight_bonds, highlightBondColors=highlight_bond_colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg

def highlight_mol_per_graph(raw_edge_index, raw_attention_scores, tree_edge_index, tree_attention_scores, batch, tree_batch, prev_batch_num_graphs, deep_search=False, deep_search_threshold=0.80, final=False):
    # Get the number of graphs in the batch
    node_batch = batch.batch  # batch.batch is a tensor that assigns to each node its graph index
    tree_node_batch = tree_batch.batch
    num_graphs = node_batch.max().item() + 1  # number of graphs present in the batch
    high_score_fragments = []

    for i in range(num_graphs):
        index = prev_batch_num_graphs + i
        smiles = batch.smiles[i]

        # Select the nodes and edges belonging to the i-th graph
        node_mask = (node_batch == i)
        edge_mask = node_mask[raw_edge_index[0]] & node_mask[raw_edge_index[1]]
        if edge_mask.sum() == 0:
            continue

        # Extract the local indices of the nodes and corresponding edge_index in the batch
        local_edge_index = raw_edge_index[:, edge_mask] - node_mask.nonzero(as_tuple=True)[0].min()
        local_attention_scores = raw_attention_scores[edge_mask]

        # Select the nodes and edges belonging to the i-th tree 
        tree_node_mask = (tree_node_batch == i)
        tree_edge_mask = tree_node_mask[tree_edge_index[0]] & tree_node_mask[tree_edge_index[1]]
        if tree_edge_mask.sum() == 0:
            continue

        # Extract the local indices of the nodes and corresponding edge_index in tree_batch
        tree_local_edge_index = tree_edge_index[:, tree_edge_mask] - tree_node_mask.nonzero(as_tuple=True)[0].min()
        tree_local_attention_scores = tree_attention_scores[tree_edge_mask]

        # Map the tree nodes to the molecule nodes
        tree_to_molecule_mapping = {}
        nodes_dict = tree_batch.mol_tree[i].nodes_dict
        tree_edges = tree_batch.mol_tree[i].all_edges()

        for tree_node_idx, node_data in nodes_dict.items():
            raw_bonds = node_data['bonds']  # Get the list of atom indices associated with this node
            tree_to_molecule_mapping[tree_node_idx] = raw_bonds

        ###### Capture high-relevance fragments ######
        # Initialize variables for deep search
        high_score_fragments_for_graph = None
        medium_score_fragments_for_graph = None
        visited_nodes = set()

        # if deep_search == True search all the cliques connected through tree_local_attention_scores[j] > 0.79
        if deep_search == True:
            high_score_fragments_for_graph = set()
            medium_score_fragments_for_graph = set()
            # take the edges with attention greater than 0.79
            for j in range(len(tree_local_attention_scores)):
                if tree_local_attention_scores[j] > deep_search_threshold:
                    start = tree_local_edge_index[0][j].item()
                    end = tree_local_edge_index[1][j].item()
                    if start == end:
                        continue
                    high_score_fragments_for_graph.add(start)
                    high_score_fragments_for_graph.add(end)

                if tree_local_attention_scores[j] <= deep_search_threshold and tree_local_attention_scores[j] >= 0.50:
                    start = tree_local_edge_index[0][j].item()
                    end = tree_local_edge_index[1][j].item()
                    if start == end:
                        continue
                    medium_score_fragments_for_graph.add(start)
                    medium_score_fragments_for_graph.add(end)
        ###### End capture of high-relevance fragments ######
       
       ######### PART TO EXTRACT FRAGMENTS WITH ATTENTION GREATER THAN 0.79 ########## 
       # fill high_score_fragments with the edges that have attention greater than 0.79
        for j in range(len(tree_local_attention_scores)):
            if tree_local_attention_scores[j] > deep_search_threshold:

                if len(tree_edges[0]) == 0:  # if there are no edges, there are no fragments, but the entire molecule is a fragment
                    high_score_fragments.append((smiles, 0))
                    continue

                # take the node pairs associated with this edge
                start = tree_local_edge_index[0][j].item()
                end = tree_local_edge_index[1][j].item()

                if start == end:  # if start == end, there are no fragments
                    continue
                

                # extract number of fragments with which each fragment is connected
                # count the number of occurrences of start in bonds
                first_elements = tree_edges[0]
                start_count = torch.sum(first_elements == start).item() - 1  # -1 because to consider the link between start and end
                end_count = torch.sum(first_elements == end).item() - 1 
                # assertion that start_bonds or end_bonds are 0 is possible only when tree_edges == 1, that is between start and end
                assert start_count >= 0 and end_count >= 0, "start_bonds e end_bonds non possono essere negativi"
                assert (start_count != 0 or end_count != 0) or int(len(tree_edges[0]) / 2) == 1, "almeno uno tra start_bonds e end_count non può essere 0 se ci sono più di un edge"
                
                try:
                    if deep_search == True:
                        if start not in visited_nodes and end not in visited_nodes:
                            final_mol, connected_high_score_frags = unify_fragments(tree_batch.mol_tree[i], [start, end], high_score_fragments_for_graph, deep_search=deep_search)
                            visited_nodes.update(connected_high_score_frags)
                        else:
                            continue
                    else:
                        final_mol, _ = unify_fragments(tree_batch.mol_tree[i], [start, end], high_score_fragments_for_graph, deep_search=deep_search)

                except Exception:
                    # throw exception if it is not possible to merge the two fragments
                    print(f"Errore durante l'unione dei frammenti {start} e {end} di smiles {smiles}")
                    exit(-1)

                # Sanitize molecule before converting it to SMILES
                assert final_mol is not None, "Problema con estrazione frammenti"
                try:
                    Chem.SanitizeMol(final_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                except Exception as e:
                    print(f"Errore di sanitizzazione: {e}")
                    continue
                final_smiles = Chem.MolToSmiles(final_mol)

                # count number of fragments with which each fragment is connected
                neig_count = 0
                if deep_search == False:
                    neig_count = start_count + end_count
                else:
                    for frag in connected_high_score_frags:
                        neig_count += torch.sum(first_elements == frag).item() - 1
                    
                final_smiles_num_neig = (final_smiles, neig_count, index)
                high_score_fragments.append(final_smiles_num_neig)

        ######### END PART FOR EXTRACTING FRAGMENTS WITH ATTENTION GREATER THAN 0.79 ##########
        # Display the molecule associated with this graph
        higlight_mol(local_attention_scores, local_edge_index, smiles, index, final)
        # the third condition is to handle those cases with peculiar molecules that are decomposed incorrectly
        if deep_search == False or len(high_score_fragments) == 0:
            highlight_molecule_with_tree_edge_attention(smiles, tree_local_edge_index, tree_local_attention_scores, tree_to_molecule_mapping, index, final)
        # modified version of highlight_molecule_with_tree_edge_attention that uses high_score_fragments_for_graph
        else:
            # I added a check to see if the function returns False, meaning that no edge was captured, in which case I call the function to highlight the tree edges
            res = highlight_molecule_with_high_fragments(smiles, high_score_fragments_for_graph, medium_score_fragments_for_graph, tree_to_molecule_mapping, index, final)
    
    return batch.num_graphs, high_score_fragments
