from collections import deque
from rdkit import Chem
from rdkit.Chem import Draw
try:
    from model_architecture.tree_building import DGLMolTree
except:
    from tree_building import DGLMolTree

def bfs_connected_high_fragments(tree, start_nodes, high_fragment):
    """
    Searches for all nodes connected from the starting nodes that are part of the high_fragment list.
    
    Args:
    - tree: DGLMolTree object representing the molecular tree with an 'edges' attribute containing the edges.
    - start_nodes: List of starting nodes from which to begin the search.
    - high_fragment: List of fragments (nodes) that are of high interest.
    
    Returns:
    - connected_fragments: Set of nodes that are in the high_fragment list and connected to the starting nodes.
    """
    # Set to track visited nodes
    visited = set()

    # Set to contain the connected high-interest fragments found
    connected_fragments = set()

    # Queue for the BFS
    queue = deque(start_nodes)

    # Build a neighbor map from tree.edges
    neighbors_map = {}
    edges = tree.mol_edge_map.keys()
    for (start, end) in edges:
        if start not in neighbors_map:
            neighbors_map[start] = []
        if end not in neighbors_map:
            neighbors_map[end] = []
        neighbors_map[start].append(end)
        neighbors_map[end].append(start)

    # Start the BFS
    while queue:
        node = queue.popleft()
        
        # If we have already visited this node, skip it
        if node in visited:
            continue

        # Mark the node as visited
        visited.add(node)

        # If the node is in the high_fragment list, add it
        if node in high_fragment:
            connected_fragments.add(node)

        # Expand the search to all neighbors of the current node
        neighbors = neighbors_map.get(node, [])  # Get the neighbors of the node
        for neighbor in neighbors:
            # Add the neighbor to the queue only if it has not been visited and is in the high_fragment list
            if neighbor not in visited and neighbor in high_fragment:
                queue.append(neighbor)

    return connected_fragments


def extract_substructure_preserving_rings(mol, atom_indices):
    """
    Extracts a substructure from a molecule while preserving rings, stereochemistry, and other atomic properties.
    
    Args:
    - mol: RDKit molecule from which to extract the substructure.
    - atom_indices: List of atomic indices to include in the substructure.
    
    Returns:
    - submol: Substructure as an RDKit object.
    """
    # Create a new RWMol object to allow modifications
    rw_mol = Chem.RWMol()

    # Mapping between atomic indices of the original molecule and those of the new submol
    atom_map = {}

    # Add the specified atoms, preserving properties such as stereochemistry and charge
    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetChiralTag(atom.GetChiralTag())  # Preserve the chiral configuration
        new_atom.SetHybridization(atom.GetHybridization())  # Preserve the hybridization
        new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())  # Preserve the number of explicit hydrogens
        new_idx = rw_mol.AddAtom(new_atom)
        atom_map[idx] = new_idx

    # Add bonds between the specified atoms, preserving properties such as order and stereochemistry
    for bond in mol.GetBonds():
        start_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        
        # Add the bond only if both atoms are in the specified list
        if start_atom in atom_map and end_atom in atom_map:
            new_bond_type = bond.GetBondType()
            rw_mol.AddBond(atom_map[start_atom], atom_map[end_atom], new_bond_type)

            # Preserve the bond stereochemistry (e.g., E/Z bonds)
            stereo = bond.GetStereo()
            if stereo != Chem.BondStereo.STEREONONE:
                rw_mol.GetBondBetweenAtoms(atom_map[start_atom], atom_map[end_atom]).SetStereo(stereo)

    # Sanitize the molecule to verify its validity
    try:
        Chem.SanitizeMol(rw_mol)
    except ValueError:
        return rw_mol

    # Remove explicit hydrogens if present
    submol = Chem.RemoveHs(rw_mol)
    
    return submol


def unify_fragments(tree, fragments, high_fragments=None, deep_search=False):
    """
    Combines highly relevant fragments from the tree structure of the molecule and returns a substructure.
    
    Args:
    - tree: Tree structure of the molecule.
    - fragments: List of fragment indices.
    - high_fragments: List of high-interest fragments (nodes). Must be specified if deep_search is True.
    - deep_search: Boolean flag indicating whether to perform a deep search.
    
    Returns:
    - submol: Combined molecule resulting from the joined fragments.
    - connected_fragments: Set of connected fragments found (if any).
    """
    tree_nodes = tree.nodes_dict
    connected_fragments = None

    # Start the BFS from the two nodes
    assert high_fragments is not None or deep_search == False, "If performing a deep search, the high-interest fragments must be specified"
       
    if deep_search == True and high_fragments is not None:
        connected_fragments = bfs_connected_high_fragments(tree, fragments, high_fragments)
        fragments = connected_fragments

    id_mol = set()
    for i in fragments:
        clique = tree_nodes[i]['clique']
        id_mol.update(clique)

    submol = extract_substructure_preserving_rings(tree.mol, id_mol)

    return submol, connected_fragments


# Example of use:
# if __name__ == "__main__":
#
#     smiles = 'NC(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)O'
#     original_mol = Chem.MolFromSmiles(smiles)
#     # Draw.MolToImage(original_mol).show()
#     tree = DGLMolTree(smiles)
#
#     high_frags = [10, 12, 3, 11]
#     frags = [5, 10]
#
#     (clique1, clique2) = tuple(frags)
#     assert tree.has_edges_between(clique1, clique2), "The fragments must be nodes in the tree"
#
#     submol, connected_frags = unify_fragments(tree, frags, high_frags, deep_search=True)
#
#     print(Chem.MolToSmiles(submol))
#     Draw.MolToImage(submol).show()
