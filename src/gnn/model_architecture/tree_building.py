from copy import deepcopy
from matplotlib import pyplot as plt
from rdkit import Chem
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np
from rdkit.Chem import Draw
import networkx as nx
from collections import defaultdict
from dgl import DGLGraph, node_subgraph
import torch


# Auxiliary functions (those that you provided)
def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol

def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

def sanitize(mol): 
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol

def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def copy_edit_mol(mol):
    new_mol = Chem.RWMol()
    atom_map = {}
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        idx = new_mol.AddAtom(new_atom)
        atom_map[atom.GetIdx()] = idx
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(atom_map[a1], atom_map[a2], bt)
        # Preserve the bond stereochemistry
        new_bond = new_mol.GetBondBetweenAtoms(atom_map[a1], atom_map[a2])
        new_bond.SetStereo(bond.GetStereo())
    return new_mol

def get_clique_mol(mol, atoms):
    try:
        smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    except:
        try:  # Add hydrogen atoms
            mol_with_H = Chem.AddHs(mol)
            smiles = Chem.MolFragmentToSmiles(mol_with_H, atoms, kekuleSmiles=True)
        except:
            try:    
                smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=False)
            except Exception as e:
                print('in get_clique_mol problmema con clique: {atoms} nonostante i tentativi, restituito None')
                return None
    
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()

    try:
        mol2 = sanitize(new_mol)
    except:
        return new_mol
    if mol2 is None:
        return new_mol
    return mol2

def extract_substructure_preserving_rings(mol, atom_indices):
    """
    Extracts a substructure from a molecule while preserving rings, stereochemistry and other atomic properties.
    
    Args:
    - mol: RDKit molecule from which to extract the substructure.
    - atom_indices: List of atomic indices to include in the substructure.
    
    Returns:
    - submol: The substructure as an RDKit object.
    """
    # Create a new RWMol object to allow modifications
    rw_mol = Chem.RWMol()

    # Mapping between atomic indices of the original molecule and those of the new submol
    atom_map = {}

    # Add the specified atoms, preserving properties such as stereochemistry and charge
    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
        new_atom.SetChiralTag(atom.GetChiralTag())
        new_atom.SetHybridization(atom.GetHybridization())
        new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
        new_atom.SetNoImplicit(True)  # Disable the addition of implicit hydrogens
        new_atom.SetIsAromatic(atom.GetIsAromatic())
        new_atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons())
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
        print("Errore: la sottostruttura non è valida per SanitizeMol.")
        return rw_mol

    # Remove explicit hydrogens if present
    submol = Chem.RemoveHs(rw_mol)
    
    return submol


# tree_decomp_with_mol_edges is basically tree_decomp with the difference that it also returns a dictionary mapping the MST graph edges with the molecule edges
from rdkit import Chem
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

def tree_decomp_with_mol_edges(mol):
    MST_MAX_WEIGHT = 100
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], [], {}

    cliques = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            cliques.append([a1, a2])

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques.extend(ssr)

    nei_list = [[] for _ in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    edges = defaultdict(int)
    mol_edge_map = {}  # Map to store the association between tree_edge and mol_edge

    for atom in range(n_atoms):
        if len(nei_list[atom]) <= 1:
            continue
        cnei = nei_list[atom]
        bonds = [c for c in cnei if len(cliques[c]) == 2]
        rings = [c for c in cnei if len(cliques[c]) > 4]
        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2):
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1, c2)] = 1
                mol_edge_map[(c1, c2)] = (atom, atom)
        elif len(rings) > 2:
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1, c2)] = MST_MAX_WEIGHT - 1
                mol_edge_map[(c1, c2)] = (atom, atom)
        else:
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1, c2 = cnei[i], cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1, c2)] < len(inter):
                        edges[(c1, c2)] = len(inter)
                        # Check if the edge between the cliques corresponds to a molecular edge
                        mol_edges_between_cliques = [
                            (a1, a2) for a1 in cliques[c1] for a2 in cliques[c2] 
                            if mol.GetBondBetweenAtoms(a1, a2) is not None
                        ]
                        if mol_edges_between_cliques:
                            mol_edge_map[(c1, c2)] = mol_edges_between_cliques[0]  # Associate the first found edge

    # Check if the number of cliques is less than 2
    if len(cliques) < 2:
        # Split the existing clique into two sub-cliques
        clique_to_split = cliques[0]
        if len(clique_to_split) >= 2:
            # Divide the clique in half
            mid = len(clique_to_split) // 2
            clique1 = clique_to_split[:mid]
            clique2 = clique_to_split[mid:]
            cliques[0] = clique1
            cliques.append(clique2)
            c1, c2 = 0, len(cliques) - 1
            # Add an edge between the new cliques
            edges[(c1, c2)] = MST_MAX_WEIGHT - 1
            # Find the bonds between the two new cliques
            mol_edges_between_cliques = [
                (a1, a2) for a1 in clique1 for a2 in clique2
                if mol.GetBondBetweenAtoms(a1, a2) is not None
            ]
            if mol_edges_between_cliques:
                mol_edge_map[(c1, c2)] = mol_edges_between_cliques[0]
                mol_edge_map[(c2, c1)] = (mol_edges_between_cliques[0][1], mol_edges_between_cliques[0][0]) 

    edges = [u + (MST_MAX_WEIGHT - v,) for u, v in edges.items()]
    if len(edges) == 0:
        return cliques, edges, mol_edge_map

    row, col, data = list(zip(*edges))
    n_clique = len(cliques)
    clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
    junc_tree = minimum_spanning_tree(clique_graph)
    row, col = junc_tree.nonzero()
    edges = [(row[i], col[i]) for i in range(len(row))]

    # Filter mol_edge_map to keep only the edges that are actually present in the final tree
    mol_edge_map_filtered = {}
    for r, c in edges:
        if (r, c) in mol_edge_map:
            mol_edge_map_filtered[(r, c)] = mol_edge_map[(r, c)]
            mol_edge_map_filtered[(c, r)] = (mol_edge_map[(r, c)][1], mol_edge_map[(r, c)][0])  # Add the reverse edge

    return cliques, edges, mol_edge_map_filtered

def plot_mst_graph(mol, cliques, mst_edges):
    G = nx.Graph()
    
    # Add nodes to the graph with SMILES labels
    for i, clique in enumerate(cliques):
        fragment_mol = get_clique_mol(mol, clique)
        smiles = get_smiles(fragment_mol)
        G.add_node(i, label=smiles)
    
    # Add edges to the graph
    for edge in mst_edges:
        G.add_edge(edge[0], edge[1])
    
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, labels=labels, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold')
    plt.title("Minimum Spanning Tree of Atom Cliques (SMILES)")
    plt.show()

class DGLMolTree(DGLGraph):
    def __init__(self, smiles):
        DGLGraph.__init__(self)
        self.nodes_dict = {}
        self.warning = False
        self.mol_edge_map = None
        self.edge_mol_attr = None

        if smiles is None:
            return

        self.smiles = smiles
        self.mol = get_mol(smiles)

        # cliques: a list of list of atom indices
        # edges: a list of list of edge by atoms src and dst
        # cliques, edges = tree_decomp(self.mol)
        cliques, edges, self.mol_edge_map = tree_decomp_with_mol_edges(self.mol)

        root = 0
        for i, c in enumerate(cliques):
            cmol = extract_substructure_preserving_rings(self.mol, c)
            if cmol is None:
                self.warning = True
                return
            try:
                csmiles = get_smiles(cmol)
            except:
                csmiles = Chem.MolToSmiles(cmol, kekuleSmiles=False, allHsExplicit=True)

            # Find the connections between atoms in the clique
            local_to_global = {local_idx: global_idx for local_idx, global_idx in enumerate(c)}
            bonds = []
            for bond in cmol.GetBonds():
                a1 = bond.GetBeginAtomIdx()
                a2 = bond.GetEndAtomIdx()
                global_a1 = local_to_global[a1]
                global_a2 = local_to_global[a2]
                bonds.append((global_a1, global_a2))

            # Update the nodes dictionary
            self.nodes_dict[i] = dict(
                smiles=csmiles,
                mol=copy_edit_mol(cmol),
                clique=c,
                bonds=bonds  # Add the connections between atoms in the clique
            )
            if min(c) == 0:  # if the clique contains the atom with index 0, then it is the root atom
                root = i

        self.add_nodes(len(cliques))  # remember that DGLMolTree is a subclass of DGLGraph so it has the add_nodes method

        # The clique with atom ID 0 becomes root
        if root > 0:
            for attr in self.nodes_dict[0]:
                self.nodes_dict[0][attr], self.nodes_dict[root][attr] = self.nodes_dict[root][attr], self.nodes_dict[0][attr]  # swap the root atom with the atom with index 0

        # Add edges following the breadth-first order in clique tree decomposition (edges are bi-directional so we add both directions and to do this we add the edges twice so here is explained why we have 2 * len(edges) edges)
        src = np.zeros((len(edges) * 2,), dtype='int')
        dst = np.zeros((len(edges) * 2,), dtype='int')
        for i, (_x, _y) in enumerate(edges):
            x = 0 if _x == root else root if _x == 0 else _x
            y = 0 if _y == root else root if _y == 0 else _y
            src[2 * i] = x  # 2 is caused by the fact that we add the edges twice because they are bi-directional 
            dst[2 * i] = y
            src[2 * i + 1] = y
            dst[2 * i + 1] = x

        self.add_edges(src, dst)  # remember that DGLMolTree is a subclass of DGLGraph so it has the add_edges method

    def treesize(self):
        return self.number_of_nodes()
    
    def view_graph(self):
        # Convert the DGLGraph into a NetworkX graph
        nx_graph = self.to_networkx()

        # Get the node labels from the `nodes_dict` dictionary
        labels = {i: self.nodes_dict[i]['smiles'] for i in range(self.number_of_nodes())}

        # Draw the graph with labels
        pos = nx.spring_layout(nx_graph)
        nx.draw(nx_graph, pos, with_labels=True, labels=labels, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold')
        plt.title("DGLMolTree Visualization")
        plt.show()
    
    def print_nodes(self):
        for i in range(self.number_of_nodes()):
            print(f"Node {i}: {self.nodes_dict[i]}")

    def print_mol_edge_map(self):
        print("Molecular Edge Map:")
        for tree_edge, mol_edge in self.mol_edge_map.items():
            print(f"Tree Edge {tree_edge} -> Mol Edge {mol_edge}")

    def add_mol_edge_features(self, edge_index, edge_attr):
        """Add the edge features to the DGL graph."""
        mol_edge_attr = {}
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            mol_edge_attr[(src, dst)] = edge_attr[i].tolist() if edge_attr[i].dim() > 0 else edge_attr[i].item()
        self.edge_mol_attr = mol_edge_attr

    def get_edge_index(self):
        """Extract the edge_index matrix from the DGL graph."""
        src, dst = self.edges()
        edge_index = torch.stack([src, dst], dim=0)
        return edge_index

    def get_edge_attr(self):
        """Extract the edge features from the DGL graph.
        get_edge_attr: This function collects the edge features.
        With this, you can easily generate the tensor tree_edge_attr from the molecular edge map and the molecular edge attributes."""
        # now, for each edge in self.edge_mol_attr, create a list of edge attributes
        edge_attr = []
        edge_mol_attr2 = deepcopy(self.edge_mol_attr)
        first_key = next(iter(edge_mol_attr2))
        emb_size = len(edge_mol_attr2[first_key])
       
        assert self.edge_mol_attr is not None, "Devi prima aggiungere le caratteristiche degli archi molecolari al grafo DGL."
        for tree_edge, mol_edge in self.mol_edge_map.items():
            if mol_edge in self.edge_mol_attr:
                edge_attr.append(self.edge_mol_attr[mol_edge])
            else:
                edge_attr.append([0.1] * emb_size)
        
        return torch.tensor(edge_attr).float()

    def create_subgraph(self, node_indices):
        """
        Create a subgraph from the given graph using the specified node indices.
        
        Parameters:
        - graph (DGLGraph): The original graph.
        - node_indices (list of int): List of node indices to include in the subgraph.
        
        Returns:
        - subgraph (DGLGraph): The resulting subgraph.
        """
        node_indices = torch.tensor(node_indices, dtype=torch.int64)
        subgraph = node_subgraph(self, node_indices)
        return subgraph
    

def main():
    # Example of usage

    smiles = '[C-]#N'
    # mol = Chem.MolFromSmiles(smiles)
    # cliques, mst_edges = decompose_and_mst_using_tree_decomp(mol)

    # # Display the results
    # print("Cliques (Atom Indices) and Related Atoms:")
    # for i, mapping in enumerate(cliques):
    #     print(f"Clique {i}: {mapping}")

    # print("\nMST Edge between cliques:")
    # print(mst_edges)

    # # Display the MST graph
    # plot_mst_graph(mol, cliques, mst_edges)

    # Creation of a DGLMolTree
    mol_tree = DGLMolTree(smiles)

    # Visualization of the DGLMolTree
    print("DGLMolTree:")
    print(mol_tree)
    print("\nNumber of nodes in the DGLMolTree:")
    print(mol_tree.treesize())

    # Visualization of the DGLMolTree graph
    mol_tree.print_nodes()

    edges_indexes = mol_tree.get_edge_index()

    edges = mol_tree.edges()

    print("edges: ", len(edges[1]))

    print("edges_indexes: ", edges_indexes)

    mol_tree.print_mol_edge_map()

    # print atom and bond features of the molecule get_mol(smiles)
    mol = get_mol(smiles)
    print("Atom characteristics:")
    for atom in mol.GetAtoms():
        print(f"Simbolo: {atom.GetSymbol()},id Atomo: {atom.GetIdx()}")
    print("Bond characteristics:")
    for bond in mol.GetBonds():
        print(f"Indice Atomo Iniziale: {bond.GetBeginAtomIdx()}, Indice Atomo Finale: {bond.GetEndAtomIdx()}, Tipo Legame: {bond.GetBondType()}")

    # mol_tree.view_graph()
   
#main
# main()
