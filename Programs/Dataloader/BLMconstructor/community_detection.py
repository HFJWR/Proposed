from community import community_louvain
import networkx as nx
import pandas as pd

def  detect_community(retweets, rnd_state):
    """
    input: df['author_id']['mention_id']
    output: df['user']['community']
    """

    # Node list
    author_id = list(retweets["author_id"])
    mention_id = list(retweets['mention_id'])
    node_list = list(set(author_id+mention_id))

    # Edge list
    edge_list = list(zip(retweets["author_id"], retweets["mention_id"]))

    # Graph
    G = nx.Graph()
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)

    DG = nx.MultiDiGraph()
    DG.add_nodes_from(node_list)
    DG.add_edges_from(edge_list)

    # Largest connected components
    largest_cc = max(nx.connected_components(G), key=len)

    DSubgraph = DG.subgraph(largest_cc)
    Subgraph = G.subgraph(largest_cc)
    edges = list(DSubgraph.edges(data=False))

    # Community
    partition = community_louvain.best_partition(Subgraph, random_state=rnd_state)
    comm_df = pd.DataFrame(list(partition.items()), columns=['user', 'community'])

    # LCC
    retweets_LCC = pd.DataFrame(edges, columns=['author_id', 'mention_id'])

    return (comm_df, retweets_LCC)