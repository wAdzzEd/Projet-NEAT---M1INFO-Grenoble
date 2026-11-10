"""
Module pour l'évolution NEAT et la visualisation des résultats.
"""

from neat_classes import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Callable, Dict, Any, Tuple
import time
from heapq import nlargest


def visualize_genome(genome: Genome, title: str = "NEAT Network", figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualise le réseau de neurones d'un génome avec networkx.
    
    Les noeuds sont positionnés par layer (couche) :
    - SENSOR (layer=0.0) à gauche
    - HIDDEN (layer dans ]0.0, 1.0[) au milieu
    - OUTPUT (layer=1.0) à droite
    
    Les connexions actives sont en trait plein, les désactivées en pointillés.
    La couleur des arêtes représente le poids (rouge=négatif, vert=positif).
    
    Args:
        genome (Genome): Le génome à visualiser
        title (str): Titre du graphique
        figsize (Tuple[int, int]): Taille de la figure matplotlib
    """
    G = nx.DiGraph()
    
    # Ajouter les noeuds avec leurs attributs
    for idx, node in enumerate(genome.nodes):
        G.add_node(idx, 
                   node_type=node.node_type.value,
                   layer=node.layer,
                   activation=node.activation.value,
                   bias=node.bias)
    
    # Ajouter les connexions
    for conn in genome.connections:
        G.add_edge(conn.input_node_idx, conn.output_node_idx,
                   weight=conn.weight,
                   enabled=conn.enabled,
                   innovation=conn.innovation)
    
    # Calculer les positions des noeuds par layer
    pos = {}
    for layer, nodes_in_layer in genome._nodes_by_layer.items():
        num_nodes = len(nodes_in_layer)
        for i, node_idx in enumerate(nodes_in_layer):
            # x = layer (0.0 à 1.0), y = position verticale centrée
            x = layer
            y = (i - (num_nodes - 1) / 2) * (1.0 if num_nodes > 1 else 0)
            pos[node_idx] = (x, y)
    
    # Créer la figure
    plt.figure(figsize=figsize)
    
    # Séparer les noeuds par type pour différentes couleurs (couleurs plus saturées)
    sensor_nodes = [idx for idx, node in enumerate(genome.nodes) if node.node_type == NodeType.SENSOR]
    hidden_nodes = [idx for idx, node in enumerate(genome.nodes) if node.node_type == NodeType.HIDDEN]
    output_nodes = [idx for idx, node in enumerate(genome.nodes) if node.node_type == NodeType.OUTPUT]
    
    # Dessiner les noeuds 
    nx.draw_networkx_nodes(G, pos, nodelist=sensor_nodes, node_color='#4A90E2',  # Bleu saturé
                          node_size=700, label='SENSOR', node_shape='s', edgecolors='black', linewidths=2)
    nx.draw_networkx_nodes(G, pos, nodelist=hidden_nodes, node_color='#7ED321',  # Vert saturé
                          node_size=700, label='HIDDEN', edgecolors='black', linewidths=2)
    nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, node_color='#F5A623',  # Orange saturé
                          node_size=700, label='OUTPUT', node_shape='s', edgecolors='black', linewidths=2)
    
    # Séparer les connexions actives et désactivées
    enabled_edges = [(u, v) for u, v, d in G.edges(data=True) if d['enabled']]
    disabled_edges = [(u, v) for u, v, d in G.edges(data=True) if not d['enabled']]
    
    # Récupérer les poids pour la couleur
    enabled_weights = [G[u][v]['weight'] for u, v in enabled_edges]
    
    # Dessiner les connexions actives avec couleur selon le poids 
    if enabled_edges:
        edges_collection = nx.draw_networkx_edges(
            G, pos, edgelist=enabled_edges, edge_color=enabled_weights,
            edge_cmap=plt.get_cmap('RdBu_r'),
            edge_vmin=-2, edge_vmax=2,
            width=2.5, arrows=True, arrowsize=20, arrowstyle='->', alpha=0.8
        )
        # Ajouter une colorbar pour les poids si possible
        if edges_collection is not None and not isinstance(edges_collection, list):
            plt.colorbar(edges_collection, label='Poids de connexion')
    
    # Dessiner les connexions désactivées
    if disabled_edges:
        nx.draw_networkx_edges(G, pos, edgelist=disabled_edges, 
                              style='dashed', edge_color='#666666',  # Gris foncé
                              width=1.5, arrows=True, arrowsize=15, alpha=0.5)
    
    # Ajouter les labels des noeuds
    labels = {idx: f"{idx}" for idx in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
    
    # Titre et légende
    plt.title(f"{title}\nGenome ID: {genome.genome_id}, Fitness: {genome.fitness:.3f}, "
             f"Nodes: {len(genome.nodes)}, Connections: {len(genome.connections)}")
    
    # Placer la légende à l'extérieur du graphique avec espacement vertical
    # markerscale réduit la taille des marqueurs, borderpad augmente la taille de la boîte
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, 
              frameon=True, fancybox=True, shadow=True,
              labelspacing=1.5, handletextpad=0.8, fontsize=10,
              markerscale=0.8,  # Réduit la taille des marqueurs de 40%
              borderpad=1.0)    # Padding interne de la boîte (défaut = 0.4)
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()



def create_initial_genome(
    genome_id: int,
    num_inputs: int,
    num_outputs: int,
    fully_connected: bool = False,
    input_activation: ActivationFunction = ActivationFunction.LINEAR,
    output_activation: ActivationFunction = ActivationFunction.SIGMOID
) -> Genome:
    """
    Crée un génome initial minimal
    
    Args:
        genome_id (int): Identifiant du génome
        num_inputs (int): Nombre de noeuds d'entrée
        num_outputs (int): Nombre de noeuds de sortie
        fully_connected (bool): Si True, connecte tous les inputs aux outputs
        input_activation (ActivationFunction): Fonction d'activation pour les inputs (défaut: LINEAR)
        output_activation (ActivationFunction): Fonction d'activation pour les outputs (défaut: SIGMOID)
        
    Returns:
        Genome: Génome initial
        
    Note:
        Les noeuds SENSOR auront automatiquement layer = 0.0
        Les noeuds OUTPUT auront automatiquement layer = 1.0
    """
    genome = Genome(genome_id=genome_id)
    
    # Créer les noeuds d'entrée (layer = 0.0 automatiquement)
    for _ in range(num_inputs):
        node = Node(
            node_type=NodeType.SENSOR,
            activation=input_activation
        )
        genome.add_node(node)
    
    # Créer les noeuds de sortie (layer = 1.0 automatiquement)
    for i in range(num_outputs):
        node = Node(
            node_type=NodeType.OUTPUT,
            activation=output_activation
        )
        genome.add_node(node)
    
    # Créer les connexions initiales
    if fully_connected:
        connections = [
            Connection(
                innovation=InnovationTracker.get_next_innovation(),
                input_node_idx=inp,
                output_node_idx=out,
                weight=np.random.randn()
            )
            for inp, out in product(range(num_inputs), range(num_inputs, num_inputs + num_outputs))
        ]
        for conn in connections:
            genome.add_connection(conn)
    
    return genome


def crossover(parent1: Genome, parent2: Genome, offspring_id: int) -> Genome:
    """
    Réalise le crossover (croisement) de deux génomes en utilisant l'historical marking.
    
    Principe du crossover NEAT:
    1. Les gènes (connexions) sont alignés par leur numéro d'innovation
    2. Gènes matching (même innovation dans les deux parents):
       - Hérités aléatoirement de l'un ou l'autre parent
    3. Gènes disjoint/excess (présents dans un seul parent):
       - Hérités de parent1 (toujours le plus fit (ou si égalité parent1 aléatoire))
    4. Tous les nœuds de parent1 sont hérités
    
    Args:
        parent1: Premier génome parent 
        parent2: Deuxième génome parent
        offspring_id: ID du génome enfant
        
    Returns:
        Genome: Nouveau génome enfant issu du crossover
        
    Note:
        Si parent2 est plus fit que parent1, ils sont échangés au début.
        Si parent1 et parent2 ont la même fitness, ils sont échangés avec probabilité 0.5.
    """
    
    # Convention: parent1 est toujours le plus fit (ou égal)
    if parent2.fitness > parent1.fitness or (parent2.fitness == parent1.fitness and np.random.random() < 0.5):
        parent1, parent2 = parent2, parent1
    
    # Créer le génome enfant avec tous les nœuds de parent1
    child = Genome(genome_id=offspring_id)
    
    # Copier tous les nœuds de parent1
    for node in parent1.nodes:
        new_node = node.copy()
        child.add_node(new_node)

    
    # Parcours parallèle des connexions (déjà triées par innovation)
    i, j = 0, 0
    conns1 = parent1.connections
    conns2 = parent2.connections
    
    while i < len(conns1) or j < len(conns2):
        conn1 = conns1[i] if i < len(conns1) else None
        conn2 = conns2[j] if j < len(conns2) else None
        
        if conn1 is None:
            break  # Plus de gènes dans parent1
        
        if (conn2 is None) or (conn1.innovation < conn2.innovation):
            # Gène uniquement dans parent1 (le plus fit)
            new_conn = conn1.copy()
            child.add_connection(new_conn)
            i += 1
            continue
        
        if conn1.innovation == conn2.innovation:
            # Gène matching: choisir aléatoirement entre les deux parents
            new_conn = conn1.copy() if np.random.random() < 0.5 else conn2.copy()
            child.add_connection(new_conn)
            i += 1
            j += 1
        else: 
            j += 1
    
    return child


def genome_distance(genome1: Genome, genome2: Genome, 
                   c1: float = 1.0, c2: float = 1.0, c3: float = 0.4) -> float:
    """
    Calcule la distance de compatibilité entre deux génomes selon NEAT.
    
    La distance δ est calculée comme:
    δ = (c1 * E / N) + (c2 * D / N) + c3 * W̄
    
    Où:
    - E = nombre de gènes excess (au-delà du dernier gène commun)
    - D = nombre de gènes disjoint (entre le premier et dernier gène commun)
    - W̄ = différence moyenne des poids des gènes matching
    - N = nombre de gènes dans le plus grand génome (normalisation)
    - c1, c2, c3 = coefficients d'importance
    
    Args:
        genome1: Premier génome
        genome2: Deuxième génome
        c1: Coefficient pour les gènes excess (défaut: 1.0)
        c2: Coefficient pour les gènes disjoint (défaut: 1.0)
        c3: Coefficient pour la différence de poids (défaut: 0.4)
        
    Returns:
        float: Distance de compatibilité entre les deux génomes
    """
    conns1 = genome1.connections
    conns2 = genome2.connections
    
    # Cas particuliers
    if len(conns1) == 0 and len(conns2) == 0:
        return 0.0
    if len(conns1) == 0 or len(conns2) == 0:
        return c1 * max(len(conns1), len(conns2))
    
    # Normalisation N (nombre de gènes dans le plus grand génome)
    N = max(len(conns1), len(conns2))
    if N < 20:  # Petits génomes : pas de normalisation
        N = 1
    
    # Parcours parallèle des connexions (déjà triées par innovation)
    num_disjoint = 0
    num_excess = 0
    num_matching = 0
    weight_diff_sum = 0.0
    
    i, j = 0, 0
    
    # Parcourir tant que les deux listes ont des éléments
    while i < len(conns1) and j < len(conns2):
        conn1 = conns1[i]
        conn2 = conns2[j]
        
        if conn1.innovation == conn2.innovation:
            # Gène matching: accumuler la différence de poids
            weight_diff_sum += abs(conn1.weight - conn2.weight)
            num_matching += 1
            i += 1
            j += 1
            
        elif conn1.innovation < conn2.innovation:
            # Gène disjoint dans genome1
            num_disjoint += 1
            i += 1
            
        else:  # conn2.innovation < conn1.innovation
            # Gène disjoint dans genome2
            num_disjoint += 1
            j += 1
    
    # Tous les gènes restants sont des excess
    num_excess = (len(conns1) - i) + (len(conns2) - j)
    
    # Différence moyenne des poids pour les gènes matching
    avg_weight_diff = weight_diff_sum / num_matching if num_matching > 0 else 0.0
    
    # Calculer la distance
    distance = (c1 * num_excess / N) + (c2 * num_disjoint / N) + (c3 * avg_weight_diff)
    
    return float(distance)


def speciate_population(
    population: List[Genome],
    species_representatives: List[Genome],
    compatibility_threshold: float = 3.0,
    c1: float = 1.0,
    c2: float = 1.0,
    c3: float = 0.4,
    next_species_id: int = 0
) -> Tuple[List[List[Genome]], List[Genome], int]:
    """
    Répartit une population de génomes en espèces selon la compatibilité.
    
    Algorithme NEAT de spéciation avec IDs permanents:
    1. Chaque espèce existante est représentée par un génome de la génération précédente
    2. Pour chaque génome de la population actuelle:
       - Le placer dans la première espèce compatible (distance < threshold)
       - Si aucune espèce n'est compatible, créer une nouvelle espèce avec un nouvel ID
    3. Les espèces ne se chevauchent pas (un génome = une seule espèce)
    4. Les IDs d'espèces sont permanents (permettent le suivi entre générations)
    
    Args:
        population: Liste des génomes de la génération actuelle
        species_representatives: Liste des représentants (chaque représentant a son species_id)
        compatibility_threshold: Seuil de distance pour la compatibilité (défaut: 3.0)
        c1: Coefficient pour les gènes excess (défaut: 1.0)
        c2: Coefficient pour les gènes disjoint (défaut: 1.0)
        c3: Coefficient pour la différence de poids (défaut: 0.4)
        next_species_id: Prochain ID d'espèce disponible
        
    Returns:
        Tuple[List[List[Genome]], List[Genome], int]:
            - Liste des espèces: [[génomes], [génomes], ...]
            - Liste des nouveaux représentants (chacun avec son species_id)
            - Prochain ID d'espèce disponible
            
    Note:
        Les représentants pour la prochaine génération sont choisis aléatoirement
        parmi les membres de chaque espèce.
        Chaque génome se voit attribuer un species_id via son attribut genome.species_id.
    """
    #todo: Implémenter la fonction de spéciation en suivant l'algorithme décrit ci-dessus.
    return species_list, new_representatives, next_species_id


def reproduce_species(
    species_list: List[List[Genome]],
    population_size: int,
    survival_rate: float,
    mutation_config: MutationConfig,
    next_genome_id: int,
    crossover_prob: float = 0.5,
) -> List[Genome]:
    """
    Reproduit chaque espèce pour créer la nouvelle génération.
    
    Algorithme:
    1. Calculer le fitness sharing (diviser par taille d'espèce)
    2. Allouer les descendants proportionnellement aux fitness ajustées
    3. Éliminer les membres faibles de chaque espèce (garder survival_rate)
    4. Reproduire (75% crossover, 25% asexué) et muter
    
    Args:
        species_list: Liste de tuples (species_id, liste de génomes)
        population_size: Taille totale de la population à générer
        survival_rate: Proportion de survivants dans chaque espèce (0.0 à 1.0)
        mutation_config: Configuration des mutations
        next_genome_id: Prochain ID de génome à utiliser
        crossover_prob: Probabilité de crossover lors de la reproduction (défaut: 0.5)
        
    Returns:
        List[Genome]: Nouvelle population
    """
    # 1. FITNESS SHARING: Ajuster la fitness en la divisant par la taille de l'espèce
    for species in species_list:
        species_size = len(species)
        for genome in species:
            genome.adjusted_fitness = genome.fitness / species_size
    
    # 2. ALLOCATION DES DESCENDANTS
    
    # Calculer la somme totale des fitness ajustées
    species_fitness = [sum(g.adjusted_fitness for g in species) for species in species_list]
    total_adjusted_fitness = sum(species_fitness)
    
    assert total_adjusted_fitness > 0, "La fitness ajustée totale est nulle, impossible d'allouer des descendants."
    
    # Calculer le nombre de descendants par espèce
    offspring_per_species = []
    total_offspring_allocated = 0

    for species, species_adjusted_fitness in zip(species_list, species_fitness):
        offspring_count = int((species_adjusted_fitness / total_adjusted_fitness) * population_size)
        offspring_per_species.append(offspring_count)
        total_offspring_allocated += offspring_count
    
    # Distribuer les descendants restants aux meilleures espèces
    remaining = population_size - total_offspring_allocated
    if remaining > 0:
        # Prendre au maximum 'remaining' meilleures espèces (cyclique si plus)
        num_to_select = min(remaining, len(species_list))
        best_species_indices = nlargest(num_to_select, 
                                        range(len(species_list)), 
                                        key=lambda idx: species_fitness[idx])
        
        # Distribuer cycliquement aux meilleures espèces
        for i in range(remaining):
            offspring_per_species[best_species_indices[i]] += 1
    
    # 3. REPRODUCTION PAR ESPÈCE
    new_population = []
    current_genome_id = next_genome_id

    for species, num_offspring in zip(species_list, offspring_per_species):
        
        if num_offspring == 0:
            continue
        species_id = species[0].species_id  # Tous les survivants ont le même species_id
        # Éliminer les membres les moins performants (garder survival_rate)
        num_survivors = max(1, int(len(species) * survival_rate))
        survivors = species[:num_survivors]
        
        # Tirer le nombre de crossovers avec une loi binomiale
        num_crossovers = np.random.binomial(num_offspring, crossover_prob) if num_survivors > 1 else 0
        num_asexual = num_offspring - num_crossovers

        offspring_list = []

        # Générer les crossovers en batch
        if num_crossovers > 0:
            idx1 = np.random.randint(0, num_survivors, size=num_crossovers)
            idx2 = np.random.randint(0, num_survivors - 1, size=num_crossovers)
            # Décaler idx2 pour éviter idx1 == idx2
            idx2 = np.where(idx2 >= idx1, idx2 + 1, idx2)
            
            for i1, i2 in zip(idx1, idx2):
                child = crossover(survivors[i1], survivors[i2], current_genome_id)
                child.species_id = species_id
                current_genome_id += 1
                offspring_list.append(child)

        # Générer les reproductions asexuées en batch
        if num_asexual > 0:
            if num_survivors == 1:
                # Cas spécial : un seul parent, clonage multiple
                parent = survivors[0]
                for _ in range(num_asexual):
                    child = parent.copy()
                    child.genome_id = current_genome_id
                    child.species_id = species_id
                    current_genome_id += 1
                    offspring_list.append(child)
            else:
                # Tirer les indices des parents aléatoirement
                parent_indices = np.random.randint(0, num_survivors, size=num_asexual)
                for idx in parent_indices:
                    child = survivors[idx].copy()
                    child.genome_id = current_genome_id
                    child.species_id = species_id
                    current_genome_id += 1
                    offspring_list.append(child)

        new_population.extend(offspring_list)
    
    for genome in new_population:
        genome.mutate(mutation_config)
    return new_population


def evolve_population(
    evaluate_fn: Callable[[Genome], float],
    num_inputs: int,
    num_outputs: int,
    population_size: int = 100,
    num_epochs: int = 150,
    survival_rate: float = 0.5,
    keep_best: int = 0,
    mutation_config: MutationConfig = MutationConfig(),
    fully_connected: bool = True,
    random_seed: int = 42,
    verbose: bool = True,
    verbose_interval: int = 5,
    early_stop_fitness: float = 0.9999,
    use_species: bool = True,
    c1: float = 1.0,
    c2: float = 1.0,
    c3: float = 1.0,
    compatibility_threshold: float = 3.0,
    crossover_prob: float = 0.5
) -> Dict[str, Any]:
    """
    Exécute l'algorithme d'évolution NEAT.
    
    Args:
        evaluate_fn: Fonction d'évaluation (Genome -> float)
        num_inputs: Nombre d'entrées du réseau
        num_outputs: Nombre de sorties du réseau
        population_size: Taille de la population
        num_epochs: Nombre d'époques
        survival_rate: Taux de survie (0.0 à 1.0)
        keep_best: Nombre d'individus élites à garder sans mutation (élitisme)
        mutation_config: Configuration des mutations (None = config par défaut)
        fully_connected: Si True, commence avec des réseaux fully connected
        random_seed: Seed aléatoire pour la reproductibilité (None = pas de seed)
        verbose: Afficher les progrès
        verbose_interval: Intervalle d'affichage (en époques)
        early_stop_fitness: Seuil de fitness pour arrêt précoce (None = pas d'arrêt)
        use_species: bool pour utiliser ou non la spéciation
        c1, c2, c3 = coefficients d'importance pour la distance de compatibilité
        compatibility_threshold: Seuil de distance pour la spéciation
        crossover_prob: Probabilité de crossover lors de la reproduction (défaut: 0.5)
    
    Returns:
        Dictionnaire contenant l'historique de l'évolution:
            - best_fitness, median_fitness, mean_fitness, worst_fitness
            - q25_fitness, q75_fitness
            - deciles: dict avec clés 0, 10, 20, ..., 100
            - avg_nodes, avg_connections, avg_hidden
            - best_genome: meilleur génome trouvé
            - population: population finale
    """
    # Fixer le seed si spécifié
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Verification que keep_best est cohérent avec la taille de la population et le taux de survie
    num_survivors = int(population_size * survival_rate)
    if keep_best >= num_survivors:
        raise ValueError(f"keep_best ({keep_best}) doit être inférieur au nombre de survivants ({num_survivors})")
    
    # Configuration par défaut si non spécifiée
    if mutation_config is None:
        mutation_config = MutationConfig()

    # Réinitialiser le tracker d'innovation
    InnovationTracker.reset()
    
    # Créer la population initiale
    population: List[Genome] = []
    for i in range(population_size):
        genome = create_initial_genome(
            genome_id=i,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            fully_connected=fully_connected
        )
        population.append(genome)
    
    # Mutation initiale pour créer de la diversité
    for genome in population:
        genome.mutate(mutation_config)
        
    species_representatives: List[Genome] = []
    next_species_id = 0
    
    # Structures pour tracker les statistiques
    history = {
        'avg_nodes': [],
        'avg_connections': [],
        'avg_hidden': [],
        'best_genome': None,
        'deciles': {d: [] for d in range(0, 101, 10)},
        # Statistiques des espèces (pour stacked area plot)
        'species_info': {},  # {species_id: {'birth': epoch, 'death': epoch or None, 'sizes': [size_per_epoch]}}
    }
    
    if verbose:
        print("=" * 70)
        print("DÉBUT DE L'ÉVOLUTION")
        print("=" * 70)
    
    
    for epoch in range(num_epochs):
        # 1. ÉVALUATION: Calculer la fitness de chaque génome (prise en compte de l'elitisme)
        start_evaluation = time.time()
        if epoch == 0 or keep_best == 0:
            for genome in population:
                genome.fitness = evaluate_fn(genome)
                assert genome.fitness > 0, "La fitness doit être positive"
        else:
            # Évaluer uniquement les nouveaux génomes (non élites)
            for genome in population[:-keep_best]:
                genome.fitness = evaluate_fn(genome)
                assert genome.fitness > 0, "La fitness doit être positive"
        time_evaluation = time.time() - start_evaluation
        
        # 2. TRI: Trier par fitness (décroissant)
        population.sort(key=lambda g: g.fitness, reverse=True)
        
####################################################################################################################################################################################
        # Statistiques des individus
        # Population déjà triée par fitness (décroissant) : on peut lire directement les déciles

        for d in range(0, 101, 10):
            idx = int((100 - d) * (population_size - 1) / 100)
            history['deciles'][d].append(population[idx].fitness)
        
        history['avg_nodes'].append(np.mean([len(g.nodes) for g in population]))
        history['avg_connections'].append(np.mean([len(g.connections) for g in population]))
        history['avg_hidden'].append(np.mean([sum(1 for n in g.nodes if n.node_type.value == 'hidden') for g in population]))
        
        # Sauvegarder le meilleur génome
        if history['best_genome'] is None or population[0].fitness > history['best_genome'].fitness:
            history['best_genome'] = population[0].copy()

####################################################################################################################################################################################

        # Affichage périodique
        if verbose and (epoch % verbose_interval == 0 or epoch == num_epochs - 1):
            print(f"\nÉpoque {epoch:3d}/{num_epochs}:")
            print(f"  Best:   {population[0].fitness:.6f} | Nodes: {len(population[0].nodes):2d} | Conn: {len(population[0].connections):2d}")
            print(f"  Median: {history['deciles'][50][-1]:.6f}")
            print(f"  Temps évaluation: {time_evaluation*1000:.2f} ms")
        
        # Arrêt précoce si solution trouvée
        if early_stop_fitness is not None and population[0].fitness >= early_stop_fitness:
            if verbose:
                print(f"\n Solution trouvée à l'époque {epoch}! (fitness: {population[0].fitness:.6f} >= {early_stop_fitness})")
            break
        
        # 3. SPÉCIATION ou SÉLECTION SIMPLE
        if use_species:
            # MODE AVEC SPÉCIATION (NEAT complet)
            start_speciation = time.time()
            species_list, species_representatives, next_species_id = speciate_population(
                population, species_representatives,
                c1=c1, c2=c2, c3=c3, 
                compatibility_threshold=compatibility_threshold,
                next_species_id=next_species_id
            )
            time_speciation = time.time() - start_speciation
####################################################################################################################################################################################
            # Statistiques des espèces (à partir de l'epoch 1)
            
            if epoch >= 1:  # Commencer le tracking à partir de la deuxième epoch
                # Dictionnaire des tailles d'espèces pour cette epoch
                current_species_sizes = {}
                
                for species in species_list:
                    if len(species) > 0:
                        # Récupérer l'ID de l'espèce depuis le premier génome
                        species_id = species[0].species_id
                        species_size = len(species)
                        current_species_sizes[species_id] = species_size
                        
                        # Si c'est une nouvelle espèce, l'enregistrer
                        if species_id not in history['species_info']:
                            history['species_info'][species_id] = {
                                'birth': epoch,
                                'death': None,
                                'sizes': []
                            }
                
                # Mettre à jour les tailles et marquer les espèces disparues
                for species_id in history['species_info']:
                    if history['species_info'][species_id]['death'] is None:  # Espèce encore active
                        if species_id in current_species_sizes:
                            # Espèce toujours présente
                            history['species_info'][species_id]['sizes'].append(current_species_sizes[species_id])
                        else:
                            # Espèce disparue cette epoch
                            history['species_info'][species_id]['death'] = epoch
                            history['species_info'][species_id]['sizes'].append(0)
            
####################################################################################################################################################################################
            
            # 4. ÉLITISME: Conserver les meilleurs génomes sans mutation
            elite_population = population[:keep_best]

            # 5. SÉLECTION ET REPRODUCTION: Créer la nouvelle génération
            next_genome_id = population_size * (epoch + 1)

            start_reproduction = time.time()
            new_population = reproduce_species(
                species_list=species_list,
                population_size=population_size - keep_best,
                survival_rate=survival_rate,
                mutation_config=mutation_config,
                next_genome_id=next_genome_id,
                crossover_prob=crossover_prob
            )
            time_reproduction = time.time() - start_reproduction
            
            # 6. COMBINER: Élites + Nouveaux génomes
            new_population.extend(elite_population)
            population = new_population
            
            # Affichage des temps et espèces
            if verbose and (epoch % verbose_interval == 0 or epoch == num_epochs - 1):
                print(f"  Espèces actives: {len(species_list)}")
                print(f"  Temps spéciation: {time_speciation*1000:.2f} ms | Temps reproduction: {time_reproduction*1000:.2f} ms")
        
        else:
            # MODE SANS SPÉCIATION (Algorithme génétique simple)
            
            # 4. ÉLITISME: Conserver les meilleurs génomes sans mutation
            elite_population = population[:keep_best]
            
            # 5. SÉLECTION: Déterminer le nombre de survivants
            num_survivors = int(population_size * survival_rate)
            survivors = population[:num_survivors]
            
            # 6. REPRODUCTION: Générer population_size - keep_best nouveaux génomes
            num_offspring = population_size - keep_best
            next_genome_id = population_size * (epoch + 1)
            
            start_reproduction = time.time()
            
            # Déterminer combien de crossovers vs asexuel (binomial distribution)
            num_crossovers = np.random.binomial(num_offspring, crossover_prob)
            num_asexual = num_offspring - num_crossovers
            
            new_population = []
            
            # CROSSOVER: Générer les enfants par crossover
            if num_crossovers > 0:
                # Vectorisation: générer toutes les paires de parents en une fois
                idx1 = np.random.randint(0, num_survivors, size=num_crossovers)
                idx2 = np.random.randint(0, num_survivors - 1, size=num_crossovers)
                idx2 = np.where(idx2 >= idx1, idx2 + 1, idx2)
                
                for i in range(num_crossovers):
                    child = crossover(survivors[idx1[i]], survivors[idx2[i]], next_genome_id)
                    child.mutate(mutation_config)
                    new_population.append(child)
                    next_genome_id += 1
            
            # ASEXUEL: Reproduction par mutation seule
            if num_asexual > 0:
                parent_indices = np.random.randint(0, num_survivors, size=num_asexual)
                for idx in parent_indices:
                    parent = survivors[idx]
                    child = parent.copy()
                    child.genome_id = next_genome_id
                    child.mutate(mutation_config)
                    new_population.append(child)
                    next_genome_id += 1
            
            time_reproduction = time.time() - start_reproduction
            
            # 7. COMBINER: Nouveaux génomes + Élites
            new_population.extend(elite_population)
            population = new_population
            
            # Affichage du temps de reproduction
            if verbose and (epoch % verbose_interval == 0 or epoch == num_epochs - 1):
                print(f"  Temps reproduction: {time_reproduction*1000:.2f} ms")

    
    if verbose:
        print("\n" + "=" * 70)
        print("FIN DE L'ÉVOLUTION")
        print("=" * 70)
        print(f"Innovation finale: {InnovationTracker.get_current()}")
    
    # Sauvegarder la population finale
    history['population'] = population
    
    return history


def plot_evolution_results(
    history: Dict[str, Any],
    population_size: int = 100,
    figsize_fitness: tuple = (14, 7),
    figsize_complexity: tuple = (18, 5),
    log_scale: bool = True
):
    """
    Visualise les résultats de l'évolution.
    
    Args:
        history: Dictionnaire retourné par evolve_population()
        population_size: Taille de la population (pour le titre, optionnel)
        figsize_fitness: Taille de la figure fitness
        figsize_complexity: Taille de la figure complexité
        log_scale: Utiliser l'échelle logarithmique sur l'axe Y
    """
    deciles = history['deciles']
    num_epochs = len(deciles[0])
    epochs_range = range(num_epochs)
    
    # === GRAPHIQUE 1: Tous les déciles ===
    plt.figure(figsize=figsize_fitness)
    
    # Tracer les déciles en gris (sauf médiane et best)
    for d in [0, 10, 20, 30, 40, 60, 70, 80, 90]:
        plt.plot(epochs_range, deciles[d], linewidth=1.2, color='gray', alpha=0.4, zorder=1)
    
    # Médiane en rouge (plus épaisse)
    plt.plot(epochs_range, deciles[50], linewidth=2.5, color='red', label='Médiane (D50)', zorder=3)
    
    # Meilleure en noir (la plus épaisse)
    plt.plot(epochs_range, deciles[100], linewidth=3, color='black', label='Best (D100)', zorder=4)
    
    # Ajouter les labels des déciles gris dans la légende
    plt.plot([], [], linewidth=1.2, color='gray', alpha=0.4, label='Déciles (D0-D90)')
    
    plt.xlabel('Époque', fontsize=13, fontweight='bold')
    plt.ylabel('Fitness', fontsize=13, fontweight='bold')
    plt.title('Évolution de la Fitness - Distribution complète (déciles)', fontsize=15, fontweight='bold')
    
    if log_scale:
        plt.yscale('log')
    
    plt.legend(loc='lower right', fontsize=11, framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.7)
    plt.tight_layout()
    plt.show()
    
    # === GRAPHIQUE 2: Complexité (Noeuds, Connexions, Cachés) ===
    fig, axes = plt.subplots(1, 3, figsize=figsize_complexity)
    
    # Noeuds
    axes[0].plot(epochs_range, history['avg_nodes'], linewidth=2.5, color='blue', marker='o', markersize=4)
    axes[0].set_xlabel('Époque', fontsize=12)
    axes[0].set_ylabel('Nombre moyen' + (' (log)' if log_scale else ''), fontsize=12)
    axes[0].set_title('Noeuds Totaux', fontsize=14, fontweight='bold')
    if log_scale:
        axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Connexions
    axes[1].plot(epochs_range, history['avg_connections'], linewidth=2.5, color='green', marker='o', markersize=4)
    axes[1].set_xlabel('Époque', fontsize=12)
    axes[1].set_ylabel('Nombre moyen' + (' (log)' if log_scale else ''), fontsize=12)
    axes[1].set_title('Connexions', fontsize=14, fontweight='bold')
    if log_scale:
        axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    # Noeuds cachés
    axes[2].plot(epochs_range, history['avg_hidden'], linewidth=2.5, color='orange', marker='o', markersize=4)
    axes[2].set_xlabel('Époque', fontsize=12)
    axes[2].set_ylabel('Nombre moyen' + (' (log)' if log_scale else ''), fontsize=12)
    axes[2].set_title('Noeuds Cachés', fontsize=14, fontweight='bold')
    if log_scale:
        axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Titre avec info sur la population si disponible
    if population_size is not None:
        plt.suptitle(f'Évolution de la Complexité ({population_size} individus, {num_epochs} époques)', 
                     fontsize=14, fontweight='bold', y=1.02)
    else:
        plt.suptitle(f'Évolution de la Complexité ({num_epochs} époques)', 
                     fontsize=14, fontweight='bold', y=1.02)
    
    plt.show()


def plot_species_evolution(history: Dict[str, Any], figsize: tuple = (14, 6)):
    """
    Visualise l'évolution des espèces avec un stacked area plot.
    
    Utilise history['species_info'] pour tracer l'évolution de la taille de chaque espèce
    au fil des epochs, montrant leur apparition et disparition.
    
    Args:
        history: Dictionnaire retourné par evolve_population() (doit contenir 'species_info')
        figsize: Taille de la figure pour chaque graphique
    """
    species_info = history.get('species_info', {})
    
    if not species_info:
        print("Aucune donnée d'espèce disponible. Assurez-vous d'avoir utilisé use_species=True.")
        return
    
    # Déterminer le nombre total d'epochs
    max_epochs = max(len(info['sizes']) for info in species_info.values())
    epochs_range = range(max_epochs)
    
    # Trier les espèces par ordre d'apparition (birth)
    all_species_ids = sorted(species_info.keys(), key=lambda sid: species_info[sid]['birth'])
    
    # Créer une matrice où chaque ligne = une espèce
    species_matrix = np.zeros((len(all_species_ids), max_epochs))
    
    for idx, species_id in enumerate(all_species_ids):
        info = species_info[species_id]
        birth = info['birth']
        sizes = info['sizes']
        
        # Remplir la matrice : 0 avant birth, puis les tailles
        for epoch in range(max_epochs):
            if epoch < birth:
                species_matrix[idx, epoch] = 0
            elif epoch - birth < len(sizes):
                species_matrix[idx, epoch] = sizes[epoch - birth]
            else:
                species_matrix[idx, epoch] = 0
    
    # ========== GRAPHIQUE 1: Stacked Area Plot ==========
    plt.figure(figsize=figsize)
    colors = plt.cm.get_cmap('gist_rainbow')(np.linspace(0, 1, len(all_species_ids)))
    
    plt.stackplot(epochs_range, species_matrix, 
                  labels=[f'Espèce {sid}' for sid in all_species_ids],
                  colors=colors, alpha=0.8)
    
    plt.xlabel('Époque', fontsize=12, fontweight='bold')
    plt.ylabel('Nombre de génomes', fontsize=12, fontweight='bold')
    plt.title('Évolution des Espèces - Stacked Area Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Légende seulement si pas trop d'espèces
    if len(all_species_ids) <= 20:
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    plt.show()
    
    # ========== GRAPHIQUE 2: Nombre d'espèces actives ==========
    # Calculer le nombre d'espèces actives par epoch
    num_species_per_epoch = []
    for epoch in range(max_epochs):
        count = np.sum(species_matrix[:, epoch] > 0)
        num_species_per_epoch.append(count)
    
    plt.figure(figsize=figsize)
    plt.plot(epochs_range, num_species_per_epoch, 
             linewidth=2.5, color='darkblue', marker='o', markersize=4)
    plt.fill_between(epochs_range, 0, num_species_per_epoch, alpha=0.3, color='lightblue')
    
    plt.xlabel('Époque', fontsize=12, fontweight='bold')
    plt.ylabel('Espèces actives', fontsize=12, fontweight='bold')
    plt.title('Nombre d\'Espèces Actives par Époque', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Statistiques textuelles
    avg_species = np.mean(num_species_per_epoch)
    max_species_count = max(num_species_per_epoch)
    min_species_count = min(num_species_per_epoch)
    total_species = len(all_species_ids)
    
    textstr = f'Total: {total_species}\nMoyenne: {avg_species:.1f}\nMax: {max_species_count}\nMin: {min_species_count}'
    plt.text(0.98, 0.98, textstr, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Afficher un résumé textuel des espèces
    print("\n" + "=" * 70)
    print("RÉSUMÉ DES ESPÈCES")
    print("=" * 70)
    print(f"Nombre total d'espèces apparues: {total_species}")
    print(f"Espèces actives à la fin: {num_species_per_epoch[-1]}")
    print(f"\nTop 5 espèces les plus durables:")
    
    # Calculer la durée de vie de chaque espèce
    species_lifespans = []
    for species_id in all_species_ids:
        info = species_info[species_id]
        birth = info['birth']
        death = info['death'] if info['death'] is not None else max_epochs
        lifespan = death - birth
        max_size = max(info['sizes']) if info['sizes'] else 0
        species_lifespans.append((species_id, birth, death, lifespan, max_size))
    
    # Trier par durée de vie (puis par taille max)
    species_lifespans.sort(key=lambda x: (x[3], x[4]), reverse=True)
    
    for i, (sid, birth, death, lifespan, max_size) in enumerate(species_lifespans[:5]):
        death_str = f"{death}" if death < max_epochs else "fin"
        print(f"  {i+1}. Espèce {sid}: Epoch {birth} → {death_str} ({lifespan} epochs) | Taille max: {max_size}")

