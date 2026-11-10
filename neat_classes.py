"""
Implémentation NEAT (NeuroEvolution of Augmenting Topologies)
Classes principales pour l'algorithme génétique

Classes principales:
    - MutationConfig: Configuration des hyperparamètres de mutation
    - Node: Représente un neurone (avec méthode mutate)
    - Connection: Représente une connexion entre neurones (avec méthode mutate)
    - Genome: Représente un réseau de neurones complet (avec méthode mutate)
    
Mutations disponibles:
    - Noeuds: Changement de fonction d'activation, modification du biais
    - Connexions: Modification du poids, activation/désactivation
    - Génome: Application de toutes les mutations ci-dessus
"""

from enum import Enum
from typing import List, Optional, Dict
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class MutationConfig:
    """
    Configuration des hyperparamètres de mutation
    
    Tous les taux de mutation sont des probabilités entre 0.0 et 1.0
    """
    # Mutations des noeuds
    prob_mutate_activation: float = 0.1      # Probabilité de changer la fonction d'activation
    prob_mutate_bias: float = 0.3            # Probabilité de modifier le biais
    bias_mutation_power: float = 0.2         # Amplitude de la mutation du biais (écart-type)
    bias_replace_rate: float = 0.01           # Probabilité de remplacer complètement le biais
    
    # Mutations des connexions
    prob_mutate_weight: float = 0.8          # Probabilité de modifier un poids
    weight_mutation_power: float = 0.5       # Amplitude de la mutation du poids (écart-type)
    weight_replace_rate: float = 0.01         # Probabilité de remplacer complètement le poids
    
    prob_toggle_connection: float = 0.01     # Probabilité d'activer/désactiver une connexion
    
    # Mutations structurelles
    prob_add_connection: float = 0.05        # Probabilité d'ajouter une nouvelle connexion
    prob_add_node: float = 0.03              # Probabilité d'ajouter un nouveau noeud
    
    # Limites
    bias_min: float = -2.0                   # Valeur minimale du biais
    bias_max: float = 2.0                    # Valeur maximale du biais
    weight_min: float = -5.0                 # Valeur minimale du poids
    weight_max: float = 5.0                  # Valeur maximale du poids


class InnovationTracker:
    """
    Gestionnaire global des numéros d'innovation pour les connexions
    
    Chaque nouvelle connexion reçoit un numéro d'innovation unique et incrémental.
    """
    _global_innovation_number: int = 0
    
    @classmethod
    def get_next_innovation(cls) -> int:
        """Retourne le prochain numéro d'innovation et incrémente le compteur"""
        cls._global_innovation_number += 1
        return cls._global_innovation_number
    
    @classmethod
    def reset(cls) -> None:
        """Réinitialise le compteur d'innovation (utile pour les tests)"""
        cls._global_innovation_number = 0
    
    @classmethod
    def get_current(cls) -> int:
        """Retourne le numéro d'innovation actuel sans l'incrémenter"""
        return cls._global_innovation_number


class NodeType(Enum):
    """Types de noeuds dans le réseau de neurones"""
    SENSOR = "sensor"      # Noeud d'entrée
    OUTPUT = "output"      # Noeud de sortie
    HIDDEN = "hidden"      # Noeud caché


class ActivationFunction(Enum):
    """Fonctions d'activation disponibles"""
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    LINEAR = "linear"


class Node:
    """
    Représente un noeud (neurone) dans le réseau de neurones
    
    Attributs:
        node_type (NodeType): Type du noeud (sensor, output, hidden)
        activation (ActivationFunction): Fonction d'activation du noeud
        bias (float): Biais du noeud
        layer (float): Couche du noeud (pour l'ordonnancement)
                      - SENSOR: layer = 0.0
                      - OUTPUT: layer = 1.0
                      - HIDDEN: layer dans ]0.0, 1.0[
        value (float): Valeur actuelle du noeud
    """
    
    def __init__(
        self, 
        node_type: NodeType,
        activation: ActivationFunction = ActivationFunction.RELU,
        bias: float = 0.0,
        layer: Optional[float] = None
    ):
        self.node_type = node_type
        self.activation = activation
        self.bias = bias
        
        # Définir automatiquement la couche selon le type de noeud
        if layer is None:
            if node_type == NodeType.SENSOR:
                self.layer = 0.0
            elif node_type == NodeType.OUTPUT:
                self.layer = 1.0
            else:  # HIDDEN
                raise ValueError("Les noeuds HIDDEN doivent avoir layer dans ]0.0, 1.0[. Aucune valeur n'est donnée.")
        else:
            # Vérifier que la couche respecte les contraintes
            if node_type == NodeType.SENSOR and layer != 0.0:
                raise ValueError("Les noeuds SENSOR doivent avoir layer = 0.0")
            elif node_type == NodeType.OUTPUT and layer != 1.0:
                raise ValueError("Les noeuds OUTPUT doivent avoir layer = 1.0")
            elif node_type == NodeType.HIDDEN and not (0.0 < layer < 1.0):
                raise ValueError("Les noeuds HIDDEN doivent avoir layer dans ]0.0, 1.0[")
            self.layer = layer
        
        self.value = 0.0

        
    def activate(self, x: float) -> float:
        """
        Applique la fonction d'activation au signal d'entrée
        
        Args:
            x (float): Signal d'entrée
            
        Returns:
            float: Signal après activation
        """
        x = x + self.bias
        
        if self.activation == ActivationFunction.SIGMOID:
            if x >= 0: #Implémentation avec stabilité numérique
                z = np.exp(-x)
                self.value = 1.0 / (1.0 + z)
            else:
                z = np.exp(x)
                self.value = z / (1.0 + z)
        elif self.activation == ActivationFunction.TANH:
            self.value = np.tanh(x)
        elif self.activation == ActivationFunction.RELU:
            self.value = max(0.0, x)
        elif self.activation == ActivationFunction.LINEAR:
            self.value = x
        else:
            self.value = x

        return self.value

    def __repr__(self) -> str:
        return f"Node(type={self.node_type.value}, activation={self.activation.value}, bias={self.bias:.2f}, layer={self.layer:.2f}, value={self.value:.2f})"
    
    def copy(self) -> 'Node':
        """Crée une copie du noeud (sans les connexions, elles seront reconstruites)"""
        return Node(
            node_type=self.node_type,
            activation=self.activation,
            bias=self.bias,
            layer=self.layer
        )
    
    def mutate(self, config: MutationConfig) -> None:
        """
        Applique des mutations au noeud selon la configuration
        
        Les noeuds SENSOR et OUTPUT ne peuvent pas avoir leur activation mutée
        Les noeuds SENSOR ne peuvent pas avoir leur biais muté (pas de biais)
        
        Args:
            config (MutationConfig): Configuration des paramètres de mutation
        """
        # Les sensors ne mutent pas (activation LINEAR fixe, pas de biais)
        if self.node_type == NodeType.SENSOR:
            return
        
        # Mutation de la fonction d'activation (seulement pour HIDDEN)
        if self.node_type == NodeType.HIDDEN and np.random.random() < config.prob_mutate_activation:
            # Choisir une nouvelle fonction d'activation différente de l'actuelle
            available_activations = [act for act in ActivationFunction if act != self.activation]
            if available_activations:
                self.activation = available_activations[np.random.randint(len(available_activations))]
        
        # Mutation du biais (pour HIDDEN et OUTPUT)
        if np.random.random() < config.prob_mutate_bias:
            if np.random.random() < config.bias_replace_rate:
                # Remplacer complètement le biais
                self.bias = np.random.uniform(config.bias_min, config.bias_max)
            else:
                # Perturber le biais existant
                perturbation = np.random.normal(0, config.bias_mutation_power)
                self.bias += perturbation
                # Appliquer les limites
                self.bias = np.clip(self.bias, config.bias_min, config.bias_max)


class Connection:
    """
    Représente une connexion entre deux noeuds
    
    Attributs:
        innovation (int): Numéro d'innovation (historique de la mutation)
        input_node_idx (int): Index du noeud d'entrée
        output_node_idx (int): Index du noeud de sortie
        weight (float): Poids de la connexion
        enabled (bool): Indique si la connexion est active
    """
    
    def __init__(
        self,
        innovation: int,
        input_node_idx: int,
        output_node_idx: int,
        weight: Optional[float] = None,
        enabled: bool = True,
    ):
        self.innovation = innovation
        self.input_node_idx = input_node_idx
        self.output_node_idx = output_node_idx
        self.weight = weight if weight is not None else np.random.randn()
        self.enabled = enabled
        
    
    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        return f"Connection(innov={self.innovation}, {self.input_node_idx}->{self.output_node_idx}, w={self.weight:.3f}, {status})"
    
    def copy(self) -> 'Connection':
        """Crée une copie de la connexion"""
        return Connection(
            innovation=self.innovation,
            input_node_idx=self.input_node_idx,
            output_node_idx=self.output_node_idx,
            weight=self.weight,
            enabled=self.enabled
        )
    
    def mutate(self, config: MutationConfig) -> None:
        """
        Applique des mutations à la connexion selon la configuration
        
        Args:
            config (MutationConfig): Configuration des paramètres de mutation
        """
        # Mutation du poids
        if np.random.random() < config.prob_mutate_weight:
            if np.random.random() < config.weight_replace_rate:
                # Remplacer complètement le poids
                self.weight = np.random.uniform(config.weight_min, config.weight_max)
            else:
                # Perturber le poids existant
                perturbation = np.random.normal(0, config.weight_mutation_power)
                self.weight += perturbation
                # Appliquer les limites
                self.weight = np.clip(self.weight, config.weight_min, config.weight_max)
        
        # Mutation de l'état (activer/désactiver)
        if np.random.random() < config.prob_toggle_connection:
            self.enabled = not self.enabled


class Genome:
    """
    Représente un individu (génome) dans la population
    Un génome est un réseau de neurones représenté par des noeuds et des connexions
    
    Attributs:
        genome_id (int): Identifiant unique du génome
        nodes (List[Node]): Liste des noeuds du réseau
        connections (List[Connection]): Liste des connexions du réseau
        fitness (float): Score de fitness du génome
        num_inputs (int): Nombre de noeuds d'entrée (sensors)
        num_outputs (int): Nombre de noeuds de sortie (outputs)
        _graph (List[List[Connection]]): Graphe des connexions entrantes par noeud
        _layer_ordering (List[int]): Ordre des noeuds triés par layer (cache)
        _needs_reordering (bool): Indique si le tri par layer doit être recalculé
        _nodes_by_layer (Dict[float, List[int]]): Mapping layer -> indices des noeuds (pour mutations)
    """
    
    def __init__(
        self,
        genome_id: int,
        nodes: Optional[List[Node]] = None,
        connections: Optional[List[Connection]] = None
    ):
        self.genome_id = genome_id
        self.fitness = 0.0
        self.adjusted_fitness = 0.0  # Pour le fitness sharing avec spéciation
        self.species_id: int = -1  # ID de l'espèce (pour le tracking des espèces)
        
        self.nodes = []
        self.connections = []
        
        self._graph: List[List[Connection]] = []
        self._layer_ordering: List[int] = []
        self._needs_reordering: bool = True

        # Compteurs pour les inputs et outputs
        self.num_inputs = 0
        self.num_outputs = 0
        
        # Structure pour optimiser add_connection_mutation
        self._nodes_by_layer: Dict[float, List[int]] = {}  # layer -> liste des indices de noeuds

        if nodes:
            for node in nodes:
                self.add_node(node)

        if connections:
            for conn in connections:
                self.add_connection(conn)

        
    def add_node(self, node: Node) -> None:
        """Ajoute un noeud au génome"""
        
        node_idx = len(self.nodes)
        self.nodes.append(node)
        self._graph.append([])
        
        # Mettre à jour le compteur si c'est un input ou output
        if node.node_type == NodeType.SENSOR:
            self.num_inputs += 1
        elif node.node_type == NodeType.OUTPUT:
            self.num_outputs += 1
        
        # Mettre à jour le dictionnaire pour add_connection_mutation
        if node.layer not in self._nodes_by_layer:
            self._nodes_by_layer[node.layer] = []
        self._nodes_by_layer[node.layer].append(node_idx)
        
        # Marquer qu'il faut retrier les noeuds par layer
        self._needs_reordering = True
    
    def add_connection(self, connection: Connection) -> None:
        """Ajoute une connexion au génome"""
        
        if not ( 0 <= connection.input_node_idx < len(self.nodes)) :
            raise ValueError(f"Index du noeud d'entrée invalide: {connection.input_node_idx}")

        if not ( 0 <= connection.output_node_idx < len(self.nodes)) :
            raise ValueError(f"Index du noeud de sortie invalide: {connection.output_node_idx}")
        
        if connection.input_node_idx == connection.output_node_idx:
            raise ValueError("Une connexion ne peut pas relier un noeud à lui-même.")
        
        if self.nodes[connection.input_node_idx].layer >= self.nodes[connection.output_node_idx].layer:
            raise ValueError(
                f"Connexion invalide: le noeud d'entrée (layer={self.nodes[connection.input_node_idx].layer}) "
                f"doit avoir une couche strictement inférieure à celle du noeud de sortie (layer={self.nodes[connection.output_node_idx].layer})"
            )

        self.connections.append(connection)
        
        # Ajouter la connexion à la liste des connexions entrantes du noeud de sortie
        self._graph[connection.output_node_idx].append(connection)
    
    
    def forward_pass(self, inputs: List[float]) -> List[float]:
        """
        Effectue une passe forward dans le réseau.
        
        Le tri par layer est mis en cache et recalculé uniquement si un noeud a été ajouté.
        
        Args:
            inputs (List[float]): Valeurs d'entrée (doit avoir len == self.num_inputs)
            
        Returns:
            List[float]: Valeurs de sortie
            
        Raises:
            ValueError: Si len(inputs) != self.num_inputs
        """
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {len(inputs)}")
        
        # Réinitialiser les valeurs
        for node in self.nodes:
            node.value = 0.0
        
        # Définir les valeurs d'entrée
        for i in range(self.num_inputs):
            self.nodes[i].value = inputs[i]
        
        # Recalculer l'ordre des noeuds par layer si nécessaire
        if self._needs_reordering:
            self._layer_ordering = sorted(range(len(self.nodes)), 
                                         key=lambda x: self.nodes[x].layer)
            self._needs_reordering = False
        
        # Activer les noeuds dans l'ordre des layers (skip sensors)
        for node_idx in self._layer_ordering[self.num_inputs:]:
            node = self.nodes[node_idx]

            weighted_sum = 0.0
            for conn in self._graph[node_idx]:
                if conn.enabled:
                    weighted_sum += self.nodes[conn.input_node_idx].value * conn.weight
                
            node.activate(weighted_sum)
            
        # Récupérer les outputs (en gardant l'ordre)
        return [node.value for node in self.nodes 
                if node.node_type == NodeType.OUTPUT]
    
    def copy(self) -> 'Genome':
        """Crée une copie profonde du génome"""
        new_nodes = [node.copy() for node in self.nodes]
        new_connections = [conn.copy() for conn in self.connections]
        new_genome = Genome(
            genome_id=self.genome_id,
            nodes=new_nodes,
            connections=new_connections
        )
        new_genome.fitness = self.fitness
        # Marquer qu'il faut retrier (car on vient de copier les noeuds)
        new_genome._needs_reordering = True
        return new_genome
    
    def mutate(self, config: MutationConfig) -> None:
        """
        Applique des mutations au génome entier selon la configuration
        
        Cette méthode mute les noeuds (activation, biais) et les connexions (poids, état).
        Les mutations sont appliquées in-place.
        
        Args:
            config (MutationConfig): Configuration des paramètres de mutation
        """
        # Muter tous les noeuds (sauf les sensors qui ne mutent pas)
        for node in self.nodes:
            node.mutate(config)
        
        # Muter toutes les connexions
        for connection in self.connections:
            connection.mutate(config)
        
        # Mutations structurelles
        if np.random.random() < config.prob_add_connection:
            self.add_connection_mutation()
        
        if np.random.random() < config.prob_add_node:
            self.add_node_mutation()
    
    def add_connection_mutation(self) -> bool:
        """
        Mutation structurelle : Ajouter une nouvelle connexion
        
        1. Tire au hasard deux layers distinctes avec layer_in < layer_out
        2. Pour chaque layer, tire aléatoirement un noeud
        3. Vérifie que la connexion n'existe pas déjà
        4. Ajoute la connexion avec un nouveau numéro d'innovation
        
        Returns:
            bool: True si une connexion a été ajoutée, False sinon
        """
        # Vérifier qu'il y a au moins 2 layers différents
        if len(self._nodes_by_layer) < 2:
            return False
        
        # Obtenir la liste des layers disponibles
        available_layers = list(self._nodes_by_layer.keys())
        
        # Essayer jusqu'à 20 fois de trouver une connexion valide
        max_attempts = 20
        for _ in range(max_attempts):
            # Tirer 2 layers distincts sans remise
            layer1, layer2 = np.random.choice(available_layers, size=2, replace=False)
            
            # Déterminer input et output (min et max)
            in_layer = min(layer1, layer2)
            out_layer = max(layer1, layer2)
            
            # Choisir un noeud aléatoire dans chaque layer
            input_node_idx = self._nodes_by_layer[in_layer][
                np.random.randint(len(self._nodes_by_layer[in_layer]))
            ]
            output_node_idx = self._nodes_by_layer[out_layer][
                np.random.randint(len(self._nodes_by_layer[out_layer]))
            ]
            
            # Vérifier que la connexion n'existe pas déjà (utilise _graph pour un accès rapide)
            connection_exists = any(
                conn.input_node_idx == input_node_idx
                for conn in self._graph[output_node_idx]
            )
            
            if not connection_exists:
                # Créer et ajouter la nouvelle connexion
                new_connection = Connection(
                    innovation=InnovationTracker.get_next_innovation(),
                    input_node_idx=input_node_idx,
                    output_node_idx=output_node_idx,
                    weight=np.random.randn()
                )
                self.add_connection(new_connection)
                return True
        
        # Aucune connexion valide trouvée après max_attempts
        return False
    
    def add_node_mutation(self) -> bool:
        """
        Mutation structurelle : Ajouter un nouveau noeud
        
        1. Choisit une connexion aléatoire et la désactive
        2. Ajoute un nouveau noeud caché avec:
           - layer = moyenne des layers des noeuds de la connexion
           - activation = LINEAR
           - bias = 0.0
        3. Ajoute deux nouvelles connexions:
           - input → nouveau noeud (même poids que l'ancienne connexion)
           - nouveau noeud → output (poids = 1.0)
        
        Returns:
            bool: True si un noeud a été ajouté, False sinon
        """
        # Vérifier qu'il y a au moins une connexion
        if not self.connections:
            return False
        
        # Choisir une connexion aléatoire
        old_connection = self.connections[np.random.randint(len(self.connections))]
        
        # Désactiver l'ancienne connexion
        old_connection.enabled = False
        
        # Récupérer les noeuds de la connexion
        input_node = self.nodes[old_connection.input_node_idx]
        output_node = self.nodes[old_connection.output_node_idx]
        
        # Calculer le layer du nouveau noeud (moyenne des layers)
        new_layer = (input_node.layer + output_node.layer) / 2.0
        
        # Créer le nouveau noeud caché
        new_node = Node(
            node_type=NodeType.HIDDEN,
            activation=ActivationFunction.LINEAR,
            bias=0.0,
            layer=new_layer
        )
        
        # Ajouter le nouveau noeud
        self.add_node(new_node)
        new_node_idx = len(self.nodes) - 1
        
        # Créer la première connexion : input → nouveau noeud (même poids)
        conn1 = Connection(
            innovation=InnovationTracker.get_next_innovation(),
            input_node_idx=old_connection.input_node_idx,
            output_node_idx=new_node_idx,
            weight=old_connection.weight
        )
        
        # Créer la deuxième connexion : nouveau noeud → output (poids = 1.0)
        conn2 = Connection(
            innovation=InnovationTracker.get_next_innovation(),
            input_node_idx=new_node_idx,
            output_node_idx=old_connection.output_node_idx,
            weight=1.0
        )
        
        # Ajouter les deux nouvelles connexions
        self.add_connection(conn1)
        self.add_connection(conn2)
        
        return True
    
    def __repr__(self) -> str:
        """Représentation détaillée du génome"""
        # Compter les types de noeuds
        num_sensors = sum(1 for n in self.nodes if n.node_type == NodeType.SENSOR)
        num_hidden = sum(1 for n in self.nodes if n.node_type == NodeType.HIDDEN)
        num_outputs = sum(1 for n in self.nodes if n.node_type == NodeType.OUTPUT)
        
        # Compter les connexions actives/désactivées
        num_enabled = sum(1 for c in self.connections if c.enabled)
        num_disabled = len(self.connections) - num_enabled
        
        # Construire la représentation
        parts = [
            f"Genome(id={self.genome_id}",
            f"fitness={self.fitness:.4f}",
            f"structure={num_sensors}→{num_hidden}→{num_outputs}",
            f"conn={num_enabled}",
        ]
        
        # Ajouter info sur les connexions désactivées si présentes
        if num_disabled > 0:
            parts.append(f"disabled={num_disabled}")
        
        return ", ".join(parts) + ")"
    
    def __str__(self) -> str:
        """Représentation lisible multi-lignes du génome"""
        # Compter les types de noeuds
        num_sensors = sum(1 for n in self.nodes if n.node_type == NodeType.SENSOR)
        num_hidden = sum(1 for n in self.nodes if n.node_type == NodeType.HIDDEN)
        num_outputs = sum(1 for n in self.nodes if n.node_type == NodeType.OUTPUT)
        
        # Compter les connexions
        num_enabled = sum(1 for c in self.connections if c.enabled)
        num_disabled = len(self.connections) - num_enabled
        
        lines = [
            f"╔═══════════════════════════════════════════════╗",
            f"║ Genome ID: {self.genome_id:<34} ║",
            f"║ Fitness:   {self.fitness:<34.6f} ║",
            f"╠═══════════════════════════════════════════════╣",
            f"║ STRUCTURE                                     ║",
            f"║   Inputs:  {num_sensors:<34} ║",
            f"║   Hidden:  {num_hidden:<34} ║",
            f"║   Outputs: {num_outputs:<34} ║",
            f"║   Total:   {len(self.nodes):<34} ║",
            f"╠═══════════════════════════════════════════════╣",
            f"║ CONNECTIONS                                   ║",
            f"║   Active:   {num_enabled:<33} ║",
            f"║   Disabled: {num_disabled:<33} ║",
            f"║   Total:    {len(self.connections):<33} ║",
            f"╚═══════════════════════════════════════════════╝",
        ]
        
        return "\n".join(lines)
