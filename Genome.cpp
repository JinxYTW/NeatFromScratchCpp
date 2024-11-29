#include "Genome.h"
#include "neat.h"
#include <optional>
#include <iostream>
#include <vector>
#include <functional>

// Constructeur par défaut
Genome::Genome() : genome_id(0), num_inputs(0), num_outputs(0) {}

Genome::Genome(int id, int num_inputs, int num_outputs)
    : genome_id(id), num_inputs(num_inputs), num_outputs(num_outputs) {}

// Crée un nouveau génome avec les neurones d'entrée, de sortie et un certain nombre de neurones cachés
// Fonction auxiliaire pour vérifier si un lien créerait un cycle
bool Genome::would_create_cycle(int input_id, int output_id) const {
    std::unordered_set<int> visited;
    std::unordered_map<int, std::vector<int>> graph;

    // Construire le graphe actuel des connexions
    for (const auto &link : links) {
        if (link.is_enabled) {
            graph[link.link_id.input_id].push_back(link.link_id.output_id);
        }
    }

    // Effectuer une recherche en profondeur pour voir s'il existe un chemin de output_id vers input_id
    std::function<bool(int)> dfs = [&](int current) {
        if (current == input_id) return true;
        if (visited.count(current)) return false;

        visited.insert(current);
        for (int neighbor : graph[current]) {
            if (dfs(neighbor)) return true;
        }
        return false;
    };

    return dfs(output_id);
}

// Fonction de création du génome avec vérification des cycles
Genome Genome::create_genome(int id, int num_inputs, int num_outputs, int num_hidden_neurons, RNG &rng) {
    Genome genome(id, num_inputs, num_outputs);

    // Ajoute neurones d'entrée
    for (int i = 0; i < num_inputs; ++i) {
        genome.add_neuron(neat::NeuronGene{i, 0.0, Activation(Activation::Type::Sigmoid)});
    }

    // Ajoute neurones de sortie
    for (int i = 0; i < num_outputs; ++i) {
        int output_id = num_inputs + i;
        genome.add_neuron(neat::NeuronGene{output_id, 0.0, Activation(Activation::Type::Sigmoid)});
    }

    // Ajoute neurones cachés
    for (int i = 0; i < num_hidden_neurons; ++i) {
        int hidden_id = num_inputs + num_outputs + i;
        genome.add_neuron(neat::NeuronGene{hidden_id, 0.0, Activation(Activation::Type::Sigmoid)});
    }

    // Liens : entrée -> cachés
    for (int input_id = 0; input_id < num_inputs; ++input_id) {
        for (int hidden_id = num_inputs + num_outputs; hidden_id < num_inputs + num_outputs + num_hidden_neurons; ++hidden_id) {
            if (!genome.would_create_cycle(input_id, hidden_id)) {
                genome.add_link(genome.create_link(input_id, hidden_id, rng));
            }
        }
    }

    // Liens : cachés -> cachés (pour favoriser l'émergence de structures complexes)
    for (int hidden_id = num_inputs + num_outputs; hidden_id < num_inputs + num_outputs + num_hidden_neurons; ++hidden_id) {
        for (int target_hidden_id = hidden_id + 1; target_hidden_id < num_inputs + num_outputs + num_hidden_neurons; ++target_hidden_id) {
            if (!genome.would_create_cycle(hidden_id, target_hidden_id)) {
                genome.add_link(genome.create_link(hidden_id, target_hidden_id, rng));
            }
        }
    }

    // Liens : cachés -> sorties
    for (int hidden_id = num_inputs + num_outputs; hidden_id < num_inputs + num_outputs + num_hidden_neurons; ++hidden_id) {
        for (int output_id = num_inputs; output_id < num_inputs + num_outputs; ++output_id) {
            if (!genome.would_create_cycle(hidden_id, output_id)) {
                genome.add_link(genome.create_link(hidden_id, output_id, rng));
            }
        }
    }

    // Affichage pour débogage
    std::cout << "Genome ID: " << id << std::endl;
    std::cout << "Neurones:" << std::endl;
    for (const auto &neuron : genome.get_neurons()) {
        std::cout << "  Neuron ID: " << neuron.neuron_id 
                  << ", Biais: " << neuron.bias << std::endl;
    }
    std::cout << "Liens:" << std::endl;
    for (const auto &link : genome.get_links()) {
        std::cout << "  Link from " << link.link_id.input_id 
                  << " to " << link.link_id.output_id 
                  << " with weight " << link.weight << std::endl;
    }

    return genome;
}




// Crée un lien avec des poids aléatoires
neat::LinkGene Genome::create_link(int input_id, int output_id, RNG &rng) {
    return neat::LinkGene{{input_id, output_id}, rng.next_gaussian(0.0, 1.0), true};
}

neat::NeuronGene Genome::create_neuron(int neuron_id) {
    return neat::NeuronGene{neuron_id, 0.0, Activation(Activation::Type::Sigmoid)};
}


int Genome::get_num_inputs() const {
    return num_inputs;  // Retourne le nombre d'entrées
}

int Genome::get_num_outputs() const {
    return num_outputs;  // Retourne le nombre de sorties
}

int Genome::get_genome_id() const {
    return genome_id;  // Retourne l'ID du génome
}

std::vector<neat::NeuronGene> Genome::get_neurons() const {
    return neurons;  // Retourne les neurones du génome
}

std::vector<neat::LinkGene> Genome::get_links() const {
    return links;  // Retourne les liens du génome
}

std::vector<neat::NeuronGene>& Genome::get_neurons() {
    return neurons;  // Retourne les neurones du génome
}

std::vector<neat::LinkGene>& Genome::get_links() {
    return links;  // Retourne les liens du génome
}

int Genome::generate_next_neuron_id() {
    int max_id = 0;
    for (const auto& neuron : neurons) {
        if (neuron.neuron_id > max_id) {
            max_id = neuron.neuron_id;
        }
    }
    return max_id + 1;
}

// Ajout des fonctions de gestion de neurones et liens
void Genome::add_neuron(const neat::NeuronGene &neuron) {
    neurons.push_back(neuron);
}

void Genome::add_link(const neat::LinkGene &link) {
    links.push_back(link);
}

// Recherche un neurone dans le génome par ID
std::optional<neat::NeuronGene> Genome::find_neuron(int neuron_id) const {
    for (const auto &neuron : neurons) {
        if (neuron.neuron_id == neuron_id) {
            return neuron;  // Retourne le neurone s'il est trouvé
        }
    }
    return std::nullopt;  // Retourne un optional vide si non trouvé
}

// Recherche un lien dans le génome par ID de lien
std::optional<neat::LinkGene> Genome::find_link(neat::LinkId link_id) const {
    for (const auto &link : links) {
        if (link.link_id.input_id == link_id.input_id && link.link_id.output_id == link_id.output_id) {
            return link;  // Retourne le lien s'il est trouvé
        }
    }
    return std::nullopt;  // Retourne un optional vide si non trouvé
}

// Génère un vecteur contenant les identifiants des nœuds d’entrée
std::vector<int> Genome::make_input_ids() const {
    std::vector<int> input_ids;
    for (int i = 0; i < num_inputs; i++) {
        input_ids.push_back(i);  // Ajoute les IDs des entrées
    }
    return input_ids;
}

// Génère un vecteur contenant les identifiants des nœuds de sortie
std::vector<int> Genome::make_output_ids() const {
    std::vector<int> output_ids;
    for (int i = 0; i < num_outputs; i++) {
        output_ids.push_back(num_inputs + i);  // Ajoute les IDs des sorties
    }
    return output_ids;
}
