#include "Population.h"
#include "ComputeFitness.h"
#include "NeuralNetwork.h"
#include "Utils.h"
#include "NeatConfig.h"
#include <iostream>
#include <vector>
#include <numeric>  // Pour std::accumulate
#include <fstream>  // Pour écrire les données dans un fichier CSV


int main() {
    RNG rng;
    NeatConfig config;
    Population population(config, rng);

    // Conteneur pour la fitness moyenne par génération
    std::vector<double> average_fitness_per_generation;

    // Affichage initial des génomes
    for (const auto &individual : population.get_individuals()) {
        const Genome &genome = *individual.genome;
        std::cout << "Génome ID: " << genome.get_genome_id() << std::endl;

        std::cout << "Neurones:" << std::endl;
        for (const auto &neuron : genome.get_neurons()) {
            std::cout << "  Neuron ID: " << neuron.neuron_id 
                      << ", Biais: " << static_cast<int>(neuron.bias) 
                      << std::endl;
        }

        std::cout << "Liens:" << std::endl;
        for (const auto &link : genome.get_links()) {
            std::cout << "  Link from " << link.link_id.input_id 
                      << " to " << link.link_id.output_id 
                      << " with weight " << link.weight << std::endl;
        }
    }

    // Créer l'objet de calcul de fitness
    ComputeFitness compute_fitness(rng);

    const int num_generations = 10;
    const int num_rounds = 10;  // Nombre de rounds pour chaque simulation
    const int num_ants = 10;    // Nombre d'individus dans chaque génération

  

    for (int generation = 0; generation < num_generations; ++generation) {
        std::cout << "Génération " << generation + 1 << " : " << std::endl;

        // Initialisation des fitness
        for (auto &individual : population.get_individuals()) {
            individual.fitness = 0.0;
        }

        // Simulation pour chaque individu
        for (int ant_id = 0; ant_id < num_ants; ++ant_id) {
            for (auto &individual : population.get_individuals()) {
                // 1. Créer un réseau neuronal pour cet individu
                FeedForwardNeuralNetwork network = FeedForwardNeuralNetwork::create_from_genome(*individual.genome);

                // 2. Simuler les rounds de pierre-papier-ciseaux
                for (int round = 0; round < num_rounds; ++round) {
                    // 3. Obtenir l'état de jeu (choix aléatoire de l'adversaire)
                    std::vector<double> game_state = get_game_state_rpc(ant_id, rng);

                    // 4. Activer le réseau avec l'état de jeu (choix de l'individu)
                    std::vector<double> actions = network.activate(game_state);

                    // 5. Exécuter l'action du réseau
                    perform_action_rpc(actions, ant_id);

                    // 6. Évaluer la fitness de l'individu pour ce round
                    individual.fitness += compute_fitness.evaluate_rpc(*individual.genome, ant_id);
                }
            }
        }

        // Calcul de la fitness moyenne
        double total_fitness = 0.0;
        for (const auto &individual : population.get_individuals()) {
            total_fitness += individual.fitness;
        }
        double average_fitness = total_fitness / population.get_individuals().size();
        average_fitness_per_generation.push_back(average_fitness);

        //std::cout << "Fitness moyenne de la génération : " << average_fitness << std::endl;

     

        // Sauvegarde des génomes pour suivi
        int individual_index = 0;
        for (const auto &individual : population.get_individuals()) {
            save(*individual.genome, "genome_saves/genome_generation_" 
                                     + std::to_string(generation) 
                                     + "_individual_" 
                                     + std::to_string(individual_index++) + ".txt");
        }

        // Mise à jour du meilleur individu
        population.update_best();

        // Génération de la nouvelle population
        auto new_generation = population.reproduce();
        if (new_generation.empty()) {
            throw std::runtime_error("Erreur : La génération produite est vide !");
        }

        population.replace_population(std::move(new_generation));

        // Affiche la meilleure fitness de la génération
        std::cout << "Meilleure fitness de la génération : " 
                  << population.get_individuals().front().fitness << std::endl;

        // Sauvegarde du meilleur génome
        save(*population.get_individuals().front().genome, "best_genome_generation_" + std::to_string(generation + 1) + ".txt");
    }

    // Exporter les données de fitness moyenne dans un fichier CSV
    std::ofstream file("fitness_moyenne.csv");
    if (file.is_open()) {
        for (size_t i = 0; i < average_fitness_per_generation.size(); ++i) {
            file << i + 1 << "," << average_fitness_per_generation[i] << "\n";
        }
        file.close();
        std::cout << "Les données de fitness moyenne ont été exportées dans 'fitness_moyenne.csv'.\n";
    }

  

    return 0;
}