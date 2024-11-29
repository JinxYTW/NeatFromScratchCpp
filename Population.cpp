#include "Population.h"
#include "Mutator.h"
#include "RNG.h"
#include "ComputeFitness.h"
#include "Neat.h"
#include "Genome.h"
#include <iostream>
#include <memory>



Population::Population(NeatConfig config, RNG &rng) 
    : config{config}, rng{rng}, next_genome_id{0} {
    for (int i = 0; i < config.population_size; ++i) {
        int num_hidden_neurons = rng.next_int(1, 4);  // Random hidden neurons
std::shared_ptr<Genome> genome = std::make_shared<Genome>(Genome::create_genome(generate_next_genome_id(), config.num_inputs, config.num_outputs, num_hidden_neurons, rng));
individuals.emplace_back(genome);

    }
}

std::vector<neat::Individual>& Population::get_individuals() {
    return individuals;
}

int Population::generate_next_genome_id() {
    return next_genome_id++;
}

void Population::mutate(Genome &genome) {
    Mutator::mutate(genome, config, rng);
}

std::vector<neat::Individual> Population::reproduce() {
    auto old_members = sort_individuals_by_fitness(individuals);
    int reproduction_cutoff = std::ceil(config.survival_threshold * old_members.size());
    std::vector<neat::Individual> new_generation;

    std::cout << "Reproducing..." << std::endl;

    while (new_generation.size() < config.population_size) {
        neat::Individual& p1 = rng.choose_random(old_members, reproduction_cutoff);
        neat::Individual& p2 = rng.choose_random(old_members, reproduction_cutoff);

        std::cout << "Crossover between " << p1.genome->get_genome_id() << " and " << p2.genome->get_genome_id() << std::endl;

        neat::Neat neat_instance;
        Genome offspring = neat_instance.crossover(p1.genome, p2.genome, generate_next_genome_id());

        std::cout << "Offspring genome ID: " << offspring.get_genome_id() << std::endl;

        mutate(offspring);

        

        new_generation.push_back(neat::Individual(std::make_shared<Genome>(offspring)));

    }

    return new_generation;
}




std::vector<neat::Individual> Population::reproduce_from_genomes(const std::vector<std::shared_ptr<Genome>>& genomes) {
    if (genomes.empty()) {
        throw std::runtime_error("Erreur : La liste de génomes est vide. Impossible de reproduire.");
    }

    // Initialiser une instance de ComputeFitness
    ComputeFitness compute_fitness(rng);

    // Trier les génomes par fitness évaluée (sans stocker la fitness dans les objets Genome)
    std::vector<std::shared_ptr<Genome>> sorted_genomes = genomes;
    std::sort(sorted_genomes.begin(), sorted_genomes.end(),
        [&compute_fitness](const std::shared_ptr<Genome>& a, const std::shared_ptr<Genome>& b) {
            // Évaluer la fitness pour comparer les génomes
            return compute_fitness(*a, /* ant_id */ 0) > compute_fitness(*b, /* ant_id */ 0);
        });

    int reproduction_cutoff = std::ceil(config.survival_threshold * sorted_genomes.size());
    std::vector<neat::Individual> new_generation;

    std::cout << "Reproducing from custom genome list..." << std::endl;

    // Boucle pour créer la nouvelle génération
    while (new_generation.size() < config.population_size) {
        const std::shared_ptr<Genome>& p1 = rng.choose_random(sorted_genomes, reproduction_cutoff);
        const std::shared_ptr<Genome>& p2 = rng.choose_random(sorted_genomes, reproduction_cutoff);

        std::cout << "Crossover between " << p1->get_genome_id() << " and " << p2->get_genome_id() << std::endl;

        neat::Neat neat_instance;
        Genome offspring_genome = neat_instance.alt_crossover(p1, p2, generate_next_genome_id());
        std::shared_ptr<Genome> offspring = std::make_shared<Genome>(offspring_genome);

        std::cout << "Offspring genome ID: " << offspring->get_genome_id() << std::endl;

        mutate(*offspring);

        new_generation.push_back(neat::Individual(offspring));
    }

    return new_generation;
}




std::vector<neat::Individual> Population::sort_individuals_by_fitness(const std::vector<neat::Individual>& individuals) {
    std::vector<neat::Individual> sorted_individuals = individuals;
    std::sort(sorted_individuals.begin(), sorted_individuals.end(), 
        [](const neat::Individual& a, const neat::Individual& b) {
            return a.fitness > b.fitness;
        });
    return sorted_individuals;
}

void Population::update_best() {
    auto best_it = std::max_element(individuals.begin(), individuals.end(), 
        [](const neat::Individual& a, const neat::Individual& b) {
            return a.fitness < b.fitness;
        });
    if (best_it != individuals.end()) {
        best_individual = *best_it;
    }
}

void Population::replace_population(std::vector<neat::Individual> new_generation) {
    if (new_generation.empty()) {
        throw std::runtime_error("Erreur : La nouvelle génération est vide. Impossible de remplacer la population.");
    }

    // Remplace les individus actuels par ceux de la nouvelle génération
    individuals = std::move(new_generation);

    // Met à jour le meilleur individu avec la nouvelle population
    update_best();

    std::cout << "Population remplacée. Taille actuelle : " << individuals.size() << std::endl;
}





