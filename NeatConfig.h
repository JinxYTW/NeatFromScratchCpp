#ifndef NEATCONFIG_H
#define NEATCONFIG_H

struct NeatConfig {
    int population_size = 6;        // Taille de la population
    int num_inputs = 3;               // Nombre d'entrées
    int num_outputs = 2;              // Nombre de sorties

    // Probabilités de mutation
    double probability_add_link = 0.05;       // Probabilité d'ajouter un lien
    double probability_remove_link = 0.02;    // Probabilité de supprimer un lien
    double probability_add_neuron = 0.01;     // Probabilité d'ajouter un neurone
    double probability_remove_neuron = 0.01;  // Probabilité de supprimer un neurone
    double probability_mutate_link_weight = 0.8;  // Probabilité de muter le poids d'un lien
    double probability_mutate_neuron_bias = 0.7;     // Probabilité de muter le biais d'un neurone

    // Seuil de survie pour la sélection
    double survival_threshold = 0.3;  // Pourcentage d'individus qui survivent à chaque génération
};

#endif // NEATCONFIG_H
