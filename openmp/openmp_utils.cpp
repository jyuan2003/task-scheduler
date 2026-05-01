#include "openmp_scheduler.h"
#include <algorithm>
#include <omp.h>
#include <random>
#include <iostream>


void initialize(Scheduler &scheduler) {
    std::random_device rd;
    std::mt19937 gen(rd());  
    std::uniform_int_distribution<int> dist(0, scheduler.num_colors - 1);
    for (std::size_t i = 0; i < scheduler.colors.size(); i++) {
        scheduler.colors[i] = dist(gen);
    }
    omp_init_lock(&(scheduler.lock));
}

void report_program_stats(Scheduler &scheduler) {
    double span = scheduler.max_runtime;
    std::cout << "Runtime span:" << " " << span << std::endl;
}