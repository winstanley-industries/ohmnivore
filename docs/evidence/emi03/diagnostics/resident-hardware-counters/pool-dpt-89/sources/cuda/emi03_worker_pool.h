#ifndef OHMNIVORE_CUDA_EMI03_WORKER_POOL_H_
#define OHMNIVORE_CUDA_EMI03_WORKER_POOL_H_

#include <string>
#include <vector>

// Private resident-worker entry point. Each inherited channel owns one pinned
// host thread and one complete job at a time in the process's CUDA context.
int RunEmi03WorkerPool(int argc, char **argv,
                       int (*execute)(std::vector<std::string>));

#endif
