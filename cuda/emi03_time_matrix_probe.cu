#include "cuda/emi03_time_probe_common.cuh"
int main(int argc, char **argv) {
  try {
    if (argc < 4 || argc > 5 ||
        (argc == 5 && std::string(argv[4]) != "--validate-only"))
      return 2;
    std::cout << std::setprecision(17);
    return ohmnivore::TimeProbe(argv[1], argv[2], argv[3], argc == 5);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
