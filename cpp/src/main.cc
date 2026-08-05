#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

#include "ohmnivore/simulator.h"
#include "ohmnivore/status.h"

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: ohmnivore <netlist.spice>\n";
    return 2;
  }

  std::filesystem::path netlist_path(argv[1]);
  std::ifstream input(netlist_path, std::ios::binary);
  if (!input && netlist_path.is_relative()) {
    if (const char *workspace = std::getenv("BUILD_WORKSPACE_DIRECTORY");
        workspace != nullptr) {
      netlist_path = std::filesystem::path(workspace) / netlist_path;
      input.clear();
      input.open(netlist_path, std::ios::binary);
    }
  }
  if (!input) {
    std::cerr << "io error: could not open " << argv[1] << '\n';
    return 1;
  }
  const std::string netlist((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
  if (!input.good() && !input.eof()) {
    std::cerr << "io error: failed while reading " << argv[1] << '\n';
    return 1;
  }

  auto result = ohmnivore::SimulateToCsv(netlist);
  if (!result.ok()) {
    std::cerr << ohmnivore::ErrorCodeName(result.error().code)
              << " error: " << result.error().message << '\n';
    return 1;
  }
  std::cout << result.value();
  return 0;
}
