#include <filesystem>
#include <iostream>
#include <string_view>
#include <system_error>

int main(int argc, char **argv) {
  if (argc == 4 && std::string_view(argv[1]) == "--copy") {
    std::error_code copy_error;
    if (!std::filesystem::copy_file(
            argv[2], argv[3], std::filesystem::copy_options::overwrite_existing,
            copy_error)) {
      std::cerr << "cannot copy pinned input: " << copy_error.message() << '\n';
      return 1;
    }
    return 0;
  }
  if (argc < 4) {
    std::cerr << "usage: builder BUSYBOX OUTPUT APPLET...\n";
    return 2;
  }
  const std::filesystem::path busybox = std::filesystem::absolute(argv[1]);
  const std::filesystem::path output = argv[2];
  std::error_code error;
  std::filesystem::create_directories(output, error);
  if (error) {
    std::cerr << "cannot create output directory: " << error.message() << '\n';
    return 1;
  }
  const std::filesystem::path relative =
      busybox.lexically_relative(std::filesystem::absolute(output));
  if (relative.empty()) {
    std::cerr << "cannot compute BusyBox relative path\n";
    return 1;
  }
  for (int index = 3; index < argc; ++index) {
    std::filesystem::create_symlink(relative, output / argv[index], error);
    if (error) {
      std::cerr << "cannot create " << argv[index] << ": " << error.message()
                << '\n';
      return 1;
    }
  }
  std::filesystem::create_symlink("/bin/bash", output / "bash", error);
  if (error) {
    std::cerr << "cannot expose execution-platform bash: " << error.message()
              << '\n';
    return 1;
  }
  return 0;
}
