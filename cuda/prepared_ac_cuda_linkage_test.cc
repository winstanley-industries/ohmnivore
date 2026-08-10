#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Elf64Header {
  std::array<unsigned char, 16> identity;
  std::uint16_t type;
  std::uint16_t machine;
  std::uint32_t version;
  std::uint64_t entry;
  std::uint64_t program_header_offset;
  std::uint64_t section_header_offset;
  std::uint32_t flags;
  std::uint16_t header_size;
  std::uint16_t program_header_entry_size;
  std::uint16_t program_header_count;
  std::uint16_t section_header_entry_size;
  std::uint16_t section_header_count;
  std::uint16_t section_name_index;
};

struct Elf64SectionHeader {
  std::uint32_t name;
  std::uint32_t type;
  std::uint64_t flags;
  std::uint64_t address;
  std::uint64_t offset;
  std::uint64_t size;
  std::uint32_t link;
  std::uint32_t info;
  std::uint64_t address_alignment;
  std::uint64_t entry_size;
};

struct Elf64Dynamic {
  std::int64_t tag;
  std::uint64_t value;
};

struct Elf64Symbol {
  std::uint32_t name;
  unsigned char info;
  unsigned char other;
  std::uint16_t section_index;
  std::uint64_t value;
  std::uint64_t size;
};

static_assert(sizeof(Elf64Header) == 64);
static_assert(sizeof(Elf64SectionHeader) == 64);
static_assert(sizeof(Elf64Dynamic) == 16);
static_assert(sizeof(Elf64Symbol) == 24);

[[nodiscard]] std::string ReadFile(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot read GPU-02 audit input: " + path);
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

template <typename T>
[[nodiscard]] T ReadObject(const std::string &binary, std::size_t offset) {
  if (offset > binary.size() || sizeof(T) > binary.size() - offset) {
    throw std::runtime_error("ELF object extends beyond GPU-02 binary");
  }
  T object{};
  std::memcpy(&object, binary.data() + offset, sizeof(T));
  return object;
}

[[nodiscard]] std::string ReadElfString(const std::string &binary,
                                        const Elf64SectionHeader &strings,
                                        std::uint64_t string_offset) {
  if (strings.offset > binary.size() ||
      strings.size > binary.size() - strings.offset ||
      string_offset >= strings.size) {
    throw std::runtime_error("invalid ELF string-table offset");
  }
  const std::size_t begin =
      static_cast<std::size_t>(strings.offset + string_offset);
  const std::size_t limit =
      static_cast<std::size_t>(strings.offset + strings.size);
  const std::size_t end = binary.find('\0', begin);
  if (end == std::string::npos || end >= limit) {
    throw std::runtime_error("unterminated ELF string-table entry");
  }
  return binary.substr(begin, end - begin);
}

[[nodiscard]] std::vector<Elf64SectionHeader>
ReadSectionHeaders(const std::string &binary, const Elf64Header &header) {
  if (header.section_header_entry_size != sizeof(Elf64SectionHeader)) {
    throw std::runtime_error("unexpected ELF section-header size");
  }
  if (header.section_header_offset > binary.size() ||
      static_cast<std::size_t>(header.section_header_count) >
          (binary.size() - header.section_header_offset) /
              sizeof(Elf64SectionHeader)) {
    throw std::runtime_error("ELF section table extends beyond GPU-02 binary");
  }

  std::vector<Elf64SectionHeader> sections;
  sections.reserve(header.section_header_count);
  for (std::size_t index = 0; index < header.section_header_count; ++index) {
    const std::size_t offset =
        static_cast<std::size_t>(header.section_header_offset) +
        index * sizeof(Elf64SectionHeader);
    sections.push_back(ReadObject<Elf64SectionHeader>(binary, offset));
  }
  return sections;
}

void AuditElf(const std::string &binary) {
  constexpr std::uint32_t kSectionSymbolTable = 2;
  constexpr std::uint32_t kSectionDynamic = 6;
  constexpr std::uint32_t kSectionDynamicSymbols = 11;
  constexpr std::int64_t kDynamicNull = 0;
  constexpr std::int64_t kDynamicNeeded = 1;
  constexpr std::uint16_t kUndefinedSection = 0;

  const Elf64Header header = ReadObject<Elf64Header>(binary, 0);
  if (header.identity[0] != 0x7f || header.identity[1] != 'E' ||
      header.identity[2] != 'L' || header.identity[3] != 'F' ||
      header.identity[4] != 2 || header.identity[5] != 1) {
    throw std::runtime_error(
        "GPU-02 benchmark is not little-endian ELF64 as required");
  }
  const std::vector<Elf64SectionHeader> sections =
      ReadSectionHeaders(binary, header);

  std::set<std::string> needed_libraries;
  for (const Elf64SectionHeader &section : sections) {
    if (section.type != kSectionDynamic) {
      continue;
    }
    if (section.link >= sections.size() ||
        section.entry_size != sizeof(Elf64Dynamic) ||
        section.size % sizeof(Elf64Dynamic) != 0 ||
        section.offset > binary.size() ||
        section.size > binary.size() - section.offset) {
      throw std::runtime_error("invalid ELF dynamic section");
    }
    const Elf64SectionHeader &strings = sections[section.link];
    for (std::size_t offset = 0; offset < section.size;
         offset += sizeof(Elf64Dynamic)) {
      const Elf64Dynamic dynamic = ReadObject<Elf64Dynamic>(
          binary, static_cast<std::size_t>(section.offset) + offset);
      if (dynamic.tag == kDynamicNull) {
        break;
      }
      if (dynamic.tag == kDynamicNeeded) {
        needed_libraries.insert(ReadElfString(binary, strings, dynamic.value));
      }
    }
  }
  if (needed_libraries.empty()) {
    throw std::runtime_error("GPU-02 benchmark has no auditable DT_NEEDED");
  }
  const std::set<std::string> allowed_host_abi_libraries{
      "ld-linux-x86-64.so.2", "libc.so.6",  "libdl.so.2", "libm.so.6",
      "libpthread.so.0",      "librt.so.1",
  };
  for (const std::string &needed : needed_libraries) {
    if (!allowed_host_abi_libraries.contains(needed)) {
      throw std::runtime_error(
          "GPU-02 ELF has undeclared dynamic dependency: " + needed);
    }
  }
  if (!needed_libraries.contains("libc.so.6")) {
    throw std::runtime_error("GPU-02 ELF is missing its declared glibc ABI");
  }

  bool found_embedded_cudss_create = false;
  bool found_embedded_cublas_create = false;
  for (const Elf64SectionHeader &section : sections) {
    if (section.type != kSectionSymbolTable &&
        section.type != kSectionDynamicSymbols) {
      continue;
    }
    if (section.link >= sections.size() ||
        section.entry_size != sizeof(Elf64Symbol) ||
        section.size % sizeof(Elf64Symbol) != 0 ||
        section.offset > binary.size() ||
        section.size > binary.size() - section.offset) {
      throw std::runtime_error("invalid ELF symbol table");
    }
    const Elf64SectionHeader &strings = sections[section.link];
    for (std::size_t offset = 0; offset < section.size;
         offset += sizeof(Elf64Symbol)) {
      const Elf64Symbol symbol = ReadObject<Elf64Symbol>(
          binary, static_cast<std::size_t>(section.offset) + offset);
      if (symbol.name != 0 && symbol.section_index != kUndefinedSection) {
        const std::string name = ReadElfString(binary, strings, symbol.name);
        found_embedded_cudss_create |= name == "cudssCreate";
        found_embedded_cublas_create |= name == "cublasCreate_v2";
      }
    }
  }
  if (!found_embedded_cudss_create) {
    throw std::runtime_error(
        "GPU-02 ELF is missing the embedded static cudssCreate definition");
  }
  if (!found_embedded_cublas_create) {
    throw std::runtime_error(
        "GPU-02 ELF is missing the embedded static cublasCreate_v2 "
        "definition");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) {
      throw std::runtime_error(
          "GPU-02 linkage test requires exactly the evidence benchmark ELF");
    }
    AuditElf(ReadFile(argv[1]));
    std::cout << "GPU-02 linkage passed: cuDSS and cuBLAS are embedded "
                 "statically; CUDA and GCC runtimes are static; DT_NEEDED "
                 "contains only the declared glibc host ABI\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "GPU-02 linkage failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
