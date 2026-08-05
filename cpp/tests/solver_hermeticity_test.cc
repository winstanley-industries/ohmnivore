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

#include "klu.h"

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
    throw std::runtime_error("cannot read audit input: " + path);
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

template <typename T>
[[nodiscard]] T ReadObject(const std::string &binary, std::size_t offset) {
  if (offset > binary.size() || sizeof(T) > binary.size() - offset) {
    throw std::runtime_error("ELF object extends beyond production binary");
  }
  T object{};
  std::memcpy(&object, binary.data() + offset, sizeof(T));
  return object;
}

[[nodiscard]] std::string ReadElfString(const std::string &binary,
                                        const Elf64SectionHeader &strings,
                                        std::uint32_t string_offset) {
  if (string_offset >= strings.size || strings.offset > binary.size() ||
      strings.size > binary.size() - strings.offset) {
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

  const Elf64Header header = ReadObject<Elf64Header>(binary, 0);
  if (header.identity[0] != 0x7f || header.identity[1] != 'E' ||
      header.identity[2] != 'L' || header.identity[3] != 'F' ||
      header.identity[4] != 2 || header.identity[5] != 1) {
    throw std::runtime_error(
        "production binary is not little-endian ELF64 as required");
  }
  const std::vector<Elf64SectionHeader> sections =
      ReadSectionHeaders(binary, header);

  const std::set<std::string> allowed_needed = {
      "ld-linux-x86-64.so.2", "libc.so.6",      "libdl.so.2", "libm.so.6",
      "libpthread.so.0",      "libresolv.so.2", "librt.so.1",
  };
  bool found_dynamic = false;
  for (const Elf64SectionHeader &section : sections) {
    if (section.type != kSectionDynamic) {
      continue;
    }
    found_dynamic = true;
    if (section.link >= sections.size() ||
        section.entry_size != sizeof(Elf64Dynamic) ||
        section.size % sizeof(Elf64Dynamic) != 0) {
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
        const std::string needed = ReadElfString(
            binary, strings, static_cast<std::uint32_t>(dynamic.value));
        if (!allowed_needed.contains(needed)) {
          throw std::runtime_error("production ELF has forbidden DT_NEEDED: " +
                                   needed);
        }
      }
    }
  }
  if (!found_dynamic) {
    throw std::runtime_error("production ELF has no auditable dynamic section");
  }

  std::set<std::string> symbols;
  for (const Elf64SectionHeader &section : sections) {
    if (section.type != kSectionSymbolTable &&
        section.type != kSectionDynamicSymbols) {
      continue;
    }
    if (section.link >= sections.size() ||
        section.entry_size != sizeof(Elf64Symbol) ||
        section.size % sizeof(Elf64Symbol) != 0) {
      throw std::runtime_error("invalid ELF symbol table");
    }
    const Elf64SectionHeader &strings = sections[section.link];
    for (std::size_t offset = 0; offset < section.size;
         offset += sizeof(Elf64Symbol)) {
      const Elf64Symbol symbol = ReadObject<Elf64Symbol>(
          binary, static_cast<std::size_t>(section.offset) + offset);
      if (symbol.name != 0) {
        symbols.insert(ReadElfString(binary, strings, symbol.name));
      }
    }
  }
  for (const std::string_view required :
       {"klu_factor", "klu_refactor", "klu_z_factor", "klu_z_refactor"}) {
    if (!symbols.contains(std::string(required))) {
      throw std::runtime_error("production ELF is missing embedded symbol: " +
                               std::string(required));
    }
  }
  if (symbols.contains("ohmnivore_dense_oracle_test_only_marker")) {
    throw std::runtime_error("production ELF contains dense-oracle symbol");
  }
}

void RequireContains(const std::string &text, std::string_view marker,
                     std::string_view context) {
  if (text.find(marker) == std::string::npos) {
    throw std::runtime_error(std::string(context) +
                             " is missing marker: " + std::string(marker));
  }
}

void RequireAbsent(const std::string &text, std::string_view marker,
                   std::string_view context) {
  if (text.find(marker) != std::string::npos) {
    throw std::runtime_error(
        std::string(context) +
        " contains forbidden marker: " + std::string(marker));
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 4) {
      throw std::runtime_error(
          "solver hermeticity test requires ELF, dependency, and CcInfo "
          "manifests");
    }
    static_assert(KLU_MAIN_VERSION == 2);
    static_assert(KLU_SUB_VERSION == 3);
    static_assert(KLU_SUBSUB_VERSION == 6);

    const std::string binary = ReadFile(argv[1]);
    AuditElf(binary);
    RequireContains(binary, "KLU numeric factorization", "production ELF");
    RequireAbsent(binary, "OHMNIVORE_DENSE_ORACLE_TEST_ONLY_v1",
                  "production ELF");

    const std::string dependencies = ReadFile(argv[2]);
    RequireContains(dependencies, "suitesparse_7_12_3//:klu",
                    "production dependency manifest");
    for (const std::string_view forbidden :
         {"//cpp:dense_oracle", "/amd_l", "/btf_l", "/colamd_l.c", "/klu_l",
          "/klu_zl"}) {
      RequireAbsent(dependencies, forbidden, "production dependency manifest");
    }

    const std::string cc_manifest = ReadFile(argv[3]);
    RequireContains(cc_manifest, "status=zero_forbidden_entries",
                    "production CcInfo manifest");
    RequireContains(cc_manifest, "KLU/Include/klu.h",
                    "production CcInfo manifest");
    RequireContains(cc_manifest, "libklu.a", "production CcInfo manifest");

    std::cout << "production solver boundary passed: KLU 2.3.6 symbols are "
                 "embedded; ELF DT_NEEDED is ABI-only; dependency and CcInfo "
                 "manifests exclude dense, long-index, system numerical, and "
                 "system include/library inputs\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "solver hermeticity failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
