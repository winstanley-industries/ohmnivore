#include "cuda/emi03_worker_pool.h"

#include <poll.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <streambuf>
#include <thread>

namespace {
thread_local FILE *diagnostic = nullptr;

// Preserve the existing runner's typed diagnostics on each owner's private
// inherited log. No process-wide file descriptor is redirected during a job.
class Diagnostics final : public std::streambuf {
public:
  Diagnostics() : previous_(std::cerr.rdbuf(this)) {}
  ~Diagnostics() override { std::cerr.rdbuf(previous_); }

protected:
  std::streamsize xsputn(const char *text, std::streamsize size) override {
    if (diagnostic)
      return static_cast<std::streamsize>(
          std::fwrite(text, 1, static_cast<std::size_t>(size), diagnostic));
    std::lock_guard lock(mutex_);
    return previous_->sputn(text, size);
  }
  int_type overflow(int_type value) override {
    if (traits_type::eq_int_type(value, traits_type::eof()))
      return traits_type::not_eof(value);
    const char byte = traits_type::to_char_type(value);
    return xsputn(&byte, 1) == 1 ? value : traits_type::eof();
  }
  int sync() override {
    if (diagnostic)
      return std::fflush(diagnostic);
    std::lock_guard lock(mutex_);
    return previous_->pubsync();
  }

private:
  std::streambuf *previous_;
  std::mutex mutex_;
};

std::string JsonString(const std::string &value) {
  std::string result = "\"";
  constexpr char hex[] = "0123456789abcdef";
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') {
      result += '\\';
      result += static_cast<char>(c);
    } else if (c < 32) {
      result += "\\u00";
      result += hex[c >> 4];
      result += hex[c & 15];
    } else {
      result += static_cast<char>(c);
    }
  }
  return result + '"';
}

bool Reply(FILE *output, const std::string &text) {
  return std::fwrite(text.data(), 1, text.size(), output) == text.size() &&
         std::fflush(output) == 0;
}

using File = std::unique_ptr<FILE, decltype(&std::fclose)>;
struct Channel {
  File input{nullptr, std::fclose}, output{nullptr, std::fclose},
      log{nullptr, std::fclose};
  int core = 0;
};

struct Paths {
  std::mutex mutex;
  std::set<std::filesystem::path> active;
};

int Serve(Channel &channel, int owner, const std::string &program,
          int (*execute)(std::vector<std::string>), Paths &paths,
          std::stop_token stop) {
  diagnostic = channel.log.get();
  struct Reset {
    ~Reset() { diagnostic = nullptr; }
  } reset;
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  CPU_SET(channel.core, &affinity);
  if (sched_setaffinity(0, sizeof(affinity), &affinity) != 0)
    return 2;
  const auto tid = static_cast<long>(syscall(SYS_gettid));
  if (!Reply(channel.output.get(),
             "{\"status\":\"ready\",\"owner\":" + std::to_string(owner) +
                 ",\"thread_id\":" + std::to_string(tid) + "}\n"))
    return 1;
  while (true) {
    std::string line;
    int byte;
    while (true) {
      if (stop.stop_requested())
        return 2;
      pollfd pending{fileno(channel.input.get()), POLLIN, 0};
      const int ready = poll(&pending, 1, 20);
      if (ready < 0 && errno != EINTR)
        return 1;
      if (ready <= 0)
        continue;
      byte = std::fgetc(channel.input.get());
      if (byte == EOF || byte == '\n')
        break;
      if (line.size() == 16 * 1024)
        return 2;
      line += static_cast<char>(byte);
    }
    if (byte == EOF)
      return line.empty() && !std::ferror(channel.input.get()) ? 0 : 2;
    const auto first = line.find('\t');
    const auto second = first == std::string::npos ? std::string::npos
                                                   : line.find('\t', first + 1);
    const std::string input = line.substr(0, first);
    int code = 2;
    if (first != std::string::npos && second != std::string::npos &&
        line.find('\t', second + 1) == std::string::npos &&
        line.find('\0') == std::string::npos) {
      std::vector<std::string> arguments{
          program, input, line.substr(first + 1, second - first - 1),
          line.substr(second + 1)};
      std::set<std::filesystem::path> owned;
      bool valid = true;
      for (std::size_t i = 1; i < arguments.size(); ++i) {
        valid &= std::filesystem::path(arguments[i]).is_absolute();
        owned.insert(std::filesystem::weakly_canonical(arguments[i]));
      }
      valid &= owned.size() == 3;
      for (const auto &suffix : {".partial", ".gpu.json", ".gpu.json.partial"})
        owned.insert(std::filesystem::weakly_canonical(arguments[3] + suffix));
      owned.insert(
          std::filesystem::weakly_canonical(arguments[2] + ".partial"));
      valid &= owned.size() == 7;
      {
        std::lock_guard lock(paths.mutex);
        for (const auto &path : owned)
          valid &= !paths.active.contains(path);
        if (valid)
          paths.active.insert(owned.begin(), owned.end());
      }
      if (valid) {
        code = execute(std::move(arguments));
        std::lock_guard lock(paths.mutex);
        for (const auto &path : owned)
          paths.active.erase(path);
      } else {
        std::cerr
            << "invalid-structure: conflicting or nonabsolute pool job paths\n";
      }
    }
    if (!Reply(channel.output.get(),
               "{\"status\":\"" +
                   std::string(code == 0 ? "complete" : "typed_failure") +
                   "\",\"input\":" + JsonString(input) +
                   ",\"owner\":" + std::to_string(owner) +
                   ",\"thread_id\":" + std::to_string(tid) +
                   ",\"exit_code\":" + std::to_string(code) + "}\n"))
      return 1;
  }
}
} // namespace

int RunEmi03WorkerPool(int argc, char **argv,
                       int (*execute)(std::vector<std::string>)) {
  if (argc < 3 || argc > 18)
    return 2;
  std::vector<Channel> channels;
  std::set<int> descriptors, cores;
  for (int i = 2; i < argc; ++i) {
    std::array<int, 4> values{};
    const std::string text(argv[i]);
    const char *begin = text.data(), *end = begin + text.size();
    for (int j = 0; j < 4; ++j) {
      auto parsed = std::from_chars(begin, end, values[j]);
      if (parsed.ec != std::errc{} || values[j] < (j == 3 ? 0 : 3) ||
          (j != 3 && (parsed.ptr == end || *parsed.ptr != ':')) ||
          (j == 3 && (parsed.ptr != end || values[j] >= CPU_SETSIZE)))
        return 2;
      begin = parsed.ptr + (j != 3);
    }
    for (int j = 0; j < 3; ++j)
      if (!descriptors.insert(values[j]).second)
        return 2;
    if (!cores.insert(values[3]).second)
      return 2;
    Channel channel;
    channel.input.reset(fdopen(values[0], "r"));
    channel.output.reset(fdopen(values[1], "w"));
    channel.log.reset(fdopen(values[2], "w"));
    channel.core = values[3];
    if (!channel.input || !channel.output || !channel.log)
      return 2;
    if (setvbuf(channel.input.get(), nullptr, _IONBF, 0) != 0)
      return 2;
    channels.push_back(std::move(channel));
  }
  Diagnostics diagnostics;
  Paths paths;
  std::vector<int> results(channels.size(), 2);
  std::vector<std::jthread> owners;
  for (std::size_t i = 0; i < channels.size(); ++i)
    owners.emplace_back([&, i](std::stop_token stop) {
      try {
        results[i] = Serve(channels[i], static_cast<int>(i), argv[0], execute,
                           paths, stop);
      } catch (...) {
        results[i] = 1;
      }
      channels[i].input.reset();
      channels[i].output.reset();
      channels[i].log.reset();
    });
  for (auto &owner : owners)
    owner.join();
  owners.clear();
  for (int code : results)
    if (code != 0)
      return code;
  return 0;
}
