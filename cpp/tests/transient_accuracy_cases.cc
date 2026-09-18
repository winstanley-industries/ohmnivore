#include "cpp/tests/transient_accuracy_cases.h"

#include <cmath>
#include <complex>
#include <iomanip>
#include <sstream>
#include <utility>

namespace ohmnivore::test {
namespace {
std::string Number(double x) {
  std::ostringstream text;
  text << std::setprecision(17) << x;
  return text.str();
}

std::string Wave(double amplitude, double tau) {
  return "DC 0 PWL(0 0 " + Number(.1 * tau) + " " + Number(amplitude) + " " +
         Number(10 * tau) + " " + Number(amplitude) + " " + Number(10.1 * tau) +
         " " + Number(-amplitude) + " " + Number(30 * tau) + " " +
         Number(-amplitude) + " " + Number(30.1 * tau) + " 0 " +
         Number(40 * tau) + " 0)\n";
}

constexpr std::array<double, 6> kCorners{0, .1, 10, 10.1, 30, 30.1};
constexpr std::array<double, 6> kSlopes{10, -10, -20, 20, 10, -10};
} // namespace

std::vector<TransientAccuracyCase> TransientAccuracyCases() {
  std::vector<TransientAccuracyCase> cases;
  for (const auto &[amplitude, tau, bias] :
       {std::array<double, 3>{1e-4, 1e-6, 0},
        {1e-4, 1e-6, 400},
        {1, 1e-9, 0},
        {1, 1e-3, 0},
        {1, 1, 0},
        {100, 1e-3, 0}}) {
    cases.push_back({
        .name = "rc_a" + Number(amplitude) + "_t" + Number(tau) + "_b" +
                Number(bias),
        .deck = "Vreference ref 0 " + Number(bias) + "\nVdrive drive ref " +
                Wave(amplitude, tau) + "Rseries drive out 1\nCstore out ref " +
                Number(tau) + "\nBzero out ref I={0}\n",
        .analysis = {.5 * tau, 40 * tau, 0, false},
        .bias = bias,
        .voltage_limit = 2e-4 * amplitude + 2e-7,
        .current_limit = 0,
        .has_inductor = false,
        .exact =
            [=](double time) {
              const double t = time / tau;
              double v = 0;
              for (std::size_t i = 0; i < kCorners.size(); ++i) {
                const double u = t - kCorners[i];
                if (u > 0)
                  v += amplitude * kSlopes[i] * (u + std::expm1(-u));
              }
              return std::array<double, 2>{v, 0};
            },
    });
  }
  for (const double resistance : {.02, .5, 20.0}) {
    for (const auto &[amplitude, tau, bias] :
         {std::array<double, 3>{1e-3, 1e-6, 0},
          {1, 1e-6, 0},
          {1, 1e-3, 400},
          {100, 1e-3, 0}}) {
      const std::complex<double> discriminant =
          std::sqrt(std::complex<double>(resistance * resistance - 4, 0));
      const auto a = (-resistance + discriminant) / 2.0;
      const auto b = (-resistance - discriminant) / 2.0;
      cases.push_back({
          .name = "rlc_r" + Number(resistance) + "_a" + Number(amplitude) +
                  "_t" + Number(tau) + "_b" + Number(bias),
          .deck = "Vreference ref 0 " + Number(bias) + "\nVdrive drive ref " +
                  Wave(amplitude, tau) + "Rseries drive series " +
                  Number(resistance) + "\nLstore series out " + Number(tau) +
                  "\nCstore out ref " + Number(tau) + "\nBzero out ref I={0}\n",
          .analysis = {.2 * tau, 40 * tau, 0, false},
          .bias = bias,
          .voltage_limit = .005 * amplitude + 2e-6,
          .current_limit = .005 * amplitude + 2e-8,
          .has_inductor = true,
          .exact =
              [=](double time) {
                double v = 0, current = 0;
                for (std::size_t i = 0; i < kCorners.size(); ++i) {
                  const double u = time / tau - kCorners[i];
                  if (u <= 0)
                    continue;
                  const auto ea = std::exp(a * u), eb = std::exp(b * u);
                  const auto ramp =
                      u + (b * (ea - 1.0) / a - a * (eb - 1.0) / b) / (a - b);
                  const auto step = 1.0 + (b * ea - a * eb) / (a - b);
                  v += amplitude * kSlopes[i] * ramp.real();
                  current += amplitude * kSlopes[i] * step.real();
                }
                return std::array<double, 2>{v, current};
              },
      });
    }
  }
  for (const double amplitude : {1e-3, 1.0, 100.0}) {
    constexpr double tau = 1e-6;
    cases.push_back({
        .name = "nonlinear_charge_a" + Number(amplitude),
        .deck = "Icharge 0 out " + Wave(amplitude, tau) + "Cstore out sense " +
                Number(tau) + "\nVsense sense 0 0\nBextra out 0 I={i(Vsense)*" +
                Number(3 / (amplitude * amplitude)) + "*v(out)*v(out)}\n",
        .analysis = {.5 * tau, 40 * tau, 0, false},
        .bias = 0,
        .voltage_limit = .002 * amplitude + 2e-7,
        .current_limit = 0,
        .has_inductor = false,
        .exact =
            [=](double time) {
              double charge = 0;
              for (std::size_t i = 0; i < kCorners.size(); ++i) {
                const double u = time / tau - kCorners[i];
                if (u > 0)
                  charge += .5 * kSlopes[i] * u * u;
              }
              // Invert the independently integrated constitutive law q=y+y^3.
              const double v =
                  2 / std::sqrt(3.0) *
                  std::sinh(std::asinh(1.5 * std::sqrt(3.0) * charge) / 3);
              return std::array<double, 2>{amplitude * v, 0};
            },
    });
  }
  return cases;
}
} // namespace ohmnivore::test
