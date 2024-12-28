#include "Timing.h"

double cpuSecond() {
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  auto microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(duration);
  return microseconds.count() * 1.0e-6;
}