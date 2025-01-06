/** A mixed-precision implicit Particle-in-Cell simulator for heterogeneous
 * systems **/

// Allocator for 2D, 3D and 4D array: chain of pointers
#include "Alloc.h"

// Precision: fix precision for different quantities
#include "PrecisionTypes.h"
// Simulation Parameter - structure
#include "Parameters.h"
// Grid structure
#include "Grid.h"
// Interpolated Quantities Structures
#include "InterpDensNet.h"
#include "InterpDensSpecies.h"

// Field structure
#include "EMfield.h"      // Just E and Bn
#include "EMfield_aux.h"  // Bc, Phi, Eth, D

// Particles structure
#include "Particles.h"
#include "Particles_aux.h"  // Needed only if dointerpolation on GPU - avoid reduction on GPU

// Initial Condition
#include "IC.h"
// Boundary Conditions
#include "BC.h"
// timing
#include "Timing.h"
// Read and output operations
#include <fstream>
#include <memory>
#include <optional>
#include <vector>

#include "RW_IO.h"

struct SimulationResult {
  std::unique_ptr<Particle[]> data;
  int size;
  double elaps;
};

SimulationResult runSimulation(parameters &param, bool useGPU) {
  NVTX3_FUNC_RANGE();
  // Timing variables
  double iStart = cpuSecond();

  double iMover, iInterp, eMover = 0.0, eInterp = 0.0;

  nvtxRangePushA("allocations");

  grid grd;
  setGrid(&param, &grd);

  // Allocate Fields
  EMfield field;
  field_allocate(&grd, &field);
  EMfield_aux field_aux;
  field_aux_allocate(&grd, &field_aux);

  // Allocate Interpolated Quantities
  // per species
  interpDensSpecies *ids = new interpDensSpecies[param.ns];
  for (int is = 0; is < param.ns; is++)
    interp_dens_species_allocate(&grd, &ids[is], is);
  // Net densities
  interpDensNet idn;
  interp_dens_net_allocate(&grd, &idn);

  // Allocate Particles
  particles *part = new particles[param.ns];
  // allocation
  for (int is = 0; is < param.ns; is++) {
    particle_allocate(&param, &part[is], is);
  }
  nvtxRangePop();

  // Initialization
  initGEM(&param, &grd, &field, &field_aux, part, ids);

  // **********************************************************//
  // **** Start the Simulation!  Cycle index start from 1  *** //
  // **********************************************************//
  {
    auto rId = useGPU ? "simulation_GPU" : "simulation_CPU";
    nvtx3::scoped_range r{rId};
    for (int cycle = param.first_cycle_n;
         cycle < (param.first_cycle_n + param.ncycles); cycle++) {
      std::cout << std::endl;
      std::cout << "***********************" << std::endl;
      std::cout << "   cycle = " << cycle << std::endl;
      std::cout << "***********************" << std::endl;

      // set to zero the densities - needed for interpolation
      setZeroDensities(&idn, ids, &grd, param.ns);

      // implicit mover
      iMover = cpuSecond();  // start timer for mover
      {
        nvtx3::scoped_range moverRange{useGPU ? "moverPC_GPU_cycle"
                                              : "moverPC_CPU_cycle"};
        if (useGPU) {
          // mover_PC_GPU(part, &field, &grd, &param);
          for (int is = 0; is < param.ns; is++)
            mover_PC(&part[is], &field, &grd, &param);
        } else {
          for (int is = 0; is < param.ns; is++)
            mover_PC(&part[is], &field, &grd, &param);
        }
      }

      eMover += (cpuSecond() - iMover);  // stop timer for mover

      // interpolation particle to grid
      iInterp = cpuSecond();  // start timer for the interpolation step
      // interpolate species
      {
        nvtx3::scoped_range interpRange{useGPU ? "interpP2G_GPU_cycle"
                                               : "interpP2G_CPU_cycle"};
        for (int is = 0; is < param.ns; is++) {
          if (useGPU)
            interpP2G_GPU(&part[is], &ids[is], &grd);
          else
            interpP2G(&part[is], &ids[is], &grd);
        }
      }
      // apply BC to interpolated densities
      for (int is = 0; is < param.ns; is++) applyBCids(&ids[is], &grd, &param);
      // sum over species
      sumOverSpecies(&idn, ids, &grd, param.ns);
      // interpolate charge density from center to node
      applyBCscalarDensN(idn.rhon, &grd, &param);

      // write E, B, rho to disk
      if (cycle % param.FieldOutputCycle == 0) {
        nvtx3::scoped_range ioScope{"write_fields"};
        VTK_Write_Vectors(cycle, &grd, &field);
        VTK_Write_Scalars(cycle, &grd, ids, &idn);
      }

      eInterp += (cpuSecond() - iInterp);  // stop timer for interpolation
    }  // end of one PIC cycle
  }
  Particle *data = new Particle[part->nop];
  for (int i = 0; i < part->nop; i++) {
    data[i] = part->data[i];
  }
  SimulationResult res;
  res.data = std::unique_ptr<Particle[]>(data);
  res.size = part->nop;

  {
    nvtx3::scoped_range r{"deallocations"};
    /// Release the resources
    // deallocate field
    grid_deallocate(&grd);
    field_deallocate(&grd, &field);
    // interp
    interp_dens_net_deallocate(&grd, &idn);

    // Deallocate interpolated densities and particles
    for (int is = 0; is < param.ns; is++) {
      interp_dens_species_deallocate(&grd, &ids[is]);
      particle_deallocate(&part[is]);
    }
  }
  double iElaps = cpuSecond() - iStart;

  std::cout << std::endl;
  std::cout << "**************************************" << std::endl;
  std::cout << "   Tot. Simulation Time (s) = " << iElaps << std::endl;
  std::cout << "   Mover Time / Cycle   (s) = " << eMover / param.ncycles
            << std::endl;
  std::cout << "   Interp. Time / Cycle (s) = " << eInterp / param.ncycles
            << std::endl;
  std::cout << "**************************************" << std::endl;

  res.elaps = iElaps;
  return res;
}

double vecDiff(Vec3<FPpart> &v1, Vec3<FPpart> &v2) {
  auto diff = v1 - v2;
  auto sums = abs(v1) + abs(v2);
  auto deltaX = abs(diff.x) / sums.x;
  auto deltaY = abs(diff.y) / sums.y;
  auto deltaZ = abs(diff.z) / sums.z;
  return max(deltaX, max(deltaY, deltaZ));
}

int main(int argc, char **argv) {
  // Read the inputfile and fill the param structure
  parameters param;
  // Read the input file name from command line
  {
    nvtx3::scoped_range r{"input_read"};
    readInputFile(&param, argc, argv);
    printParameters(&param);
    saveParameters(&param);
  }

  bool runCpu = false;
  bool runGpu = true;
  auto oCpuResults = std::optional<SimulationResult>{};
  auto oGpuResults = std::optional<SimulationResult>{};
  if (runCpu) {
    std::cout << "running CPU simulation" << std::endl;
    oCpuResults = std::optional<SimulationResult>{runSimulation(param, false)};
  }
  if (runGpu) {
    std::cout << "running GPU simulation" << std::endl;
    oGpuResults = std::optional<SimulationResult>{runSimulation(param, true)};
  }

  if (runCpu && runGpu) {
    double maxDelta = 0.0, meanDelta = 0.0;
    // Open a file to write delta values
    std::ofstream deltaFile("delta_values.txt");

    SimulationResult &cpuResults = oCpuResults.value();
    SimulationResult &gpuResults = oGpuResults.value();

    for (int i = 0; i < gpuResults.size; i++) {
      auto cpuVec = Vec3<FPpart>(cpuResults.data[i].x, cpuResults.data[i].y,
                                 cpuResults.data[i].z);
      auto gpuVec = Vec3<FPpart>(gpuResults.data[i].x, gpuResults.data[i].y,
                                 gpuResults.data[i].z);
      auto delta = vecDiff(cpuVec, gpuVec);

      meanDelta += delta;
      maxDelta = max(delta, maxDelta);

      // Write the delta value to the file
      deltaFile << delta << std::endl;
    }
    meanDelta /= gpuResults.size;

    // Close the file
    deltaFile.close();
    std::cout << "Max delta: " << maxDelta << ", Mean delta: " << meanDelta
              << std::endl;
  }
}
