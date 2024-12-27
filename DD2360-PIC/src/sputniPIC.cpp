/** A mixed-precision implicit Particle-in-Cell simulator for heterogeneous systems **/

// Allocator for 2D, 3D and 4D array: chain of pointers
#include "Alloc.h"

// Precision: fix precision for different quantities
#include "PrecisionTypes.h"
// Simulation Parameter - structure
#include "Parameters.h"
// Grid structure
#include "Grid.h"
// Interpolated Quantities Structures
#include "InterpDensSpecies.h"
#include "InterpDensNet.h"

// Field structure
#include "EMfield.h" // Just E and Bn
#include "EMfield_aux.h" // Bc, Phi, Eth, D

// Particles structure
#include "Particles.h"
#include "Particles_aux.h" // Needed only if dointerpolation on GPU - avoid reduction on GPU

// Initial Condition
#include "IC.h"
// Boundary Conditions
#include "BC.h"
// timing
#include "Timing.h"
// Read and output operations
#include "RW_IO.h"


int main(int argc, char **argv){
    
    // Read the inputfile and fill the param structure
    parameters param;
    // Read the input file name from command line
    readInputFile(&param,argc,argv);
    printParameters(&param);
    saveParameters(&param);

    double test = 1.0;

    std::cout << test << std::endl;
    
    // Timing variables
    // double iStart = cpuSecond();
    double iMover, iInterp = 0.0,  eInterp= 0.0; //eMover = 0.0,
    double iMoverGPU = 0.0;
    double iMoverCPU = 0.0;
    double iInterp_CPU = 0.0;
    double iInterp_GPU = 0.0;

    // Set-up the grid information
    grid grd;
    setGrid(&param, &grd);
    
    // Allocate Fields
    EMfield field;
    field_allocate(&grd,&field);
    EMfield_aux field_aux;
    field_aux_allocate(&grd,&field_aux);
    
    
    // Allocate Interpolated Quantities
    // per species
    interpDensSpecies *ids = new interpDensSpecies[param.ns];
    for (int is=0; is < param.ns; is++)
        interp_dens_species_allocate(&grd,&ids[is],is);
    // Net densities
    interpDensNet idn;
    interp_dens_net_allocate(&grd,&idn);
    
    // Allocate Particles
    particles *part = new particles[param.ns];
    // allocation
    for (int is=0; is < param.ns; is++){
        particle_allocate(&param,&part[is],is);
    }
    
    // Initialization
    initGEM(&param,&grd,&field,&field_aux,part,ids);
    
    std::cout << "Running GPU Simulation..." << std::endl;

    
    // **********************************************************//
    // **** Start the Simulation!  Cycle index start from 1  *** //
    // **********************************************************//
    for (int cycle = param.first_cycle_n; cycle < (param.first_cycle_n + param.ncycles); cycle++) {
        
        // std::cout << std::endl;
        // std::cout << "***********************" << std::endl;
        // std::cout << "   cycle = " << cycle << std::endl;
        // std::cout << "***********************" << std::endl;
    
        // set to zero the densities - needed for interpolation
        setZeroDensities(&idn,ids,&grd,param.ns);
        
        
        
        // implicit mover
        iMover = cpuSecond(); // start timer for mover
        for (int is=0; is < param.ns; is++)
            mover_PC_GPU(&part[is],&field,&grd,&param);
        iMoverGPU += (cpuSecond() - iMover); // stop timer for mover
        
        
        
        
        // interpolation particle to grid
        iInterp = cpuSecond(); // start timer for the interpolation step
        // interpolate species
        for (int is=0; is < param.ns; is++)
            interpP2G_GPU(&part[is],&ids[is],&grd);
          
        iInterp_GPU = cpuSecond() - iInterp;

        // apply BC to interpolated densities
        for (int is=0; is < param.ns; is++)
            applyBCids(&ids[is],&grd,&param);
        // sum over species
        sumOverSpecies(&idn,ids,&grd,param.ns);
        // interpolate charge density from center to node
        applyBCscalarDensN(idn.rhon,&grd,&param);
        
        
        
        // write E, B, rho to disk
        if (cycle%param.FieldOutputCycle==0){
            VTK_Write_Vectors(cycle, &grd,&field);
            VTK_Write_Scalars(cycle, &grd,ids,&idn);
        }
        
        eInterp += (cpuSecond() - iInterp); // stop timer for interpolation
        
        
    
    }  // end of one PIC cycle


    // Set-up the grid information
    grid grd_CPU;
    setGrid(&param, &grd_CPU);
    
    // Allocate Fields
    EMfield field_CPU;
    field_allocate(&grd_CPU,&field_CPU);
    EMfield_aux field_aux_CPU;
    field_aux_allocate(&grd_CPU,&field_aux_CPU);
    
    
    // Allocate Interpolated Quantities
    // per species
    interpDensSpecies *ids_CPU = new interpDensSpecies[param.ns];
    for (int is=0; is < param.ns; is++)
        interp_dens_species_allocate(&grd_CPU,&ids_CPU[is],is);
    // Net densities
    interpDensNet idn_CPU;
    interp_dens_net_allocate(&grd_CPU,&idn_CPU);
    
    // Allocate Particles
    particles *part_CPU = new particles[param.ns];
    // allocation
    for (int is=0; is < param.ns; is++){
        particle_allocate(&param,&part_CPU[is],is);
    }
    
    // Initialization
    initGEM(&param,&grd_CPU,&field_CPU,&field_aux_CPU,part_CPU,ids_CPU);

    std::cout << "Running CPU Simulation..." << std::endl;

    
    // **********************************************************//
    // **** Start the Simulation!  Cycle index start from 1  *** //
    // **********************************************************//
    for (int cycle = param.first_cycle_n; cycle < (param.first_cycle_n + param.ncycles); cycle++) {
        
        // std::cout << std::endl;
        // std::cout << "***********************" << std::endl;
        // std::cout << "   cycle = " << cycle << std::endl;
        // std::cout << "***********************" << std::endl;
    
        // set to zero the densities - needed for interpolation
        setZeroDensities(&idn_CPU,ids_CPU,&grd_CPU,param.ns);
        
        
        
        // implicit mover
        iMover = cpuSecond(); // start timer for mover
        for (int is=0; is < param.ns; is++)
            mover_PC(&part_CPU[is],&field_CPU,&grd_CPU,&param);
        iMoverCPU += (cpuSecond() - iMover); // stop timer for mover
        
        
        
        
        // interpolation particle to grid
        iInterp = cpuSecond(); // start timer for the interpolation step
        // interpolate species
        for (int is=0; is < param.ns; is++)
            interpP2G(&part_CPU[is],&ids_CPU[is],&grd_CPU);
        // apply BC to interpolated densities
        iInterp_CPU = cpuSecond() - iInterp;
        for (int is=0; is < param.ns; is++)
            applyBCids(&ids_CPU[is],&grd_CPU,&param);
        // sum over species
        sumOverSpecies(&idn_CPU,ids_CPU,&grd_CPU,param.ns);
        // interpolate charge density from center to node
        applyBCscalarDensN(idn_CPU.rhon,&grd_CPU,&param);
        
        
        
        // write E, B, rho to disk
        if (cycle%param.FieldOutputCycle==0){
            VTK_Write_Vectors(cycle, &grd_CPU,&field_CPU);
            VTK_Write_Scalars(cycle, &grd_CPU,ids_CPU,&idn_CPU);
        }
        
        eInterp += (cpuSecond() - iInterp); // stop timer for interpolation
        
        
    
    }  // end of one PIC cycle


    // Compare results
    FPpart* x_GPU = part->x;
    FPpart* y_GPU = part->y;
    FPpart* z_GPU = part->z;
    FPpart* x_CPU = part_CPU->x;
    FPpart* y_CPU = part_CPU->y;
    FPpart* z_CPU = part_CPU->z;
    FPpart*** rhon_GPU = ids->rhon;
    FPpart*** Jx_GPU = ids->Jx;
    FPpart*** Jy_GPU = ids->Jy;
    FPpart*** Jz_GPU = ids->Jz;
    FPpart*** pxx_GPU = ids->pxx;
    FPpart*** pxy_GPU = ids->pxy;
    FPpart*** pxz_GPU = ids->pxz;
    FPpart*** pyy_GPU = ids->pyy;
    FPpart*** pyz_GPU = ids->pyz;
    FPpart*** pzz_GPU = ids->pzz;


    FPpart*** rhon_CPU = ids_CPU->rhon;
    FPpart*** Jx_CPU = ids_CPU->Jx;
    FPpart*** Jy_CPU = ids_CPU->Jy;
    FPpart*** Jz_CPU = ids_CPU->Jz;
    FPpart*** pxx_CPU = ids_CPU->pxx;
    FPpart*** pxy_CPU = ids_CPU->pxy;
    FPpart*** pxz_CPU = ids_CPU->pxz;
    FPpart*** pyy_CPU = ids_CPU->pyy;
    FPpart*** pyz_CPU = ids_CPU->pyz;
    FPpart*** pzz_CPU = ids_CPU->pzz;



    int nxn = grd.nxn; 
    int nyn = grd.nyn;
    int nzn = grd.nzn;
    int nop = part->nop;

    FPpart xptilde_GPU, yptilde_GPU, zptilde_GPU;
    FPpart xptilde_CPU, yptilde_CPU, zptilde_CPU;

    FPpart tol = 1e-5;

    bool CHECK = true;

    int counter = 0;

    for (int i = 0; i<nop; i++) {
      xptilde_CPU = x_CPU[i];
      yptilde_CPU = y_CPU[i];
      zptilde_CPU = z_CPU[i];
      xptilde_GPU = x_GPU[i];
      yptilde_GPU = y_GPU[i];
      zptilde_GPU = z_GPU[i];
      // std::cout << "CPU: " << xptilde_CPU  << std::endl;
      // std::cout << xptilde_GPU  << std::endl;


      if (fabs(xptilde_CPU - xptilde_GPU) > tol ||
        fabs(yptilde_CPU - yptilde_GPU) > tol ||
        fabs(zptilde_CPU - zptilde_GPU) > tol) {
        CHECK = false;
        // break;
        counter += 1;
        // std::cout << "CPU: " << xptilde_CPU  << std::endl;
        // std::cout << "CPU: " << yptilde_CPU  << std::endl;
        // std::cout << "CPU: " << zptilde_CPU  << std::endl;
        // std::cout << "GPU: " << xptilde_GPU  << std::endl;
        // std::cout << "GPU: " << yptilde_GPU  << std::endl;
        // std::cout << "GPU: " << zptilde_GPU  << std::endl;

    }

  }


  std::cout << "x, y, z: " << counter << std::endl;
  tol = 1e-6;
  counter = 0;
  for (int i = 0; i < nxn; i++) {
    for (int j=0; j < nyn; j++) {
      for (int k=0; k < nzn; k++) {

        if (fabs(rhon_CPU[i][j][k] - rhon_GPU[i][j][k]) > tol) {
          std::cout << "rhon CPU: " << rhon_CPU[i][j][k]  << std::endl;
          std::cout << "rhon GPU: " << rhon_GPU[i][j][k]  << std::endl;

         
          counter += 1;
        }
        if (fabs(Jx_CPU[i][j][k] - Jx_GPU[i][j][k]) > tol ) {
          counter += 1;
          std::cout << "Jx CPU: " << Jx_CPU[i][j][k]  << std::endl;
          std::cout << "Jx GPU: " << Jx_GPU[i][j][k]  << std::endl;
        }
        if (fabs(Jy_CPU[i][j][k] - Jy_GPU[i][j][k])  > tol ) {
          counter += 1;

          std::cout << "Jy CPU: " << Jy_CPU[i][j][k]  << std::endl;
          std::cout << "Jy GPU: " << Jy_GPU[i][j][k]  << std::endl;
        }
        if (fabs(Jz_CPU[i][j][k] - Jz_GPU[i][j][k])  > tol) {
          counter += 1;

          std::cout << "Jz CPU: " << Jz_CPU[i][j][k]  << std::endl;
          std::cout << "Jz GPU: " << Jz_GPU[i][j][k]  << std::endl;
        }
        if (fabs(pxx_CPU[i][j][k] - pxx_GPU[i][j][k])  > tol) {
          counter += 1;

          std::cout << "pxx CPU: " << pxx_CPU[i][j][k]  << std::endl;
          std::cout << "pxx GPU: " << pxx_GPU[i][j][k]  << std::endl;
        }
        if (fabs(pxy_CPU[i][j][k] - pxy_GPU[i][j][k])  > tol ) {
          counter += 1;

          std::cout << "pxy CPU: " << pxy_CPU[i][j][k]  << std::endl;
          std::cout << "pxy GPU: " << pxy_GPU[i][j][k]  << std::endl;
        }
        if (fabs(pxz_CPU[i][j][k] - pxz_GPU[i][j][k]) > tol) {
          counter += 1;

          std::cout << "pxz CPU: " << pxz_CPU[i][j][k]  << std::endl;
          std::cout << "pxz GPU: " << pxz_GPU[i][j][k]  << std::endl;
        }
        if (fabs(pyy_CPU[i][j][k] - pyy_GPU[i][j][k]) > tol) {
          counter += 1;

          std::cout << "pyy CPU: " << pyy_CPU[i][j][k]  << std::endl;
          std::cout << "pyy GPU: " << pyy_GPU[i][j][k]  << std::endl;
        }
        if (fabs(pyz_CPU[i][j][k] - pyz_GPU[i][j][k]) > tol) {
          counter += 1;

          std::cout << "pzy CPU: " << pyz_CPU[i][j][k]  << std::endl;
          std::cout << "pzy GPU: " << pyz_GPU[i][j][k]  << std::endl;
        }
        if (fabs(pzz_CPU[i][j][k] - pzz_GPU[i][j][k]) > tol) {
          counter += 1;

          std::cout << "pzz CPU: " << pzz_CPU[i][j][k]  << std::endl;
          std::cout << "pzz GPU: " << pzz_GPU[i][j][k]  << std::endl;
        }
      }
    }
  }

  std::cout << "All the other: " << counter << std::endl;


  if (CHECK) {
    std::cout << "True result " << std::endl;
  }
  else {
    std::cout << "False result " << std::endl;
  }

    
    
    /// Release the resources
    // deallocate field
    grid_deallocate(&grd);
    field_deallocate(&grd,&field);
    // interp
    interp_dens_net_deallocate(&grd,&idn);
    
    // Deallocate interpolated densities and particles
    for (int is=0; is < param.ns; is++){
        interp_dens_species_deallocate(&grd,&ids[is]);
        particle_deallocate(&part[is]);
    }

    /// Release the resources
    // deallocate field
    grid_deallocate(&grd_CPU);
    field_deallocate(&grd_CPU,&field_CPU);
    // interp
    interp_dens_net_deallocate(&grd_CPU,&idn_CPU);
    
    // Deallocate interpolated densities and particles
    for (int is=0; is < param.ns; is++){
        interp_dens_species_deallocate(&grd_CPU,&ids_CPU[is]);
        particle_deallocate(&part_CPU[is]);
    }
    
    
    // stop timer
    // double iElaps = cpuSecond() - iStart;
    
    // Print timing of simulation
    // std::cout << std::endl;
    // std::cout << "**************************************" << std::endl;
    // std::cout << "   Tot. Simulation Time (s) = " << iElaps << std::endl;
    // std::cout << "   Mover Time / Cycle   (s) = " << eMover/param.ncycles << std::endl;
    // std::cout << "   Interp. Time / Cycle (s) = " << eInterp/param.ncycles  << std::endl;
    // std::cout << "**************************************" << std::endl;

    // Print results
    std::cout << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "   GPU Mover Time / Cycle   (s) = " << iMoverGPU / param.ncycles << std::endl;
    std::cout << "   CPU Mover Time / Cycle   (s) = " << iMoverCPU / param.ncycles << std::endl;
    // std::cout << "   Interp. Time / Cycle     (s) = " << iInterp / param.ncycles << std::endl;
    std::cout << "**************************************" << std::endl;

    std::cout << "**************************************" << std::endl;
    std::cout << "   GPU inter Time / Cycle   (s) = " << iInterp_GPU / param.ncycles << std::endl;
    std::cout << "   CPU inter Time / Cycle   (s) = " << iInterp_CPU / param.ncycles << std::endl;
    // std::cout << "   Interp. Time / Cycle     (s) = " << iInterp / param.ncycles << std::endl;
    std::cout << "**************************************" << std::endl;
    


    
    // exit
    return 0;
}


