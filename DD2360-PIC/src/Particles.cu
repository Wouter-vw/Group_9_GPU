#include <cuda.h>
#include <cuda_runtime.h>

#include "Alloc.h"
#include "Particles.h"

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part,
                       int is) {
  // set species ID
  part->species_ID = is;
  // number of particles
  part->nop = param->np[is];
  // maximum number of particles
  part->npmax = param->npMax[is];

  // choose a different number of mover iterations for ions and electrons
  if (param->qom[is] < 0) {
    // electrons
    part->NiterMover = param->NiterMover;
    part->n_sub_cycles = param->n_sub_cycles;
  } else {
    // ions: only one iteration
    part->NiterMover = 1;
    part->n_sub_cycles = 1;
  }

  // particles per cell
  part->npcelx = param->npcelx[is];
  part->npcely = param->npcely[is];
  part->npcelz = param->npcelz[is];
  part->npcel = part->npcelx * part->npcely * part->npcelz;

  // cast it to required precision
  part->qom = (FPpart)param->qom[is];

  long npmax = part->npmax;

  // initialize drift and thermal velocities
  // drift
  part->u0 = (FPpart)param->u0[is];
  part->v0 = (FPpart)param->v0[is];
  part->w0 = (FPpart)param->w0[is];
  // thermal
  part->uth = (FPpart)param->uth[is];
  part->vth = (FPpart)param->vth[is];
  part->wth = (FPpart)param->wth[is];

  //////////////////////////////
  /// ALLOCATION PARTICLE ARRAYS
  //////////////////////////////
  part->data = new Particle[npmax];
}

/** deallocate */
void particle_deallocate(struct particles* part) {
  // deallocate particle variables
  delete[] part->data;
}

int particleUpdate(int i, struct particles* part, struct EMfield* field,
                   struct grid* grd, struct parameters* param) {
  // print species and subcycling
  // std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " -
  // species " << part->species_ID << " ***" << std::endl;

  // auxiliary variables
  FPpart dt_sub_cycling = (FPpart)param->dt / ((double)part->n_sub_cycles);
  FPpart dto2 = .5 * dt_sub_cycling, qomdt2 = part->qom * dto2 / param->c;
  FPpart omdtsq, denom, ut, vt, wt, udotb;

  // local (to the particle) electric and magnetic field
  FPfield Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

  // interpolation densities
  int ix, iy, iz;
  FPfield weight[2][2][2];
  FPfield xi[2], eta[2], zeta[2];

  // intermediate particle position and velocity
  FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

  // start subcycling
  xptilde = part->data[i].x;
  yptilde = part->data[i].y;
  zptilde = part->data[i].z;
  // calculate the average velocity iteratively
  for (int innter = 0; innter < part->NiterMover; innter++) {
    // interpolation G-->P
    ix = 2 + int((part->data[i].x - grd->xStart) * grd->invdx);
    iy = 2 + int((part->data[i].y - grd->yStart) * grd->invdy);
    iz = 2 + int((part->data[i].z - grd->zStart) * grd->invdz);

    // calculate weights
    xi[0] = part->data[i].x - grd->XN[ix - 1][iy][iz];
    eta[0] = part->data[i].y - grd->YN[ix][iy - 1][iz];
    zeta[0] = part->data[i].z - grd->ZN[ix][iy][iz - 1];
    xi[1] = grd->XN[ix][iy][iz] - part->data[i].x;
    eta[1] = grd->YN[ix][iy][iz] - part->data[i].y;
    zeta[1] = grd->ZN[ix][iy][iz] - part->data[i].z;
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;

    // set to zero local electric and magnetic field
    Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++) {
          Exl += weight[ii][jj][kk] * field->Ex[ix - ii][iy - jj][iz - kk];
          Eyl += weight[ii][jj][kk] * field->Ey[ix - ii][iy - jj][iz - kk];
          Ezl += weight[ii][jj][kk] * field->Ez[ix - ii][iy - jj][iz - kk];
          Bxl += weight[ii][jj][kk] * field->Bxn[ix - ii][iy - jj][iz - kk];
          Byl += weight[ii][jj][kk] * field->Byn[ix - ii][iy - jj][iz - kk];
          Bzl += weight[ii][jj][kk] * field->Bzn[ix - ii][iy - jj][iz - kk];
        }

    // end interpolation
    omdtsq = qomdt2 * qomdt2 * (Bxl * Bxl + Byl * Byl + Bzl * Bzl);
    denom = 1.0 / (1.0 + omdtsq);
    // solve the position equation
    ut = part->data[i].u + qomdt2 * Exl;
    vt = part->data[i].v + qomdt2 * Eyl;
    wt = part->data[i].w + qomdt2 * Ezl;
    udotb = ut * Bxl + vt * Byl + wt * Bzl;
    // solve the velocity equation
    uptilde =
        (ut + qomdt2 * (vt * Bzl - wt * Byl + qomdt2 * udotb * Bxl)) * denom;
    vptilde =
        (vt + qomdt2 * (wt * Bxl - ut * Bzl + qomdt2 * udotb * Byl)) * denom;
    wptilde =
        (wt + qomdt2 * (ut * Byl - vt * Bxl + qomdt2 * udotb * Bzl)) * denom;
    // update position
    part->data[i].x = xptilde + uptilde * dto2;
    part->data[i].y = yptilde + vptilde * dto2;
    part->data[i].z = zptilde + wptilde * dto2;
  }  // end of iteration
  // update the final position and velocity
  part->data[i].u = 2.0 * uptilde - part->data[i].u;
  part->data[i].v = 2.0 * vptilde - part->data[i].v;
  part->data[i].w = 2.0 * wptilde - part->data[i].w;
  part->data[i].x = xptilde + uptilde * dt_sub_cycling;
  part->data[i].y = yptilde + vptilde * dt_sub_cycling;
  part->data[i].z = zptilde + wptilde * dt_sub_cycling;

  //////////
  //////////
  ////////// BC

  // X-DIRECTION: BC particles
  if (part->data[i].x > grd->Lx) {
    if (param->PERIODICX == true) {
      // PERIODIC
      part->data[i].x = part->data[i].x - grd->Lx;
    } else {
      // REFLECTING BC
      part->data[i].u = -part->data[i].u;
      part->data[i].x = 2 * grd->Lx - part->data[i].x;
    }
  }

  if (part->data[i].x < 0) {
    if (param->PERIODICX == true) {
      // PERIODIC
      part->data[i].x = part->data[i].x + grd->Lx;
    } else {
      // REFLECTING BC
      part->data[i].u = -part->data[i].u;
      part->data[i].x = -part->data[i].x;
    }
  }

  // Y-DIRECTION: BC particles
  if (part->data[i].y > grd->Ly) {
    if (param->PERIODICY == true) {
      // PERIODIC
      part->data[i].y = part->data[i].y - grd->Ly;
    } else {
      // REFLECTING BC
      part->data[i].v = -part->data[i].v;
      part->data[i].y = 2 * grd->Ly - part->data[i].y;
    }
  }

  if (part->data[i].y < 0) {
    if (param->PERIODICY == true) {
      // PERIODIC
      part->data[i].y = part->data[i].y + grd->Ly;
    } else {
      // REFLECTING BC
      part->data[i].v = -part->data[i].v;
      part->data[i].y = -part->data[i].y;
    }
  }

  // Z-DIRECTION: BC particles
  if (part->data[i].z > grd->Lz) {
    if (param->PERIODICZ == true) {
      // PERIODIC
      part->data[i].z = part->data[i].z - grd->Lz;
    } else {
      // REFLECTING BC
      part->data[i].w = -part->data[i].w;
      part->data[i].z = 2 * grd->Lz - part->data[i].z;
    }
  }

  if (part->data[i].z < 0) {
    if (param->PERIODICZ == true) {
      // PERIODIC
      part->data[i].z = part->data[i].z + grd->Lz;
    } else {
      // REFLECTING BC
      part->data[i].w = -part->data[i].w;
      part->data[i].z = -part->data[i].z;
    }
  }

  return (0);  // exit succcesfully
}  // end of the mover

/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd,
             struct parameters* param) {
  // print species and subcycling
  // std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " -
  // species " << part->species_ID << " ***" << std::endl;

  // start subcycling
  for (int i_sub = 0; i_sub < part->n_sub_cycles; i_sub++) {
    // move each particle with new fields
    for (int i = 0; i < part->nop; i++) {
      particleUpdate(i, part, field, grd, param);
    }  // end of subcycling
  }  // end of one particle

  return (0);  // exit succcesfully
}  // end of the mover

__global__ void mover_PC_kernel(struct particles* part, struct EMfield* field,
                                struct grid* grd, struct parameters* param) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  FPpart dt_sub_cycling = (FPpart)param->dt / ((double)part->n_sub_cycles);
  FPpart dto2 = .5 * dt_sub_cycling, qomdt2 = part->qom * dto2 / param->c;

  FPpart omdtsq, denom, ut, vt, wt, udotb;

  // local (to the particle) electric and magnetic field
  FPfield Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

  // interpolation densities
  int ix, iy, iz;
  FPfield weight[2][2][2];
  FPfield xi[2], eta[2], zeta[2];

  // intermediate particle position and velocity
  FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

  if (i < part->nop) {
    xptilde = part->data[i].x;
    yptilde = part->data[i].y;
    zptilde = part->data[i].z;

    for (int inner = 0; inner < part->NiterMover; inner++) {
      // Interpolation G-->P
      ix = 2 + int((part->data[i].x - grd->xStart) * grd->invdx);
      iy = 2 + int((part->data[i].y - grd->yStart) * grd->invdy);
      iz = 2 + int((part->data[i].z - grd->zStart) * grd->invdz);

      // Check indixing
      xi[0] =
          part->data[i].x -
          grd->XN_flat[(ix - 1) * (grd->nyn * grd->nzn) + (iy)*grd->nzn + (iz)];
      eta[0] =
          part->data[i].y -
          grd->YN_flat[ix * grd->nyn * grd->nzn + (iy - 1) * grd->nzn + (iz)];
      zeta[0] =
          part->data[i].z -
          grd->ZN_flat[ix * grd->nyn * grd->nzn + (iy)*grd->nzn + (iz - 1)];

      xi[1] = grd->XN_flat[ix * grd->nyn * grd->nzn + (iy)*grd->nzn + (iz)] -
              part->data[i].x;
      eta[1] = grd->YN_flat[ix * grd->nyn * grd->nzn + (iy)*grd->nzn + (iz)] -
               part->data[i].y;
      zeta[1] = grd->ZN_flat[ix * grd->nyn * grd->nzn + (iy)*grd->nzn + (iz)] -
                part->data[i].z;

      for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 2; jj++) {
          for (int kk = 0; kk < 2; kk++) {
            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
          }
        }
      }

      Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

      for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 2; jj++) {
          for (int kk = 0; kk < 2; kk++) {
            // Check idx index
            int idx = (ix - ii) * grd->nyn * grd->nzn + (iy - jj) * grd->nzn +
                      (iz - kk);
            Exl += weight[ii][jj][kk] * field->Ex_flat[idx];
            Eyl += weight[ii][jj][kk] * field->Ey_flat[idx];
            Ezl += weight[ii][jj][kk] * field->Ez_flat[idx];
            Bxl += weight[ii][jj][kk] * field->Bxn_flat[idx];
            Byl += weight[ii][jj][kk] * field->Byn_flat[idx];
            Bzl += weight[ii][jj][kk] * field->Bzn_flat[idx];
          }
        }
      }

      // Particle motion equations
      omdtsq = qomdt2 * qomdt2 * (Bxl * Bxl + Byl * Byl + Bzl * Bzl);
      denom = 1.0 / (1.0 + omdtsq);

      ut = part->data[i].u + qomdt2 * Exl;
      vt = part->data[i].v + qomdt2 * Eyl;
      wt = part->data[i].w + qomdt2 * Ezl;
      udotb = ut * Bxl + vt * Byl + wt * Bzl;

      uptilde =
          (ut + qomdt2 * (vt * Bzl - wt * Byl + qomdt2 * udotb * Bxl)) * denom;
      vptilde =
          (vt + qomdt2 * (wt * Bxl - ut * Bzl + qomdt2 * udotb * Byl)) * denom;
      wptilde =
          (wt + qomdt2 * (ut * Byl - vt * Bxl + qomdt2 * udotb * Bzl)) * denom;

      part->data[i].x = xptilde + uptilde * dto2;
      part->data[i].y = yptilde + vptilde * dto2;
      part->data[i].z = zptilde + wptilde * dto2;
    }

    // Update the final position and velocity
    part->data[i].u = 2.0 * uptilde - part->data[i].u;
    part->data[i].v = 2.0 * vptilde - part->data[i].v;
    part->data[i].w = 2.0 * wptilde - part->data[i].w;
    // update position
    part->data[i].x = xptilde + uptilde * dt_sub_cycling;
    part->data[i].y = yptilde + vptilde * dt_sub_cycling;
    part->data[i].z = zptilde + wptilde * dt_sub_cycling;

    // Boundary conditions
    if (part->data[i].x > grd->Lx) {
      if (grd->PERIODICX) {
        part->data[i].x -= grd->Lx;
      } else {
        part->data[i].u = -part->data[i].u;
        part->data[i].x = 2 * grd->Lx - part->data[i].x;
      }
    }
    if (part->data[i].x < 0) {
      if (grd->PERIODICX) {
        part->data[i].x += grd->Lx;
      } else {
        part->data[i].u = -part->data[i].u;
        part->data[i].x = -part->data[i].x;
      }
    }
    if (part->data[i].y > grd->Ly) {
      if (grd->PERIODICY) {
        part->data[i].y -= grd->Ly;
      } else {
        part->data[i].v = -part->data[i].v;
        part->data[i].y = 2 * grd->Ly - part->data[i].y;
      }
    }
    if (part->data[i].y < 0) {
      if (grd->PERIODICY) {
        part->data[i].y += grd->Ly;
      } else {
        part->data[i].v = -part->data[i].v;
        part->data[i].y = -part->data[i].y;
      }
    }
    if (part->data[i].z > grd->Lz) {
      if (grd->PERIODICZ) {
        part->data[i].z -= grd->Lz;
      } else {
        part->data[i].w = -part->data[i].w;
        part->data[i].z = 2 * grd->Lz - part->data[i].z;
      }
    }
    if (part->data[i].z < 0) {
      if (grd->PERIODICZ) {
        part->data[i].z += grd->Lz;
      } else {
        part->data[i].w = -part->data[i].w;
        part->data[i].z = -part->data[i].z;
      }
    }
  }
}

int mover_PC_GPU(struct particles* part, struct EMfield* field,
                 struct grid* grd, struct parameters* param) {
  // Size calculations
  int nxn = grd->nxn;
  int nyn = grd->nyn;
  int nzn = grd->nzn;
  int nop = part->nop;

  // 1. Allocate memory for structs on device
  particles* d_part;
  EMfield* d_field;
  grid* d_grd;
  parameters* d_param;

  cudaMalloc(&d_part, sizeof(particles));
  cudaMalloc(&d_field, sizeof(EMfield));
  cudaMalloc(&d_grd, sizeof(grid));
  cudaMalloc(&d_param, sizeof(parameters));

  // 2. Allocate memory for arrays on device
  Particle* d_data;

  cudaMalloc(&d_data, nop * sizeof(Particle));

  // Grid and field arrays
  FPpart *d_XN_flat, *d_YN_flat, *d_ZN_flat;
  FPfield *d_Ex_flat, *d_Ey_flat, *d_Ez_flat;
  FPfield *d_Bxn_flat, *d_Byn_flat, *d_Bzn_flat;

  cudaMalloc(&d_XN_flat, nxn * nyn * nzn * sizeof(FPpart));
  cudaMalloc(&d_YN_flat, nxn * nyn * nzn * sizeof(FPpart));
  cudaMalloc(&d_ZN_flat, nxn * nyn * nzn * sizeof(FPpart));
  cudaMalloc(&d_Ex_flat, nxn * nyn * nzn * sizeof(FPfield));
  cudaMalloc(&d_Ey_flat, nxn * nyn * nzn * sizeof(FPfield));
  cudaMalloc(&d_Ez_flat, nxn * nyn * nzn * sizeof(FPfield));
  cudaMalloc(&d_Bxn_flat, nxn * nyn * nzn * sizeof(FPfield));
  cudaMalloc(&d_Byn_flat, nxn * nyn * nzn * sizeof(FPfield));
  cudaMalloc(&d_Bzn_flat, nxn * nyn * nzn * sizeof(FPfield));

  // 3. Copy array data to device
  cudaMemcpy(d_data, part->data, nop * sizeof(Particle),
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_XN_flat, grd->XN_flat, nxn * nyn * nzn * sizeof(FPpart),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_YN_flat, grd->YN_flat, nxn * nyn * nzn * sizeof(FPpart),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_ZN_flat, grd->ZN_flat, nxn * nyn * nzn * sizeof(FPpart),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_Ex_flat, field->Ex_flat, nxn * nyn * nzn * sizeof(FPfield),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_Ey_flat, field->Ey_flat, nxn * nyn * nzn * sizeof(FPfield),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_Ez_flat, field->Ez_flat, nxn * nyn * nzn * sizeof(FPfield),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_Bxn_flat, field->Bxn_flat, nxn * nyn * nzn * sizeof(FPfield),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_Byn_flat, field->Byn_flat, nxn * nyn * nzn * sizeof(FPfield),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_Bzn_flat, field->Bzn_flat, nxn * nyn * nzn * sizeof(FPfield),
             cudaMemcpyHostToDevice);

  // 4. Create temporary structs with device pointers
  particles temp_part = *part;
  EMfield temp_field = *field;
  grid temp_grd = *grd;

  // 5. Update pointers in temporary structs to point to device memory
  temp_part.data = d_data;

  temp_field.Ex_flat = d_Ex_flat;
  temp_field.Ey_flat = d_Ey_flat;
  temp_field.Ez_flat = d_Ez_flat;
  temp_field.Bxn_flat = d_Bxn_flat;
  temp_field.Byn_flat = d_Byn_flat;
  temp_field.Bzn_flat = d_Bzn_flat;

  temp_grd.XN_flat = d_XN_flat;
  temp_grd.YN_flat = d_YN_flat;
  temp_grd.ZN_flat = d_ZN_flat;

  // 6. Copy the temporary structs with device pointers to device
  cudaMemcpy(d_part, &temp_part, sizeof(particles), cudaMemcpyHostToDevice);
  cudaMemcpy(d_field, &temp_field, sizeof(EMfield), cudaMemcpyHostToDevice);
  cudaMemcpy(d_grd, &temp_grd, sizeof(grid), cudaMemcpyHostToDevice);
  cudaMemcpy(d_param, param, sizeof(parameters), cudaMemcpyHostToDevice);

  // 7. Launch kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (part->nop + threadsPerBlock - 1) / threadsPerBlock;

  for (int i_sub = 0; i_sub < part->n_sub_cycles; i_sub++) {
    mover_PC_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_part, d_field, d_grd,
                                                        d_param);
    cudaDeviceSynchronize();
  }

  // 8. Copy results back to host
  cudaMemcpy(part->data, d_data, nop * sizeof(Particle),
             cudaMemcpyDeviceToHost);

  // 9. Free device memory
  cudaFree(d_data);
  cudaFree(d_XN_flat);
  cudaFree(d_YN_flat);
  cudaFree(d_ZN_flat);
  cudaFree(d_Ex_flat);
  cudaFree(d_Ey_flat);
  cudaFree(d_Ez_flat);
  cudaFree(d_Bxn_flat);
  cudaFree(d_Byn_flat);
  cudaFree(d_Bzn_flat);
  cudaFree(d_part);
  cudaFree(d_field);
  cudaFree(d_grd);
  cudaFree(d_param);

  return 0;
}

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids,
               struct grid* grd) {
  // arrays needed for interpolation
  FPpart weight[2][2][2];
  FPpart temp[2][2][2];
  FPpart xi[2], eta[2], zeta[2];

  // index of the cell
  int ix, iy, iz;

  for (register long long i = 0; i < part->nop; i++) {
    // determine cell: can we change to int()? is it faster?
    ix = 2 + int(floor((part->data[i].x - grd->xStart) * grd->invdx));
    iy = 2 + int(floor((part->data[i].y - grd->yStart) * grd->invdy));
    iz = 2 + int(floor((part->data[i].z - grd->zStart) * grd->invdz));

    // distances from node
    xi[0] = part->data[i].x - grd->XN[ix - 1][iy][iz];
    eta[0] = part->data[i].y - grd->YN[ix][iy - 1][iz];
    zeta[0] = part->data[i].z - grd->ZN[ix][iy][iz - 1];
    xi[1] = grd->XN[ix][iy][iz] - part->data[i].x;
    eta[1] = grd->YN[ix][iy][iz] - part->data[i].y;
    zeta[1] = grd->ZN[ix][iy][iz] - part->data[i].z;

    // calculate the weights for different nodes
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          weight[ii][jj][kk] =
              part->data[i].q * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;

    //////////////////////////
    // add charge density
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->rhon[ix - ii][iy - jj][iz - kk] +=
              weight[ii][jj][kk] * grd->invVOL;

    ////////////////////////////
    // add current density - Jx
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] = part->data[i].u * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    ////////////////////////////
    // add current density - Jy
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] = part->data[i].v * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    ////////////////////////////
    // add current density - Jz
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] = part->data[i].w * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    ////////////////////////////
    // add pressure pxx
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] =
              part->data[i].u * part->data[i].u * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    ////////////////////////////
    // add pressure pxy
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] =
              part->data[i].u * part->data[i].v * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    /////////////////////////////
    // add pressure pxz
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] =
              part->data[i].u * part->data[i].w * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    /////////////////////////////
    // add pressure pyy
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] =
              part->data[i].v * part->data[i].v * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    /////////////////////////////
    // add pressure pyz
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] =
              part->data[i].v * part->data[i].w * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    /////////////////////////////
    // add pressure pzz
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] =
              part->data[i].w * part->data[i].w * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->pzz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
  }
}

// **************************************
//    Tot. Simulation Time (s) = 17.2564
//    Mover Time / Cycle   (s) = 0.26256
//    Interp. Time / Cycle (s) = 1.15199
// ************************************
