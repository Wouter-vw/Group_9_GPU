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
    xi[0] = part->data[i].x - grd->nodes[ix - 1][iy][iz].x;
    eta[0] = part->data[i].y - grd->nodes[ix][iy - 1][iz].y;
    zeta[0] = part->data[i].z - grd->nodes[ix][iy][iz - 1].z;
    xi[1] = grd->nodes[ix][iy][iz].x - part->data[i].x;
    eta[1] = grd->nodes[ix][iy][iz].y - part->data[i].y;
    zeta[1] = grd->nodes[ix][iy][iz].z - part->data[i].z;
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;

    // set to zero local electric and magnetic field
    Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++) {
          Exl += weight[ii][jj][kk] *
                 field->electricField[ix - ii][iy - jj][iz - kk].x;
          Eyl += weight[ii][jj][kk] *
                 field->electricField[ix - ii][iy - jj][iz - kk].y;
          Ezl += weight[ii][jj][kk] *
                 field->electricField[ix - ii][iy - jj][iz - kk].z;
          Bxl += weight[ii][jj][kk] *
                 field->magneticField[ix - ii][iy - jj][iz - kk].x;
          Byl += weight[ii][jj][kk] *
                 field->magneticField[ix - ii][iy - jj][iz - kk].y;
          Bzl += weight[ii][jj][kk] *
                 field->magneticField[ix - ii][iy - jj][iz - kk].z;
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
  if (i >= part->nop) return;

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

  for (int i_sub = 0; i_sub < part->n_sub_cycles; i_sub++) {
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
          part->data[i].x - grd->nodes_flat[(ix - 1) * (grd->nyn * grd->nzn) +
                                            (iy)*grd->nzn + (iz)]
                                .x;
      eta[0] =
          part->data[i].y -
          grd->nodes_flat[ix * grd->nyn * grd->nzn + (iy - 1) * grd->nzn + (iz)]
              .y;
      zeta[0] =
          part->data[i].z -
          grd->nodes_flat[ix * grd->nyn * grd->nzn + (iy)*grd->nzn + (iz - 1)]
              .z;

      xi[1] =
          grd->nodes_flat[ix * grd->nyn * grd->nzn + (iy)*grd->nzn + (iz)].x -
          part->data[i].x;
      eta[1] =
          grd->nodes_flat[ix * grd->nyn * grd->nzn + (iy)*grd->nzn + (iz)].y -
          part->data[i].y;
      zeta[1] =
          grd->nodes_flat[ix * grd->nyn * grd->nzn + (iy)*grd->nzn + (iz)].z -
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
            Exl += weight[ii][jj][kk] * field->electricField_flat[idx].x;
            Eyl += weight[ii][jj][kk] * field->electricField_flat[idx].y;
            Ezl += weight[ii][jj][kk] * field->electricField_flat[idx].z;
            Bxl += weight[ii][jj][kk] * field->magneticField_flat[idx].x;
            Byl += weight[ii][jj][kk] * field->magneticField_flat[idx].y;
            Bzl += weight[ii][jj][kk] * field->magneticField_flat[idx].z;
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

  // 1. Allocate memory for structs on device
  EMfield* d_field;
  grid* d_grd;
  parameters* d_param;

  cudaMalloc(&d_field, sizeof(EMfield));
  cudaMalloc(&d_grd, sizeof(grid));
  cudaMalloc(&d_param, sizeof(parameters));

  // 2. Allocate memory for arrays on device

  // Grid and field arrays
  Vec3<FPpart>* d_nodes_flat;
  Vec3<FPfield>* d_electricField_flat;
  Vec3<FPfield>* d_magneticField_flat;

  cudaMalloc(&d_nodes_flat, nxn * nyn * nzn * sizeof(Vec3<FPpart>));
  cudaMalloc(&d_electricField_flat, nxn * nyn * nzn * sizeof(Vec3<FPfield>));
  cudaMalloc(&d_magneticField_flat, nxn * nyn * nzn * sizeof(Vec3<FPfield>));

  // 3. Copy array data to device

  cudaMemcpy(d_nodes_flat, grd->nodes_flat,
             nxn * nyn * nzn * sizeof(Vec3<FPpart>), cudaMemcpyHostToDevice);

  cudaMemcpy(d_electricField_flat, field->electricField_flat,
             nxn * nyn * nzn * sizeof(Vec3<FPfield>), cudaMemcpyHostToDevice);
  cudaMemcpy(d_magneticField_flat, field->magneticField_flat,
             nxn * nyn * nzn * sizeof(Vec3<FPfield>), cudaMemcpyHostToDevice);

  // 4. Create temporary structs with device pointers
  EMfield temp_field = *field;
  grid temp_grd = *grd;

  // 5. Update pointers in temporary structs to point to device memory

  temp_field.electricField_flat = d_electricField_flat;
  temp_field.magneticField_flat = d_magneticField_flat;
  temp_grd.nodes_flat = d_nodes_flat;

  // 6. Copy the temporary structs with device pointers to device
  cudaMemcpy(d_field, &temp_field, sizeof(EMfield), cudaMemcpyHostToDevice);
  cudaMemcpy(d_grd, &temp_grd, sizeof(grid), cudaMemcpyHostToDevice);
  cudaMemcpy(d_param, param, sizeof(parameters), cudaMemcpyHostToDevice);

  // 7. Launch kernel

  int threadsPerBlock = 256;
  for (int is = 0; is < param->ns; is++) {
    auto currPart = &part[is];

    int nop = currPart->nop;
    particles* d_part;
    cudaMalloc(&d_part, sizeof(particles));

    Particle* d_data;
    cudaMalloc(&d_data, nop * sizeof(Particle));

    cudaMemcpy(d_data, currPart->data, nop * sizeof(Particle),
               cudaMemcpyHostToDevice);
    particles temp_part = *currPart;
    temp_part.data = d_data;
    cudaMemcpy(d_part, &temp_part, sizeof(particles), cudaMemcpyHostToDevice);

    int blocksPerGrid = (currPart->nop + threadsPerBlock - 1) / threadsPerBlock;
    mover_PC_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_part, d_field, d_grd,
                                                        d_param);
    cudaDeviceSynchronize();

    // 8. Copy results back to host
    cudaMemcpy(currPart->data, d_data, nop * sizeof(Particle),
               cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_part);
  }

  // 9. Free device memory
  cudaFree(d_nodes_flat);
  cudaFree(d_electricField_flat);
  cudaFree(d_magneticField_flat);
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

  for (long long i = 0; i < part->nop; i++) {
    // determine cell: can we change to int()? is it faster?
    ix = 2 + int(floor((part->data[i].x - grd->xStart) * grd->invdx));
    iy = 2 + int(floor((part->data[i].y - grd->yStart) * grd->invdy));
    iz = 2 + int(floor((part->data[i].z - grd->zStart) * grd->invdz));

    // distances from node
    xi[0] = part->data[i].x - grd->nodes[ix - 1][iy][iz].x;
    eta[0] = part->data[i].y - grd->nodes[ix][iy - 1][iz].y;
    zeta[0] = part->data[i].z - grd->nodes[ix][iy][iz - 1].z;
    xi[1] = grd->nodes[ix][iy][iz].x - part->data[i].x;
    eta[1] = grd->nodes[ix][iy][iz].y - part->data[i].y;
    zeta[1] = grd->nodes[ix][iy][iz].z - part->data[i].z;

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
