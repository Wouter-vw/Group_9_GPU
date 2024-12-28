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

__device__ void call_weight(int ix, int iy, int iz, FPpart x, FPpart y,
                            FPpart z, FPpart u, FPpart v, FPpart w, FPpart q,
                            Vec3<FPfield>* d_nodes, FPpart d_invVOL,
                            FPpart weight[2][2][2], int d_nyn, int d_nzn) {
  FPpart xi[2], eta[2], zeta[2];

  // Compute offsets
  xi[0] = x - d_nodes[(ix - 1) * (d_nyn * d_nzn) + (iy)*d_nzn + (iz)].x;
  eta[0] = y - d_nodes[ix * d_nyn * d_nzn + (iy - 1) * d_nzn + (iz)].y;
  zeta[0] = z - d_nodes[ix * d_nyn * d_nzn + (iy)*d_nzn + (iz - 1)].z;

  int index = ix * d_nyn * d_nzn + (iy)*d_nzn + (iz);
  xi[1] = d_nodes[index].x - x;
  eta[1] = d_nodes[index].y - y;
  zeta[1] = d_nodes[index].z - z;

  // Compute weights
  for (int ii = 0; ii < 2; ii++) {
    for (int jj = 0; jj < 2; jj++) {
      for (int kk = 0; kk < 2; kk++) {
        weight[ii][jj][kk] =
            q * xi[ii] * eta[jj] * zeta[kk] * d_invVOL * d_invVOL;
      }
    }
  }
  // printf("Weight calculation success: (%d, %d, %d)\n", ix, iy, iz);
}

__global__ void calculate_weight(Particle* d_data, Vec3<FPfield>* d_nodes,
                                 FPpart* d_weight, FPpart d_invVOL,
                                 FPpart d_xStart, FPpart d_yStart,
                                 FPpart d_zStart, FPpart d_invdx,
                                 FPpart d_invdy, FPpart d_invdz, int d_nxn,
                                 int d_nyn, int d_nzn, int d_nop) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < d_nop) {
    // arrays needed for interpolation
    // FPpart temp[2][2][2];

    FPpart x = d_data[i].x;
    FPpart y = d_data[i].y;
    FPpart z = d_data[i].z;
    FPpart q = d_data[i].q;

    // index of the cell
    int ix, iy, iz;

    // determine cell: can we change to int()? is it faster?
    ix = 2 + int(floor((x - d_xStart) * d_invdx));
    iy = 2 + int(floor((y - d_yStart) * d_invdy));
    iz = 2 + int(floor((z - d_zStart) * d_invdz));

    // FPpart weight[2][2][2];

    FPpart xi[2], eta[2], zeta[2];

    // Compute offsets
    xi[0] = x - d_nodes[(ix - 1) * (d_nyn * d_nzn) + (iy)*d_nzn + (iz)].x;
    eta[0] = y - d_nodes[ix * d_nyn * d_nzn + (iy - 1) * d_nzn + (iz)].y;
    zeta[0] = z - d_nodes[ix * d_nyn * d_nzn + (iy)*d_nzn + (iz - 1)].z;

    int index = ix * d_nyn * d_nzn + (iy)*d_nzn + (iz);
    xi[1] = d_nodes[index].x - x;
    eta[1] = d_nodes[index].y - y;
    zeta[1] = d_nodes[index].z - z;

    // Compute weights
    for (int ii = 0; ii < 2; ii++) {
      for (int jj = 0; jj < 2; jj++) {
        for (int kk = 0; kk < 2; kk++) {
          // weight[ii][jj][kk] = q * xi[ii] * eta[jj] * zeta[kk] * d_invVOL *
          // d_invVOL;

          int ind = 8 * i + 4 * ii + 2 * jj + kk;
          d_weight[ind] = q * xi[ii] * eta[jj] * zeta[kk] * d_invVOL * d_invVOL;
        }
      }
    }
  }
}

__global__ void interpP2G_kernel_rhon(Particle* d_data, FPpart* d_rhon_flat,
                                      FPpart* d_weight, FPpart d_invVOL,
                                      FPpart d_xStart, FPpart d_yStart,
                                      FPpart d_zStart, FPpart d_invdx,
                                      FPpart d_invdy, FPpart d_invdz, int d_nxn,
                                      int d_nyn, int d_nzn, int d_nop) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < d_nop) {
    // arrays needed for interpolation
    // FPpart temp[2][2][2];

    FPpart x = d_data[i].x;
    FPpart y = d_data[i].y;
    FPpart z = d_data[i].z;

    // index of the cell
    int ix, iy, iz;

    // determine cell: can we change to int()? is it faster?
    ix = 2 + int(floor((x - d_xStart) * d_invdx));
    iy = 2 + int(floor((y - d_yStart) * d_invdy));
    iz = 2 + int(floor((z - d_zStart) * d_invdz));

    // Atomic updates to global memory
    for (int ii = 0; ii < 2; ii++) {
      for (int jj = 0; jj < 2; jj++) {
        for (int kk = 0; kk < 2; kk++) {
          int index =
              (ix - ii) * (d_nyn * d_nzn) + (iy - jj) * d_nzn + (iz - kk);
          int ind = 8 * i + 4 * ii + 2 * jj + kk;
          atomicAdd(&d_rhon_flat[index], d_weight[ind]);
        }
      }
    }
  }
}

__global__ void interpP2G_kernel_Jx(Particle* d_data, FPpart* d_Jx_flat,
                                    FPpart* d_weight, FPpart d_invVOL,
                                    FPpart d_xStart, FPpart d_yStart,
                                    FPpart d_zStart, FPpart d_invdx,
                                    FPpart d_invdy, FPpart d_invdz, int d_nxn,
                                    int d_nyn, int d_nzn, int d_nop) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < d_nop) {
    // arrays needed for interpolation
    // FPpart temp[2][2][2];

    FPpart x = d_data[i].x;
    FPpart y = d_data[i].y;
    FPpart z = d_data[i].z;
    FPpart u = d_data[i].u;

    // index of the cell
    int ix, iy, iz;

    // determine cell: can we change to int()? is it faster?
    ix = 2 + int(floor((x - d_xStart) * d_invdx));
    iy = 2 + int(floor((y - d_yStart) * d_invdy));
    iz = 2 + int(floor((z - d_zStart) * d_invdz));

    // Atomic updates to global memory
    for (int ii = 0; ii < 2; ii++) {
      for (int jj = 0; jj < 2; jj++) {
        for (int kk = 0; kk < 2; kk++) {
          int index =
              (ix - ii) * (d_nyn * d_nzn) + (iy - jj) * d_nzn + (iz - kk);
          int ind = 8 * i + 4 * ii + 2 * jj + kk;

          atomicAdd(&d_Jx_flat[index], u * d_weight[ind]);
        }
      }
    }
  }
}

__global__ void interpP2G_kernel_Jy(Particle* d_data, FPpart* d_Jy_flat,
                                    FPpart* d_weight, FPpart d_invVOL,
                                    FPpart d_xStart, FPpart d_yStart,
                                    FPpart d_zStart, FPpart d_invdx,
                                    FPpart d_invdy, FPpart d_invdz, int d_nxn,
                                    int d_nyn, int d_nzn, int d_nop) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < d_nop) {
    // arrays needed for interpolation
    // FPpart temp[2][2][2];

    FPpart x = d_data[i].x;
    FPpart y = d_data[i].y;
    FPpart z = d_data[i].z;
    FPpart v = d_data[i].v;

    // index of the cell
    int ix, iy, iz;

    // determine cell: can we change to int()? is it faster?
    ix = 2 + int(floor((x - d_xStart) * d_invdx));
    iy = 2 + int(floor((y - d_yStart) * d_invdy));
    iz = 2 + int(floor((z - d_zStart) * d_invdz));

    // std::cout << weight[0][0][0] << std::endl;
    // Atomic updates to global memory
    for (int ii = 0; ii < 2; ii++) {
      for (int jj = 0; jj < 2; jj++) {
        for (int kk = 0; kk < 2; kk++) {
          int index =
              (ix - ii) * (d_nyn * d_nzn) + (iy - jj) * d_nzn + (iz - kk);
          int ind = 8 * i + 4 * ii + 2 * jj + kk;

          atomicAdd(&d_Jy_flat[index], v * d_weight[ind]);
        }
      }
    }
  }
}

__global__ void interpP2G_kernel_Jz(Particle* d_data, FPpart* d_Jz_flat,
                                    FPpart* d_weight, FPpart d_invVOL,
                                    FPpart d_xStart, FPpart d_yStart,
                                    FPpart d_zStart, FPpart d_invdx,
                                    FPpart d_invdy, FPpart d_invdz, int d_nxn,
                                    int d_nyn, int d_nzn, int d_nop) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < d_nop) {
    // arrays needed for interpolation
    // FPpart temp[2][2][2];

    FPpart x = d_data[i].x;
    FPpart y = d_data[i].y;
    FPpart z = d_data[i].z;
    FPpart w = d_data[i].w;

    // index of the cell
    int ix, iy, iz;

    // determine cell: can we change to int()? is it faster?
    ix = 2 + int(floor((x - d_xStart) * d_invdx));
    iy = 2 + int(floor((y - d_yStart) * d_invdy));
    iz = 2 + int(floor((z - d_zStart) * d_invdz));

    // Atomic updates to global memory
    for (int ii = 0; ii < 2; ii++) {
      for (int jj = 0; jj < 2; jj++) {
        for (int kk = 0; kk < 2; kk++) {
          int index =
              (ix - ii) * (d_nyn * d_nzn) + (iy - jj) * d_nzn + (iz - kk);
          int ind = 8 * i + 4 * ii + 2 * jj + kk;

          atomicAdd(&d_Jz_flat[index], w * d_weight[ind]);
        }
      }
    }
  }
}

__global__ void interpP2G_kernel_pxx(Particle* d_data, FPpart* d_pxx_flat,
                                     FPpart* d_weight, FPpart d_invVOL,
                                     FPpart d_xStart, FPpart d_yStart,
                                     FPpart d_zStart, FPpart d_invdx,
                                     FPpart d_invdy, FPpart d_invdz, int d_nxn,
                                     int d_nyn, int d_nzn, int d_nop) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < d_nop) {
    // arrays needed for interpolation
    // FPpart temp[2][2][2];

    FPpart x = d_data[i].x;
    FPpart y = d_data[i].y;
    FPpart z = d_data[i].z;
    FPpart u = d_data[i].u;

    // index of the cell
    int ix, iy, iz;

    // determine cell: can we change to int()? is it faster?
    ix = 2 + int(floor((x - d_xStart) * d_invdx));
    iy = 2 + int(floor((y - d_yStart) * d_invdy));
    iz = 2 + int(floor((z - d_zStart) * d_invdz));

    // Atomic updates to global memory
    for (int ii = 0; ii < 2; ii++) {
      for (int jj = 0; jj < 2; jj++) {
        for (int kk = 0; kk < 2; kk++) {
          int index =
              (ix - ii) * (d_nyn * d_nzn) + (iy - jj) * d_nzn + (iz - kk);
          int ind = 8 * i + 4 * ii + 2 * jj + kk;

          atomicAdd(&d_pxx_flat[index], u * u * d_weight[ind]);
        }
      }
    }
  }
}

__global__ void interpP2G_kernel_pxy(Particle* d_data, FPpart* d_pxy_flat,
                                     FPpart* d_weight, FPpart d_invVOL,
                                     FPpart d_xStart, FPpart d_yStart,
                                     FPpart d_zStart, FPpart d_invdx,
                                     FPpart d_invdy, FPpart d_invdz, int d_nxn,
                                     int d_nyn, int d_nzn, int d_nop) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < d_nop) {
    // arrays needed for interpolation
    // FPpart temp[2][2][2];

    FPpart x = d_data[i].x;
    FPpart y = d_data[i].y;
    FPpart z = d_data[i].z;
    FPpart u = d_data[i].u;
    FPpart v = d_data[i].v;

    // index of the cell
    int ix, iy, iz;

    // determine cell: can we change to int()? is it faster?
    ix = 2 + int(floor((x - d_xStart) * d_invdx));
    iy = 2 + int(floor((y - d_yStart) * d_invdy));
    iz = 2 + int(floor((z - d_zStart) * d_invdz));

    // Atomic updates to global memory
    for (int ii = 0; ii < 2; ii++) {
      for (int jj = 0; jj < 2; jj++) {
        for (int kk = 0; kk < 2; kk++) {
          int index =
              (ix - ii) * (d_nyn * d_nzn) + (iy - jj) * d_nzn + (iz - kk);
          int ind = 8 * i + 4 * ii + 2 * jj + kk;

          atomicAdd(&d_pxy_flat[index], u * v * d_weight[ind]);
        }
      }
    }
  }
}

__global__ void interpP2G_kernel_pxz(Particle* d_data, FPpart* d_pxz_flat,
                                     FPpart* d_weight, FPpart d_invVOL,
                                     FPpart d_xStart, FPpart d_yStart,
                                     FPpart d_zStart, FPpart d_invdx,
                                     FPpart d_invdy, FPpart d_invdz, int d_nxn,
                                     int d_nyn, int d_nzn, int d_nop) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < d_nop) {
    // arrays needed for interpolation
    // FPpart temp[2][2][2];

    FPpart x = d_data[i].x;
    FPpart y = d_data[i].y;
    FPpart z = d_data[i].z;
    FPpart u = d_data[i].u;
    FPpart w = d_data[i].w;

    // index of the cell
    int ix, iy, iz;

    // determine cell: can we change to int()? is it faster?
    ix = 2 + int(floor((x - d_xStart) * d_invdx));
    iy = 2 + int(floor((y - d_yStart) * d_invdy));
    iz = 2 + int(floor((z - d_zStart) * d_invdz));

    // Atomic updates to global memory
    for (int ii = 0; ii < 2; ii++) {
      for (int jj = 0; jj < 2; jj++) {
        for (int kk = 0; kk < 2; kk++) {
          int index =
              (ix - ii) * (d_nyn * d_nzn) + (iy - jj) * d_nzn + (iz - kk);
          int ind = 8 * i + 4 * ii + 2 * jj + kk;

          atomicAdd(&d_pxz_flat[index], u * w * d_weight[ind]);
        }
      }
    }
  }
}

__global__ void interpP2G_kernel_pyy(Particle* d_data, FPpart* d_pyy_flat,
                                     FPpart* d_weight, FPpart d_invVOL,
                                     FPpart d_xStart, FPpart d_yStart,
                                     FPpart d_zStart, FPpart d_invdx,
                                     FPpart d_invdy, FPpart d_invdz, int d_nxn,
                                     int d_nyn, int d_nzn, int d_nop) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < d_nop) {
    // arrays needed for interpolation
    // FPpart temp[2][2][2];

    FPpart x = d_data[i].x;
    FPpart y = d_data[i].y;
    FPpart z = d_data[i].z;
    FPpart v = d_data[i].v;

    // index of the cell
    int ix, iy, iz;

    // determine cell: can we change to int()? is it faster?
    ix = 2 + int(floor((x - d_xStart) * d_invdx));
    iy = 2 + int(floor((y - d_yStart) * d_invdy));
    iz = 2 + int(floor((z - d_zStart) * d_invdz));

    // Atomic updates to global memory
    for (int ii = 0; ii < 2; ii++) {
      for (int jj = 0; jj < 2; jj++) {
        for (int kk = 0; kk < 2; kk++) {
          int index =
              (ix - ii) * (d_nyn * d_nzn) + (iy - jj) * d_nzn + (iz - kk);
          int ind = 8 * i + 4 * ii + 2 * jj + kk;

          atomicAdd(&d_pyy_flat[index], v * v * d_weight[ind]);
        }
      }
    }
  }
}

__global__ void interpP2G_kernel_pyz(Particle* d_data, FPpart* d_pyz_flat,
                                     FPpart* d_weight, FPpart d_invVOL,
                                     FPpart d_xStart, FPpart d_yStart,
                                     FPpart d_zStart, FPpart d_invdx,
                                     FPpart d_invdy, FPpart d_invdz, int d_nxn,
                                     int d_nyn, int d_nzn, int d_nop) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < d_nop) {
    // arrays needed for interpolation
    // FPpart temp[2][2][2];

    FPpart x = d_data[i].x;
    FPpart y = d_data[i].y;
    FPpart z = d_data[i].z;
    FPpart v = d_data[i].v;
    FPpart w = d_data[i].w;

    // index of the cell
    int ix, iy, iz;

    // determine cell: can we change to int()? is it faster?
    ix = 2 + int(floor((x - d_xStart) * d_invdx));
    iy = 2 + int(floor((y - d_yStart) * d_invdy));
    iz = 2 + int(floor((z - d_zStart) * d_invdz));

    // Atomic updates to global memory
    for (int ii = 0; ii < 2; ii++) {
      for (int jj = 0; jj < 2; jj++) {
        for (int kk = 0; kk < 2; kk++) {
          int index =
              (ix - ii) * (d_nyn * d_nzn) + (iy - jj) * d_nzn + (iz - kk);
          int ind = 8 * i + 4 * ii + 2 * jj + kk;

          // int index_w = i * nop + ii * 4 + jj * 2 + zz;
          atomicAdd(&d_pyz_flat[index], v * w * d_weight[ind]);
        }
      }
    }
  }
}

__global__ void interpP2G_kernel_pzz(Particle* d_data, FPpart* d_pzz_flat,
                                     FPpart* d_weight, FPpart d_invVOL,
                                     FPpart d_xStart, FPpart d_yStart,
                                     FPpart d_zStart, FPpart d_invdx,
                                     FPpart d_invdy, FPpart d_invdz, int d_nxn,
                                     int d_nyn, int d_nzn, int d_nop) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < d_nop) {
    // arrays needed for interpolation
    // FPpart temp[2][2][2];

    FPpart x = d_data[i].x;
    FPpart y = d_data[i].y;
    FPpart z = d_data[i].z;
    FPpart w = d_data[i].w;

    // index of the cell
    int ix, iy, iz;

    // determine cell: can we change to int()? is it faster?
    ix = 2 + int(floor((x - d_xStart) * d_invdx));
    iy = 2 + int(floor((y - d_yStart) * d_invdy));
    iz = 2 + int(floor((z - d_zStart) * d_invdz));

    // Atomic updates to global memory
    for (int ii = 0; ii < 2; ii++) {
      for (int jj = 0; jj < 2; jj++) {
        for (int kk = 0; kk < 2; kk++) {
          int index =
              (ix - ii) * (d_nyn * d_nzn) + (iy - jj) * d_nzn + (iz - kk);
          int ind = 8 * i + 4 * ii + 2 * jj + kk;

          atomicAdd(&d_pzz_flat[index], w * w * d_weight[ind]);
        }
      }
    }
  }
}

__global__ void interpP2G_kernel(
    Particle* d_data, Vec3<FPfield>* d_nodes, FPpart* d_rhon_flat,
    FPpart* d_Jx_flat, FPpart* d_Jy_flat, FPpart* d_Jz_flat, FPpart* d_pxx_flat,
    FPpart* d_pxy_flat, FPpart* d_pxz_flat, FPpart* d_pyy_flat,
    FPpart* d_pyz_flat, FPpart* d_pzz_flat, FPpart d_invVOL, FPpart d_xStart,
    FPpart d_yStart, FPpart d_zStart, FPpart d_invdx, FPpart d_invdy,
    FPpart d_invdz, int d_nxn, int d_nyn, int d_nzn, int d_nop) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < d_nop) {
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    // FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];

    FPpart x = d_data[i].x;
    FPpart y = d_data[i].y;
    FPpart z = d_data[i].z;
    FPpart u = d_data[i].u;
    FPpart v = d_data[i].v;
    FPpart w = d_data[i].w;
    FPpart q = d_data[i].q;

    // index of the cell
    int ix, iy, iz;

    // determine cell: can we change to int()? is it faster?
    ix = 2 + int(floor((x - d_xStart) * d_invdx));
    iy = 2 + int(floor((y - d_yStart) * d_invdy));
    iz = 2 + int(floor((z - d_zStart) * d_invdz));

    // Check indixing
    xi[0] = x - d_nodes[(ix - 1) * (d_nyn * d_nzn) + (iy)*d_nzn + (iz)].x;
    eta[0] = y - d_nodes[ix * d_nyn * d_nzn + (iy - 1) * d_nzn + (iz)].y;
    zeta[0] = z - d_nodes[ix * d_nyn * d_nzn + (iy)*d_nzn + (iz - 1)].z;

    int index = ix * d_nyn * d_nzn + (iy)*d_nzn + (iz);
    xi[1] = d_nodes[index].x - x;
    eta[1] = d_nodes[index].y - y;
    zeta[1] = d_nodes[index].z - z;

    for (int ii = 0; ii < 2; ii++) {
      for (int jj = 0; jj < 2; jj++) {
        for (int kk = 0; kk < 2; kk++) {
          weight[ii][jj][kk] =
              q * xi[ii] * eta[jj] * zeta[kk] * d_invVOL * d_invVOL;
        }
      }
    }

    // Atomic updates to global memory
    for (int ii = 0; ii < 2; ii++) {
      for (int jj = 0; jj < 2; jj++) {
        for (int kk = 0; kk < 2; kk++) {
          int index =
              (ix - ii) * (d_nyn * d_nzn) + (iy - jj) * d_nzn + (iz - kk);
          atomicAdd(&d_rhon_flat[index], weight[ii][jj][kk]);
          atomicAdd(&d_Jx_flat[index], u * weight[ii][jj][kk]);
          atomicAdd(&d_Jy_flat[index], v * weight[ii][jj][kk]);
          atomicAdd(&d_Jz_flat[index], w * weight[ii][jj][kk]);
          atomicAdd(&d_pxx_flat[index], u * u * weight[ii][jj][kk]);
          atomicAdd(&d_pxy_flat[index], u * v * weight[ii][jj][kk]);
          atomicAdd(&d_pxz_flat[index], u * w * weight[ii][jj][kk]);
          atomicAdd(&d_pyy_flat[index], v * v * weight[ii][jj][kk]);
          atomicAdd(&d_pyz_flat[index], v * w * weight[ii][jj][kk]);
          atomicAdd(&d_pzz_flat[index], w * w * weight[ii][jj][kk]);
        }
      }
    }
  }
}

// __global__ void interpP2G_kernel_3D(
//    Particle* d_data,
//    Vec3<FPfield>* d_nodes,
//     FPpart* d_rhon_flat,
//     FPpart* d_Jx_flat, FPpart* d_Jy_flat, FPpart* d_Jz_flat,
//     FPpart* d_pxx_flat, FPpart* d_pxy_flat, FPpart* d_pxz_flat,
//     FPpart* d_pyy_flat, FPpart* d_pyz_flat, FPpart* d_pzz_flat,
//     FPpart d_invVOL, FPpart d_xStart, FPpart d_yStart, FPpart d_zStart,
//     FPpart d_invdx, FPpart d_invdy, FPpart d_invdz,
//     int d_nxn, int d_nyn, int d_nzn, int d_nop)
// {

//     int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
//     int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
//     int tid = tx + ty * blockDim.x + tz * blockDim.x * blockDim.y;

//     int xx = tx + bx * blockDim.x;
//     int yy = ty + by * blockDim.y;
//     int zz = tz + bz * blockDim.z;

//     int i = threadIdx.x + blockDim.x * blockIdx.x;

//     // Define shared memory using extern __shared__
//     extern __shared__ FPpart shared_mem[];

//     // Allocate shared memory for each array
//     FPpart* s_rhon = shared_mem; // First portion
//     FPpart* s_Jx = &s_rhon[(blockDim.x + 2) * (blockDim.y + 2) * (blockDim.z
//     + 2)]; FPpart* s_Jy = &s_Jx[(blockDim.x + 2) * (blockDim.y + 2) *
//     (blockDim.z + 2)]; FPpart* s_Jz = &s_Jy[(blockDim.x + 2) * (blockDim.y +
//     2) * (blockDim.z + 2)]; FPpart* s_pxx = &s_Jz[(blockDim.x + 2) *
//     (blockDim.y + 2) * (blockDim.z + 2)]; FPpart* s_pxy = &s_pxx[(blockDim.x
//     + 2) * (blockDim.y + 2) * (blockDim.z + 2)]; FPpart* s_pxz =
//     &s_pxy[(blockDim.x + 2) * (blockDim.y + 2) * (blockDim.z + 2)]; FPpart*
//     s_pyy = &s_pxz[(blockDim.x + 2) * (blockDim.y + 2) * (blockDim.z + 2)];
//     FPpart* s_pyz = &s_pyy[(blockDim.x + 2) * (blockDim.y + 2) * (blockDim.z
//     + 2)]; FPpart* s_pzz = &s_pyz[(blockDim.x + 2) * (blockDim.y + 2) *
//     (blockDim.z + 2)];

//     int global_x = blockIdx.x * blockDim.x + threadIdx.x;
//     int global_y = blockIdx.y * blockDim.y + threadIdx.y;
//     int global_z = blockIdx.z * blockDim.z + threadIdx.z;

//     // Local thread index in 3D space (including padding)
//     int local_x = threadIdx.x + 1; // Offset for halo
//     int local_y = threadIdx.y + 1;
//     int local_z = threadIdx.z + 1;

//     int shared_nx = blockDim.x + 2; // Include halo
//     int shared_ny = blockDim.y + 2;
//     int shared_nz = blockDim.z + 2;

//     int shared_idx = local_x * shared_ny * shared_nz + local_y * shared_nz +
//     local_z;

//     // add zero and n+1
//     if (global_x < d_nxn && global_y < d_nyn && global_z < d_nzn) {
//         int global_idx = global_x * d_nyn * d_nzn + global_y * d_nzn +
//         global_z;

//         s_rhon[shared_idx] = d_rhon_flat[global_idx];
//         s_Jx[shared_idx] = d_Jx_flat[global_idx];
//         s_Jy[shared_idx] = d_Jy_flat[global_idx];
//         s_Jz[shared_idx] = d_Jz_flat[global_idx];
//         s_pxx[shared_idx] = d_pxx_flat[global_idx];
//         s_pxy[shared_idx] = d_pxy_flat[global_idx];
//         s_pxz[shared_idx] = d_pxz_flat[global_idx];
//         s_pyy[shared_idx] = d_pyy_flat[global_idx];
//         s_pyz[shared_idx] = d_pyz_flat[global_idx];
//         s_pzz[shared_idx] = d_pzz_flat[global_idx];
//     }

//     __syncthreads();

//     for (int i = shared_idx; i < d_nop; i += blockDim.x * blockDim.y *
//     blockDim.z) {
//         // arrays needed for interpolation
//         FPpart weight[2][2][2];
//         // FPpart temp[2][2][2];
//         FPpart xi[2], eta[2], zeta[2];

//         FPpart x = d_data[i].x;
//         FPpart y = d_data[i].y;
//         FPpart z = d_data[i].z;
//         FPpart u = d_data[i].u;
//         FPpart v = d_data[i].v;
//         FPpart w = d_data[i].w;
//         FPpart q = d_data[i].q;

//         // index of the cell
//         int ix, iy, iz;

//         // determine cell: can we change to int()? is it faster?
//         ix = 2 + int (floor((x - d_xStart) * d_invdx));
//         iy = 2 + int (floor((y - d_yStart) * d_invdy));
//         iz = 2 + int (floor((z - d_zStart) * d_invdz));

//         // Check indixing
//         xi[0] = x - d_XN_flat[(ix - 1) * (d_nyn * d_nzn) + (iy) * d_nzn +
//         (iz)]; eta[0] = y - d_YN_flat[ix * d_nyn * d_nzn + (iy - 1) * d_nzn +
//         (iz) ]; zeta[0] = z - d_ZN_flat[ix * d_nyn * d_nzn + (iy) * d_nzn +
//         (iz - 1)];

//         int index = ix * d_nyn * d_nzn + (iy) * d_nzn + (iz);
//         xi[1] = d_nodes[index].x - x;
//         eta[1] = d_nodes[index].y - y;
//         zeta[1] = d_nodes[index].z - z;

//         for (int ii = 0; ii < 2; ii++) {
//             for (int jj = 0; jj < 2; jj++) {
//                 for (int kk = 0; kk < 2; kk++) {
//                     weight[ii][jj][kk] = q * xi[ii] * eta[jj] * zeta[kk] *
//                     d_invVOL * d_invVOL;
//                 }
//             }
//         }

//         // Atomic updates to global memory
//         for (int ii = 0; ii < 2; ii++) {
//             for (int jj = 0; jj < 2; jj++) {
//                 for (int kk = 0; kk < 2; kk++) {
//                     // int index = (ix - ii) * (d_nyn * d_nzn) + (iy - jj) *
//                     d_nzn + (iz - kk); int shared_idx = local_x * shared_ny *
//                     shared_nz + local_y * shared_nz + local_z;

//                     atomicAdd(&s_rhon[index], weight[ii][jj][kk]);
//                     atomicAdd(&s_Jx[index], u * weight[ii][jj][kk]);
//                     atomicAdd(&s_Jy[index], v * weight[ii][jj][kk]);
//                     atomicAdd(&s_Jz[index], w * weight[ii][jj][kk]);
//                     atomicAdd(&s_pxx[index], u * u * weight[ii][jj][kk]);
//                     atomicAdd(&s_pxy[index], u * v * weight[ii][jj][kk]);
//                     atomicAdd(&s_pxz[index], u * w * weight[ii][jj][kk]);
//                     atomicAdd(&s_pyy[index], v * v * weight[ii][jj][kk]);
//                     atomicAdd(&s_pyz[index], v * w * weight[ii][jj][kk]);
//                     atomicAdd(&s_pzz[index], w * w * weight[ii][jj][kk]);
//                 }
//             }
//         }

//         // __syncthreads();

//         // Write back shared memory to global memory
//         // if (i < d_nxn * d_nyn * d_nzn) {
//         //     atomicAdd(&d_rhon_flat[i], s_rhon[i]);
//         //     atomicAdd(&d_Jx_flat[i], s_Jx[i]);
//         //     atomicAdd(&d_Jy_flat[i], s_Jy[i]);
//         //     atomicAdd(&d_Jz_flat[i], s_Jz[i]);
//         //     atomicAdd(&d_pxx_flat[i], s_pxx[i]);
//         //     atomicAdd(&d_pxy_flat[i], s_pxy[i]);
//         //     atomicAdd(&d_pxz_flat[i], s_pxz[i]);
//         //     atomicAdd(&d_pyy_flat[i], s_pyy[i]);
//         //     atomicAdd(&d_pyz_flat[i], s_pyz[i]);
//         //     atomicAdd(&d_pzz_flat[i], s_pzz[i]);
//         // }

//     }

//     __syncthreads();

//     if (global_x < d_nxn && global_y < d_nyn && global_z < d_nzn) {
//             int global_idx = global_x * d_nyn * d_nzn + global_y * d_nzn +
//             global_z; atomicAdd(&d_rhon_flat[global_idx],
//             s_rhon[shared_idx]); atomicAdd(&d_Jx_flat[global_idx],
//             s_Jx[shared_idx]); atomicAdd(&d_Jy_flat[global_idx],
//             s_Jy[shared_idx]); atomicAdd(&d_Jz_flat[global_idx],
//             s_Jz[shared_idx]); atomicAdd(&d_pxx_flat[global_idx],
//             s_pxx[shared_idx]); atomicAdd(&d_pxy_flat[global_idx],
//             s_pxy[shared_idx]); atomicAdd(&d_pxz_flat[global_idx],
//             s_pxz[shared_idx]); atomicAdd(&d_pyy_flat[global_idx],
//             s_pyy[shared_idx]); atomicAdd(&d_pyz_flat[global_idx],
//             s_pyz[shared_idx]); atomicAdd(&d_pzz_flat[global_idx],
//             s_pzz[shared_idx]);
//         }
// }
#define CUDA_CHECK(call)                                      \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(1);                                                \
    }                                                         \
  } while (0)
// This is Async V2
void interpP2G_GPU(struct particles* part, struct interpDensSpecies* ids,
                   struct grid* grd) {
  // corresponds to XN, YN etc. also Ex, Bxn COPY for slimpicity
  int nxn = grd->nxn;
  int nyn = grd->nyn;
  int nzn = grd->nzn;
  int nop = part->nop;

  // Flatten XN, YN, ZN, rhon, J and p -> Free at the end
  FPpart* rhon_flat = ids->rhon_flat;
  FPpart* Jx_flat = ids->Jx_flat;
  FPpart* Jy_flat = ids->Jy_flat;
  FPpart* Jz_flat = ids->Jz_flat;
  FPpart* pxx_flat = ids->pxx_flat;
  FPpart* pxy_flat = ids->pxy_flat;
  FPpart* pxz_flat = ids->pxz_flat;
  FPpart* pyy_flat = ids->pyy_flat;
  FPpart* pyz_flat = ids->pyz_flat;
  FPpart* pzz_flat = ids->pzz_flat;

  constexpr int nStreams = 10;
  cudaStream_t stream[nStreams];
  for (int i = 0; i < nStreams; ++i) CUDA_CHECK(cudaStreamCreate(&stream[i]));

  // Device pointers -> Free at the end
  Particle* d_data;
  FPpart* d_weight;
  Vec3<FPfield>* d_nodes;
  FPpart *d_rhon_flat, *d_Jx_flat, *d_Jy_flat, *d_Jz_flat;
  FPpart *d_pxx_flat, *d_pxy_flat, *d_pxz_flat, *d_pyy_flat, *d_pyz_flat,
      *d_pzz_flat;

  // Allocate memory on the device
  CUDA_CHECK(cudaMalloc(&d_data, nop * sizeof(Particle)));
  CUDA_CHECK(cudaMalloc(&d_weight, 8 * nop * sizeof(FPpart)));
  CUDA_CHECK(cudaMalloc(&d_nodes, nxn * nyn * nzn * sizeof(Vec3<FPfield>)));
  CUDA_CHECK(cudaMalloc(&d_rhon_flat, nxn * nyn * nzn * sizeof(FPpart)));
  CUDA_CHECK(cudaMalloc(&d_Jx_flat, nxn * nyn * nzn * sizeof(FPpart)));
  CUDA_CHECK(cudaMalloc(&d_Jy_flat, nxn * nyn * nzn * sizeof(FPpart)));
  CUDA_CHECK(cudaMalloc(&d_Jz_flat, nxn * nyn * nzn * sizeof(FPpart)));
  CUDA_CHECK(cudaMalloc(&d_pxx_flat, nxn * nyn * nzn * sizeof(FPpart)));
  CUDA_CHECK(cudaMalloc(&d_pxy_flat, nxn * nyn * nzn * sizeof(FPpart)));
  CUDA_CHECK(cudaMalloc(&d_pxz_flat, nxn * nyn * nzn * sizeof(FPpart)));
  CUDA_CHECK(cudaMalloc(&d_pyy_flat, nxn * nyn * nzn * sizeof(FPpart)));
  CUDA_CHECK(cudaMalloc(&d_pyz_flat, nxn * nyn * nzn * sizeof(FPpart)));
  CUDA_CHECK(cudaMalloc(&d_pzz_flat, nxn * nyn * nzn * sizeof(FPpart)));

  int currentStreamSize = nxn * nyn * nzn * sizeof(FPpart);

  // Copy data from host to device

  CUDA_CHECK(cudaMemcpyAsync(d_data, part->data, nop * sizeof(Particle),
                        cudaMemcpyHostToDevice, stream[0]));
  CUDA_CHECK(cudaMemcpyAsync(d_nodes, grd->nodes_flat,
                        nxn * nyn * nzn * sizeof(Vec3<FPfield>),
                        cudaMemcpyHostToDevice, stream[1]));

  for (int i = 0; i < 2; ++i) cudaStreamSynchronize(stream[i]);

  int threadsPerBlock = 256;
  int blocksPerGrid = (part->nop + threadsPerBlock - 1) / threadsPerBlock;

  calculate_weight<<<blocksPerGrid, threadsPerBlock, 0, stream[5]>>>(
      d_data, d_nodes, d_weight, grd->invVOL, grd->xStart, grd->yStart,
      grd->zStart, grd->invdx, grd->invdy, grd->invdz, nxn, nyn, nzn, nop);

  CUDA_CHECK(cudaGetLastError());
  cudaMemcpyAsync(d_rhon_flat, rhon_flat, currentStreamSize,
                  cudaMemcpyHostToDevice, stream[0]);
  cudaMemcpyAsync(d_Jx_flat, Jx_flat, currentStreamSize, cudaMemcpyHostToDevice,
                  stream[1]);
  cudaMemcpyAsync(d_Jy_flat, Jy_flat, currentStreamSize, cudaMemcpyHostToDevice,
                  stream[2]);
  cudaMemcpyAsync(d_Jz_flat, Jz_flat, currentStreamSize, cudaMemcpyHostToDevice,
                  stream[3]);
  cudaMemcpyAsync(d_pxx_flat, pxx_flat, currentStreamSize,
                  cudaMemcpyHostToDevice, stream[4]);
  cudaMemcpyAsync(d_pxy_flat, pxy_flat, currentStreamSize,
                  cudaMemcpyHostToDevice, stream[5]);
  cudaMemcpyAsync(d_pxz_flat, pxz_flat, currentStreamSize,
                  cudaMemcpyHostToDevice, stream[6]);
  cudaMemcpyAsync(d_pyy_flat, pyy_flat, currentStreamSize,
                  cudaMemcpyHostToDevice, stream[7]);
  cudaMemcpyAsync(d_pyz_flat, pyz_flat, currentStreamSize,
                  cudaMemcpyHostToDevice, stream[8]);
  cudaMemcpyAsync(d_pzz_flat, pzz_flat, currentStreamSize,
                  cudaMemcpyHostToDevice, stream[9]);

  for (int i = 0; i < nStreams; ++i) cudaStreamSynchronize(stream[i]);

  interpP2G_kernel_rhon<<<blocksPerGrid, threadsPerBlock, 0, stream[0]>>>(
      d_data, d_rhon_flat, d_weight, grd->invVOL, grd->xStart, grd->yStart,
      grd->zStart, grd->invdx, grd->invdy, grd->invdz, nxn, nyn, nzn, nop);
  interpP2G_kernel_Jx<<<blocksPerGrid, threadsPerBlock, 0, stream[1]>>>(
      d_data, d_Jx_flat, d_weight, grd->invVOL, grd->xStart, grd->yStart,
      grd->zStart, grd->invdx, grd->invdy, grd->invdz, nxn, nyn, nzn, nop);
  interpP2G_kernel_Jy<<<blocksPerGrid, threadsPerBlock, 0, stream[2]>>>(
      d_data, d_Jy_flat, d_weight, grd->invVOL, grd->xStart, grd->yStart,
      grd->zStart, grd->invdx, grd->invdy, grd->invdz, nxn, nyn, nzn, nop);
  interpP2G_kernel_Jz<<<blocksPerGrid, threadsPerBlock, 0, stream[3]>>>(
      d_data, d_Jz_flat, d_weight, grd->invVOL, grd->xStart, grd->yStart,
      grd->zStart, grd->invdx, grd->invdy, grd->invdz, nxn, nyn, nzn, nop);
  interpP2G_kernel_pxx<<<blocksPerGrid, threadsPerBlock, 0, stream[4]>>>(
      d_data, d_pxx_flat, d_weight, grd->invVOL, grd->xStart, grd->yStart,
      grd->zStart, grd->invdx, grd->invdy, grd->invdz, nxn, nyn, nzn, nop);
  interpP2G_kernel_pxy<<<blocksPerGrid, threadsPerBlock, 0, stream[5]>>>(
      d_data, d_pxy_flat, d_weight, grd->invVOL, grd->xStart, grd->yStart,
      grd->zStart, grd->invdx, grd->invdy, grd->invdz, nxn, nyn, nzn, nop);
  interpP2G_kernel_pxz<<<blocksPerGrid, threadsPerBlock, 0, stream[6]>>>(
      d_data, d_pxz_flat, d_weight, grd->invVOL, grd->xStart, grd->yStart,
      grd->zStart, grd->invdx, grd->invdy, grd->invdz, nxn, nyn, nzn, nop);
  interpP2G_kernel_pyy<<<blocksPerGrid, threadsPerBlock, 0, stream[7]>>>(
      d_data, d_pyy_flat, d_weight, grd->invVOL, grd->xStart, grd->yStart,
      grd->zStart, grd->invdx, grd->invdy, grd->invdz, nxn, nyn, nzn, nop);
  interpP2G_kernel_pyz<<<blocksPerGrid, threadsPerBlock, 0, stream[8]>>>(
      d_data, d_pyz_flat, d_weight, grd->invVOL, grd->xStart, grd->yStart,
      grd->zStart, grd->invdx, grd->invdy, grd->invdz, nxn, nyn, nzn, nop);
  interpP2G_kernel_pzz<<<blocksPerGrid, threadsPerBlock, 0, stream[9]>>>(
      d_data, d_pzz_flat, d_weight, grd->invVOL, grd->xStart, grd->yStart,
      grd->zStart, grd->invdx, grd->invdy, grd->invdz, nxn, nyn, nzn, nop);

  for (int i = 0; i < nStreams; ++i) cudaStreamSynchronize(stream[i]);

  cudaMemcpyAsync(rhon_flat, d_rhon_flat, currentStreamSize,
                  cudaMemcpyDeviceToHost, stream[0]);
  cudaMemcpyAsync(Jx_flat, d_Jx_flat, currentStreamSize, cudaMemcpyDeviceToHost,
                  stream[1]);
  cudaMemcpyAsync(Jy_flat, d_Jy_flat, currentStreamSize, cudaMemcpyDeviceToHost,
                  stream[2]);
  cudaMemcpyAsync(Jz_flat, d_Jz_flat, currentStreamSize, cudaMemcpyDeviceToHost,
                  stream[3]);
  cudaMemcpyAsync(pxx_flat, d_pxx_flat, currentStreamSize,
                  cudaMemcpyDeviceToHost, stream[4]);
  cudaMemcpyAsync(pxy_flat, d_pxy_flat, currentStreamSize,
                  cudaMemcpyDeviceToHost, stream[5]);
  cudaMemcpyAsync(pxz_flat, d_pxz_flat, currentStreamSize,
                  cudaMemcpyDeviceToHost, stream[6]);
  cudaMemcpyAsync(pyy_flat, d_pyy_flat, currentStreamSize,
                  cudaMemcpyDeviceToHost, stream[7]);
  cudaMemcpyAsync(pyz_flat, d_pyz_flat, currentStreamSize,
                  cudaMemcpyDeviceToHost, stream[8]);
  cudaMemcpyAsync(pzz_flat, d_pzz_flat, currentStreamSize,
                  cudaMemcpyDeviceToHost, stream[9]);

  for (int i = 0; i < nStreams; ++i) cudaStreamSynchronize(stream[i]);
  // Free cuda arrays
  cudaFree(d_data);
  cudaFree(d_nodes);
  cudaFree(d_rhon_flat);
  cudaFree(d_Jx_flat);
  cudaFree(d_Jy_flat);
  cudaFree(d_Jz_flat);
  cudaFree(d_pxx_flat);
  cudaFree(d_pxy_flat);
  cudaFree(d_pxz_flat);
  cudaFree(d_pyy_flat);
  cudaFree(d_pyz_flat);
  cudaFree(d_pzz_flat);
  cudaFree(d_weight);
  for (int i = 0; i < nStreams; ++i) cudaStreamDestroy(stream[i]);
}

// //This is without ASYNC
// void interpP2G_GPU(struct particles* part, struct interpDensSpecies* ids,
// struct grid* grd)
// {
//     // corresponds to XN, YN etc. also Ex, Bxn COPY for slimpicity
//     int nxn = grd->nxn;
//     int nyn = grd->nyn;
//     int nzn = grd->nzn;
//     int nop = part->nop;

//     // Flatten XN, YN, ZN, rhon, J and p -> Free at the end
//     FPpart* XN_flat = flattenArray(grd->XN, nxn, nyn, nzn);
//     FPpart* YN_flat = flattenArray(grd->YN, nxn, nyn, nzn);
//     FPpart* ZN_flat = flattenArray(grd->ZN, nxn, nyn, nzn);
//     FPpart* rhon_flat = flattenArray(ids->rhon, nxn, nyn, nzn);
//     FPpart* Jx_flat = flattenArray(ids->Jx, nxn, nyn, nzn);
//     FPpart* Jy_flat = flattenArray(ids->Jy, nxn, nyn, nzn);
//     FPpart* Jz_flat = flattenArray(ids->Jz, nxn, nyn, nzn);
//     FPpart* pxx_flat = flattenArray(ids->pxx, nxn, nyn, nzn);
//     FPpart* pxy_flat = flattenArray(ids->pxy, nxn, nyn, nzn);
//     FPpart* pxz_flat = flattenArray(ids->pxz, nxn, nyn, nzn);
//     FPpart* pyy_flat = flattenArray(ids->pyy, nxn, nyn, nzn);
//     FPpart* pyz_flat = flattenArray(ids->pyz, nxn, nyn, nzn);
//     FPpart* pzz_flat = flattenArray(ids->pzz, nxn, nyn, nzn);

//     // Device pointers -> Free at the end
//     FPpart *d_x, *d_y, *d_z;
//     FPpart *d_u, *d_v, *d_w;
//     FPpart *d_q;
//     FPpart *d_weight;
//     FPpart *d_XN_flat, *d_YN_flat, *d_ZN_flat;
//     FPpart *d_rhon_flat, *d_Jx_flat, *d_Jy_flat, *d_Jz_flat;
//     FPpart *d_pxx_flat, *d_pxy_flat, *d_pxz_flat, *d_pyy_flat, *d_pyz_flat,
//     *d_pzz_flat;

//     // Allocate memory on the device
//     cudaMalloc(&d_x, nop * sizeof(FPpart));
//     cudaMalloc(&d_y, nop * sizeof(FPpart));
//     cudaMalloc(&d_z, nop * sizeof(FPpart));
//     cudaMalloc(&d_u, nop * sizeof(FPpart));
//     cudaMalloc(&d_v, nop * sizeof(FPpart));
//     cudaMalloc(&d_w, nop * sizeof(FPpart));
//     cudaMalloc(&d_q, nop * sizeof(FPpart));
//     cudaMalloc(&d_weight, 8 * nop * sizeof(FPpart));

//     cudaMalloc(&d_XN_flat, nxn * nyn * nzn * sizeof(FPpart));
//     cudaMalloc(&d_YN_flat, nxn * nyn * nzn * sizeof(FPpart));
//     cudaMalloc(&d_ZN_flat, nxn * nyn * nzn * sizeof(FPpart));
//     cudaMalloc(&d_rhon_flat, nxn * nyn * nzn * sizeof(FPpart));
//     cudaMalloc(&d_Jx_flat, nxn * nyn * nzn * sizeof(FPpart));
//     cudaMalloc(&d_Jy_flat, nxn * nyn * nzn * sizeof(FPpart));
//     cudaMalloc(&d_Jz_flat, nxn * nyn * nzn * sizeof(FPpart));
//     cudaMalloc(&d_pxx_flat, nxn * nyn * nzn * sizeof(FPpart));
//     cudaMalloc(&d_pxy_flat, nxn * nyn * nzn * sizeof(FPpart));
//     cudaMalloc(&d_pxz_flat, nxn * nyn * nzn * sizeof(FPpart));
//     cudaMalloc(&d_pyy_flat, nxn * nyn * nzn * sizeof(FPpart));
//     cudaMalloc(&d_pyz_flat, nxn * nyn * nzn * sizeof(FPpart));
//     cudaMalloc(&d_pzz_flat, nxn * nyn * nzn * sizeof(FPpart));

//     // Copy data from host to device
//     cudaMemcpy(d_x, part->x, nop * sizeof(FPpart), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_y, part->y, nop * sizeof(FPpart), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_z, part->z, nop * sizeof(FPpart), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_u, part->u, nop * sizeof(FPpart), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_v, part->v, nop * sizeof(FPpart), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_w, part->w, nop * sizeof(FPpart), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_q, part->q, nop * sizeof(FPpart), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_XN_flat, XN_flat, nxn * nyn * nzn * sizeof(FPpart),
//     cudaMemcpyHostToDevice); cudaMemcpy(d_YN_flat, YN_flat, nxn * nyn * nzn *
//     sizeof(FPpart), cudaMemcpyHostToDevice); cudaMemcpy(d_ZN_flat, ZN_flat,
//     nxn * nyn * nzn * sizeof(FPpart), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_rhon_flat, rhon_flat, nxn * nyn * nzn * sizeof(FPpart),
//     cudaMemcpyHostToDevice); cudaMemcpy(d_Jx_flat, Jx_flat, nxn * nyn * nzn *
//     sizeof(FPpart), cudaMemcpyHostToDevice); cudaMemcpy(d_Jy_flat, Jy_flat,
//     nxn * nyn * nzn * sizeof(FPpart), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_Jz_flat, Jz_flat, nxn * nyn * nzn * sizeof(FPpart),
//     cudaMemcpyHostToDevice); cudaMemcpy(d_pxx_flat, pxx_flat, nxn * nyn * nzn
//     * sizeof(FPpart), cudaMemcpyHostToDevice); cudaMemcpy(d_pxy_flat,
//     pxy_flat, nxn * nyn * nzn * sizeof(FPpart), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_pxz_flat, pxz_flat, nxn * nyn * nzn * sizeof(FPpart),
//     cudaMemcpyHostToDevice); cudaMemcpy(d_pyy_flat, pyy_flat, nxn * nyn * nzn
//     * sizeof(FPpart), cudaMemcpyHostToDevice); cudaMemcpy(d_pyz_flat,
//     pyz_flat, nxn * nyn * nzn * sizeof(FPpart), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_pzz_flat, pzz_flat, nxn * nyn * nzn * sizeof(FPpart),
//     cudaMemcpyHostToDevice);

//     int threadsPerBlock = 256;
//     int blocksPerGrid = (part->nop + threadsPerBlock - 1) / threadsPerBlock;

//     interpP2G_kernel<<<blocksPerGrid, threadsPerBlock>>>(
//         d_x, d_y, d_z,
//         d_u, d_v, d_w,
//         d_q,
//         d_XN_flat, d_YN_flat, d_ZN_flat,
//         d_rhon_flat,
//         d_Jx_flat, d_Jy_flat, d_Jz_flat,
//         d_pxx_flat, d_pxy_flat, d_pxz_flat,
//         d_pyy_flat, d_pyz_flat, d_pzz_flat,
//         grd->invVOL, grd->xStart, grd->yStart, grd->zStart,
//         grd->invdx, grd->invdy, grd->invdz,
//         nxn, nyn, nzn, nop
//       );

//     // Set grid and block dimensions
//     // dim3 blockDim(8, 8, 8);
//     // dim3 gridDim((nxn + 7) / 8, (nyn + 7) / 8, (nzn + 7) / 8);
//     // int shared_mem_size = (8 + 2) * (8 + 2) * (8 + 2) * sizeof(FPpart) *
//     10;
//     // interpP2G_kernel_3D<<<gridDim, blockDim, shared_mem_size>>>(
//     //     d_x, d_y, d_z,
//     //     d_u, d_v, d_w,
//     //     d_q,
//     //     d_XN_flat, d_YN_flat, d_ZN_flat,
//     //     d_rhon_flat,
//     //     d_Jx_flat, d_Jy_flat, d_Jz_flat,
//     //     d_pxx_flat, d_pxy_flat, d_pxz_flat,
//     //     d_pyy_flat, d_pyz_flat, d_pzz_flat,
//     //     grd->invVOL, grd->xStart, grd->yStart, grd->zStart,
//     //     grd->invdx, grd->invdy, grd->invdz,
//     //     nxn, nyn, nzn, nop
//     // );

//     cudaDeviceSynchronize();

//     // Copy back to host
//     cudaMemcpy(rhon_flat, d_rhon_flat, nxn * nyn * nzn * sizeof(FPpart),
//     cudaMemcpyDeviceToHost);
// cudaMemcpy(Jx_flat, d_Jx_flat, nxn * nyn * nzn * sizeof(FPpart),
// cudaMemcpyDeviceToHost); cudaMemcpy(Jy_flat, d_Jy_flat, nxn * nyn * nzn *
// sizeof(FPpart), cudaMemcpyDeviceToHost); cudaMemcpy(Jz_flat, d_Jz_flat, nxn *
// nyn * nzn * sizeof(FPpart), cudaMemcpyDeviceToHost); cudaMemcpy(pxx_flat,
// d_pxx_flat, nxn * nyn * nzn * sizeof(FPpart), cudaMemcpyDeviceToHost);
// cudaMemcpy(pxy_flat, d_pxy_flat, nxn * nyn * nzn * sizeof(FPpart),
// cudaMemcpyDeviceToHost); cudaMemcpy(pxz_flat, d_pxz_flat, nxn * nyn * nzn *
// sizeof(FPpart), cudaMemcpyDeviceToHost); cudaMemcpy(pyy_flat, d_pyy_flat, nxn
// * nyn * nzn * sizeof(FPpart), cudaMemcpyDeviceToHost); cudaMemcpy(pyz_flat,
// d_pyz_flat, nxn * nyn * nzn * sizeof(FPpart), cudaMemcpyDeviceToHost);
// cudaMemcpy(pzz_flat, d_pzz_flat, nxn * nyn * nzn * sizeof(FPpart),
// cudaMemcpyDeviceToHost);

//     // Write to original array and unflatten
// unflattenArray(rhon_flat, ids->rhon, nxn, nyn, nzn);
// unflattenArray(Jx_flat, ids->Jx, nxn, nyn, nzn);
// unflattenArray(Jy_flat, ids->Jy, nxn, nyn, nzn);
// unflattenArray(Jz_flat, ids->Jz, nxn, nyn, nzn);
// unflattenArray(pxx_flat, ids->pxx, nxn, nyn, nzn);
// unflattenArray(pxy_flat, ids->pxy, nxn, nyn, nzn);
// unflattenArray(pxz_flat, ids->pxz, nxn, nyn, nzn);
// unflattenArray(pyy_flat, ids->pyy, nxn, nyn, nzn);
// unflattenArray(pyz_flat, ids->pyz, nxn, nyn, nzn);
// unflattenArray(pzz_flat, ids->pzz, nxn, nyn, nzn);

//     // Free cuda arrays
//     cudaFree(d_x);
//     cudaFree(d_y);
//     cudaFree(d_z);
//     cudaFree(d_u);
//     cudaFree(d_v);
//     cudaFree(d_w);
//     cudaFree(d_q);
//     cudaFree(d_XN_flat);
//     cudaFree(d_YN_flat);
//     cudaFree(d_ZN_flat);
//     cudaFree(d_rhon_flat);
//     cudaFree(d_Jx_flat);
//     cudaFree(d_Jy_flat);
//     cudaFree(d_Jz_flat);
//     cudaFree(d_pxx_flat);
//     cudaFree(d_pxy_flat);
//     cudaFree(d_pxz_flat);
//     cudaFree(d_pyy_flat);
//     cudaFree(d_pyz_flat);
//     cudaFree(d_pzz_flat);
//     cudaFree(d_weight);

//     // Free the arrays
//     free(XN_flat);
//     free(YN_flat);
//     free(ZN_flat);
//     free(rhon_flat);
//     free(Jx_flat);
//     free(Jy_flat);
//     free(Jz_flat);
//     free(pxx_flat);
//     free(pxy_flat);
//     free(pxz_flat);
//     free(pyy_flat);
//     free(pyz_flat);
//     free(pzz_flat);
// }

// // V1 - not working cba
// void interpP2G_GPU_v1(struct particles* part, struct interpDensSpecies* ids,
// struct grid* grd) {
//     // Extract dimensions
//     int nxn = grd->nxn;
//     int nyn = grd->nyn;
//     int nzn = grd->nzn;
//     int nop = part->nop;

//     // Flatten arrays on the host
//     FPpart* XN_flat = flattenArray(grd->XN, nxn, nyn, nzn);
//     FPpart* YN_flat = flattenArray(grd->YN, nxn, nyn, nzn);
//     FPpart* ZN_flat = flattenArray(grd->ZN, nxn, nyn, nzn);
//     FPpart* rhon_flat = flattenArray(ids->rhon, nxn, nyn, nzn);
//     FPpart* Jx_flat = flattenArray(ids->Jx, nxn, nyn, nzn);
//     FPpart* Jy_flat = flattenArray(ids->Jy, nxn, nyn, nzn);
//     FPpart* Jz_flat = flattenArray(ids->Jz, nxn, nyn, nzn);
//     FPpart* pxx_flat = flattenArray(ids->pxx, nxn, nyn, nzn);
//     FPpart* pxy_flat = flattenArray(ids->pxy, nxn, nyn, nzn);
//     FPpart* pxz_flat = flattenArray(ids->pxz, nxn, nyn, nzn);
//     FPpart* pyy_flat = flattenArray(ids->pyy, nxn, nyn, nzn);
//     FPpart* pyz_flat = flattenArray(ids->pyz, nxn, nyn, nzn);
//     FPpart* pzz_flat = flattenArray(ids->pzz, nxn, nyn, nzn);

//     // Device pointers
//     FPpart *d_x, *d_y, *d_z, *d_u, *d_v, *d_w, *d_q, *d_weight;
//     FPpart *d_XN_flat, *d_YN_flat, *d_ZN_flat;
//     FPpart *d_rhon_flat, *d_Jx_flat, *d_Jy_flat, *d_Jz_flat;
//     FPpart *d_pxx_flat, *d_pxy_flat, *d_pxz_flat, *d_pyy_flat, *d_pyz_flat,
//     *d_pzz_flat;

//     // Allocate memory on device
//     cudaMalloc(&d_x, nop * sizeof(FPpart));
//     cudaMalloc(&d_y, nop * sizeof(FPpart));
//     cudaMalloc(&d_z, nop * sizeof(FPpart));
//     cudaMalloc(&d_u, nop * sizeof(FPpart));
//     cudaMalloc(&d_v, nop * sizeof(FPpart));
//     cudaMalloc(&d_w, nop * sizeof(FPpart));
//     cudaMalloc(&d_q, nop * sizeof(FPpart));
//     cudaMalloc(&d_weight, 8 * nop * sizeof(FPpart));

//     cudaMalloc(&d_XN_flat, nxn * nyn * nzn * sizeof(FPpart));
//     cudaMalloc(&d_YN_flat, nxn * nyn * nzn * sizeof(FPpart));
//     cudaMalloc(&d_ZN_flat, nxn * nyn * nzn * sizeof(FPpart));
//     cudaMalloc(&d_rhon_flat, nxn * nyn * nzn * sizeof(FPpart));
//     cudaMalloc(&d_Jx_flat, nxn * nyn * nzn * sizeof(FPpart));
//     cudaMalloc(&d_Jy_flat, nxn * nyn * nzn * sizeof(FPpart));
//     cudaMalloc(&d_Jz_flat, nxn * nyn * nzn * sizeof(FPpart));
//     cudaMalloc(&d_pxx_flat, nxn * nyn * nzn * sizeof(FPpart));
//     cudaMalloc(&d_pxy_flat, nxn * nyn * nzn * sizeof(FPpart));
//     cudaMalloc(&d_pxz_flat, nxn * nyn * nzn * sizeof(FPpart));
//     cudaMalloc(&d_pyy_flat, nxn * nyn * nzn * sizeof(FPpart));
//     cudaMalloc(&d_pyz_flat, nxn * nyn * nzn * sizeof(FPpart));
//     cudaMalloc(&d_pzz_flat, nxn * nyn * nzn * sizeof(FPpart));

//     // Copy grid data to device
//     cudaMemcpy(d_XN_flat, XN_flat, nxn * nyn * nzn * sizeof(FPpart),
//     cudaMemcpyHostToDevice); cudaMemcpy(d_YN_flat, YN_flat, nxn * nyn * nzn *
//     sizeof(FPpart), cudaMemcpyHostToDevice); cudaMemcpy(d_ZN_flat, ZN_flat,
//     nxn * nyn * nzn * sizeof(FPpart), cudaMemcpyHostToDevice);

//     // Setup streams
//     const int numStreams = 1;
//     cudaStream_t streams[numStreams];
//     for (int i = 0; i < numStreams; ++i) {
//         cudaStreamCreate(&streams[i]);
//     }

//     // Chunking particles for streams
//     int chunkSize = (nop + numStreams - 1) / numStreams;
//     int threadsPerBlock = 256;

//     // Loop through streams
//     for (int i = 0; i < numStreams; ++i) {
//         int offset = i * chunkSize;
//         int currentChunkSize = std::min(chunkSize, nop - offset);

//         // Asynchronous memory copy to device
//         cudaMemcpyAsync(&d_data[offset].x, &part->data[offset].x,
//         currentChunkSize * sizeof(FPpart), cudaMemcpyHostToDevice,
//         streams[i]); cudaMemcpyAsync(&d_data[offset].y,
//         &part->data[offset].y, currentChunkSize * sizeof(FPpart),
//         cudaMemcpyHostToDevice, streams[i]);
//         cudaMemcpyAsync(&d_data[offset].z, &part->data[offset].z,
//         currentChunkSize * sizeof(FPpart), cudaMemcpyHostToDevice,
//         streams[i]); cudaMemcpyAsync(&d_data[offset].u,
//         &part->data[offset].u, currentChunkSize * sizeof(FPpart),
//         cudaMemcpyHostToDevice, streams[i]);
//         cudaMemcpyAsync(&d_data[offset].v, &part->data[offset].v,
//         currentChunkSize * sizeof(FPpart), cudaMemcpyHostToDevice,
//         streams[i]); cudaMemcpyAsync(&d_data[offset].w,
//         &part->data[offset].w, currentChunkSize * sizeof(FPpart),
//         cudaMemcpyHostToDevice, streams[i]);
//         cudaMemcpyAsync(&d_data[offset].q, &part->data[offset].q,
//         currentChunkSize * sizeof(FPpart), cudaMemcpyHostToDevice,
//         streams[i]);

//         // Launch kernel
//         int blocksPerGrid = (currentChunkSize + threadsPerBlock - 1) /
//         threadsPerBlock; interpP2G_kernel<<<blocksPerGrid, threadsPerBlock,
//         0, streams[i]>>>(
//             &d_data[offset].x, &d_data[offset].y, &d_data[offset].z,
//             &d_data[offset].u, &d_data[offset].v, &d_data[offset].w,
//             &d_data[offset].q,
//             d_XN_flat, d_YN_flat, d_ZN_flat,
//             d_rhon_flat, d_Jx_flat, d_Jy_flat, d_Jz_flat,
//             d_pxx_flat, d_pxy_flat, d_pxz_flat,
//             d_pyy_flat, d_pyz_flat, d_pzz_flat,
//             grd->invVOL, grd->xStart, grd->yStart, grd->zStart,
//             grd->invdx, grd->invdy, grd->invdz,
//             nxn, nyn, nzn, currentChunkSize
//         );

//         // Asynchronous memory copy back to host
//         cudaMemcpyAsync(&ids->rhon_flat[offset], &d_rhon_flat[offset],
//         currentChunkSize * sizeof(FPpart), cudaMemcpyDeviceToHost,
//         streams[i]); }

//     // Synchronize streams
//     for (int i = 0; i < numStreams; ++i) {
//         cudaStreamSynchronize(streams[i]);
//         cudaStreamDestroy(streams[i]);
//     }

//     cudaMemcpy(Jx_flat, d_Jx_flat, nxn * nyn * nzn * sizeof(FPpart),
//     cudaMemcpyDeviceToHost); cudaMemcpy(Jy_flat, d_Jy_flat, nxn * nyn * nzn *
//     sizeof(FPpart), cudaMemcpyDeviceToHost); cudaMemcpy(Jz_flat, d_Jz_flat,
//     nxn * nyn * nzn * sizeof(FPpart), cudaMemcpyDeviceToHost);
//     cudaMemcpy(pxx_flat, d_pxx_flat, nxn * nyn * nzn * sizeof(FPpart),
//     cudaMemcpyDeviceToHost); cudaMemcpy(pxy_flat, d_pxy_flat, nxn * nyn * nzn
//     * sizeof(FPpart), cudaMemcpyDeviceToHost); cudaMemcpy(pxz_flat,
//     d_pxz_flat, nxn * nyn * nzn * sizeof(FPpart), cudaMemcpyDeviceToHost);
//     cudaMemcpy(pyy_flat, d_pyy_flat, nxn * nyn * nzn * sizeof(FPpart),
//     cudaMemcpyDeviceToHost); cudaMemcpy(pyz_flat, d_pyz_flat, nxn * nyn * nzn
//     * sizeof(FPpart), cudaMemcpyDeviceToHost); cudaMemcpy(pzz_flat,
//     d_pzz_flat, nxn * nyn * nzn * sizeof(FPpart), cudaMemcpyDeviceToHost);

//     // Unflatten arrays back to their original shapes on the host
//     unflattenArray(rhon_flat, ids->rhon, nxn, nyn, nzn);
//     unflattenArray(Jx_flat, ids->Jx, nxn, nyn, nzn);
//     unflattenArray(Jy_flat, ids->Jy, nxn, nyn, nzn);
//     unflattenArray(Jz_flat, ids->Jz, nxn, nyn, nzn);
//     unflattenArray(pxx_flat, ids->pxx, nxn, nyn, nzn);
//     unflattenArray(pxy_flat, ids->pxy, nxn, nyn, nzn);
//     unflattenArray(pxz_flat, ids->pxz, nxn, nyn, nzn);
//     unflattenArray(pyy_flat, ids->pyy, nxn, nyn, nzn);
//     unflattenArray(pyz_flat, ids->pyz, nxn, nyn, nzn);
//     unflattenArray(pzz_flat, ids->pzz, nxn, nyn, nzn);

//     // Free device memory
//     cudaFree(d_x);
//     cudaFree(d_y);
//     cudaFree(d_z);
//     cudaFree(d_u);
//     cudaFree(d_v);
//     cudaFree(d_w);
//     cudaFree(d_q);
//     cudaFree(d_XN_flat);
//     cudaFree(d_YN_flat);
//     cudaFree(d_ZN_flat);
//     cudaFree(d_rhon_flat);
//     cudaFree(d_Jx_flat);
//     cudaFree(d_Jy_flat);
//     cudaFree(d_Jz_flat);
//     cudaFree(d_pxx_flat);
//     cudaFree(d_pxy_flat);
//     cudaFree(d_pxz_flat);
//     cudaFree(d_pyy_flat);
//     cudaFree(d_pyz_flat);
//     cudaFree(d_pzz_flat);
// }

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

// GPU Mover Time / Cycle   (s) = 0.250327
//    CPU Mover Time / Cycle   (s) = 2.89693