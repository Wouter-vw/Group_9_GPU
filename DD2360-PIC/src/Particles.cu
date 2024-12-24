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
  part->x = new FPpart[npmax];
  part->y = new FPpart[npmax];
  part->z = new FPpart[npmax];
  // allocate velocity
  part->u = new FPpart[npmax];
  part->v = new FPpart[npmax];
  part->w = new FPpart[npmax];
  // allocate charge = q * statistical weight
  part->q = new FPinterp[npmax];
}

/** deallocate */
void particle_deallocate(struct particles* part) {
  // deallocate particle variables
  delete[] part->x;
  delete[] part->y;
  delete[] part->z;
  delete[] part->u;
  delete[] part->v;
  delete[] part->w;
  delete[] part->q;
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
  for (int i_sub = 0; i_sub < part->n_sub_cycles; i_sub++) {
    xptilde = part->x[i];
    yptilde = part->y[i];
    zptilde = part->z[i];
    // calculate the average velocity iteratively
    for (int innter = 0; innter < part->NiterMover; innter++) {
      // interpolation G-->P
      ix = 2 + int((part->x[i] - grd->xStart) * grd->invdx);
      iy = 2 + int((part->y[i] - grd->yStart) * grd->invdy);
      iz = 2 + int((part->z[i] - grd->zStart) * grd->invdz);

      // calculate weights
      xi[0] = part->x[i] - grd->XN[ix - 1][iy][iz];
      eta[0] = part->y[i] - grd->YN[ix][iy - 1][iz];
      zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
      xi[1] = grd->XN[ix][iy][iz] - part->x[i];
      eta[1] = grd->YN[ix][iy][iz] - part->y[i];
      zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
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
      ut = part->u[i] + qomdt2 * Exl;
      vt = part->v[i] + qomdt2 * Eyl;
      wt = part->w[i] + qomdt2 * Ezl;
      udotb = ut * Bxl + vt * Byl + wt * Bzl;
      // solve the velocity equation
      uptilde =
          (ut + qomdt2 * (vt * Bzl - wt * Byl + qomdt2 * udotb * Bxl)) * denom;
      vptilde =
          (vt + qomdt2 * (wt * Bxl - ut * Bzl + qomdt2 * udotb * Byl)) * denom;
      wptilde =
          (wt + qomdt2 * (ut * Byl - vt * Bxl + qomdt2 * udotb * Bzl)) * denom;
      // update position
      part->x[i] = xptilde + uptilde * dto2;
      part->y[i] = yptilde + vptilde * dto2;
      part->z[i] = zptilde + wptilde * dto2;
    }  // end of iteration
    // update the final position and velocity
    part->u[i] = 2.0 * uptilde - part->u[i];
    part->v[i] = 2.0 * vptilde - part->v[i];
    part->w[i] = 2.0 * wptilde - part->w[i];
    part->x[i] = xptilde + uptilde * dt_sub_cycling;
    part->y[i] = yptilde + vptilde * dt_sub_cycling;
    part->z[i] = zptilde + wptilde * dt_sub_cycling;

    //////////
    //////////
    ////////// BC

    // X-DIRECTION: BC particles
    if (part->x[i] > grd->Lx) {
      if (param->PERIODICX == true) {
        // PERIODIC
        part->x[i] = part->x[i] - grd->Lx;
      } else {
        // REFLECTING BC
        part->u[i] = -part->u[i];
        part->x[i] = 2 * grd->Lx - part->x[i];
      }
    }

    if (part->x[i] < 0) {
      if (param->PERIODICX == true) {
        // PERIODIC
        part->x[i] = part->x[i] + grd->Lx;
      } else {
        // REFLECTING BC
        part->u[i] = -part->u[i];
        part->x[i] = -part->x[i];
      }
    }

    // Y-DIRECTION: BC particles
    if (part->y[i] > grd->Ly) {
      if (param->PERIODICY == true) {
        // PERIODIC
        part->y[i] = part->y[i] - grd->Ly;
      } else {
        // REFLECTING BC
        part->v[i] = -part->v[i];
        part->y[i] = 2 * grd->Ly - part->y[i];
      }
    }

    if (part->y[i] < 0) {
      if (param->PERIODICY == true) {
        // PERIODIC
        part->y[i] = part->y[i] + grd->Ly;
      } else {
        // REFLECTING BC
        part->v[i] = -part->v[i];
        part->y[i] = -part->y[i];
      }
    }

    // Z-DIRECTION: BC particles
    if (part->z[i] > grd->Lz) {
      if (param->PERIODICZ == true) {
        // PERIODIC
        part->z[i] = part->z[i] - grd->Lz;
      } else {
        // REFLECTING BC
        part->w[i] = -part->w[i];
        part->z[i] = 2 * grd->Lz - part->z[i];
      }
    }

    if (part->z[i] < 0) {
      if (param->PERIODICZ == true) {
        // PERIODIC
        part->z[i] = part->z[i] + grd->Lz;
      } else {
        // REFLECTING BC
        part->w[i] = -part->w[i];
        part->z[i] = -part->z[i];
      }
    }
  }  // end of one particle

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
    xptilde = part->x[i];
    yptilde = part->y[i];
    zptilde = part->z[i];

    for (int inner = 0; inner < part->NiterMover; inner++) {
      // Interpolation G-->P
      ix = 2 + int((part->x[i] - grd->xStart) * grd->invdx);
      iy = 2 + int((part->y[i] - grd->yStart) * grd->invdy);
      iz = 2 + int((part->z[i] - grd->zStart) * grd->invdz);

      // Check indixing
      xi[0] =
          part->x[i] -
          grd->XN_flat[(ix - 1) * (grd->nyn * grd->nzn) + (iy)*grd->nzn + (iz)];
      eta[0] =
          part->y[i] -
          grd->YN_flat[ix * grd->nyn * grd->nzn + (iy - 1) * grd->nzn + (iz)];
      zeta[0] =
          part->z[i] -
          grd->ZN_flat[ix * grd->nyn * grd->nzn + (iy)*grd->nzn + (iz - 1)];

      xi[1] = grd->XN_flat[ix * grd->nyn * grd->nzn + (iy)*grd->nzn + (iz)] -
              part->x[i];
      eta[1] = grd->YN_flat[ix * grd->nyn * grd->nzn + (iy)*grd->nzn + (iz)] -
               part->y[i];
      zeta[1] = grd->ZN_flat[ix * grd->nyn * grd->nzn + (iy)*grd->nzn + (iz)] -
                part->z[i];

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

      ut = part->u[i] + qomdt2 * Exl;
      vt = part->v[i] + qomdt2 * Eyl;
      wt = part->w[i] + qomdt2 * Ezl;
      udotb = ut * Bxl + vt * Byl + wt * Bzl;

      uptilde =
          (ut + qomdt2 * (vt * Bzl - wt * Byl + qomdt2 * udotb * Bxl)) * denom;
      vptilde =
          (vt + qomdt2 * (wt * Bxl - ut * Bzl + qomdt2 * udotb * Byl)) * denom;
      wptilde =
          (wt + qomdt2 * (ut * Byl - vt * Bxl + qomdt2 * udotb * Bzl)) * denom;

      part->x[i] = xptilde + uptilde * dto2;
      part->y[i] = yptilde + vptilde * dto2;
      part->z[i] = zptilde + wptilde * dto2;
    }

    // Update the final position and velocity
    part->u[i] = 2.0 * uptilde - part->u[i];
    part->v[i] = 2.0 * vptilde - part->v[i];
    part->w[i] = 2.0 * wptilde - part->w[i];
    // update position
    part->x[i] = xptilde + uptilde * dt_sub_cycling;
    part->y[i] = yptilde + vptilde * dt_sub_cycling;
    part->z[i] = zptilde + wptilde * dt_sub_cycling;

    // Boundary conditions
    if (part->x[i] > grd->Lx) {
      if (grd->PERIODICX) {
        part->x[i] -= grd->Lx;
      } else {
        part->u[i] = -part->u[i];
        part->x[i] = 2 * grd->Lx - part->x[i];
      }
    }
    if (part->x[i] < 0) {
      if (grd->PERIODICX) {
        part->x[i] += grd->Lx;
      } else {
        part->u[i] = -part->u[i];
        part->x[i] = -part->x[i];
      }
    }
    if (part->y[i] > grd->Ly) {
      if (grd->PERIODICY) {
        part->y[i] -= grd->Ly;
      } else {
        part->v[i] = -part->v[i];
        part->y[i] = 2 * grd->Ly - part->y[i];
      }
    }
    if (part->y[i] < 0) {
      if (grd->PERIODICY) {
        part->y[i] += grd->Ly;
      } else {
        part->v[i] = -part->v[i];
        part->y[i] = -part->y[i];
      }
    }
    if (part->z[i] > grd->Lz) {
      if (grd->PERIODICZ) {
        part->z[i] -= grd->Lz;
      } else {
        part->w[i] = -part->w[i];
        part->z[i] = 2 * grd->Lz - part->z[i];
      }
    }
    if (part->z[i] < 0) {
      if (grd->PERIODICZ) {
        part->z[i] += grd->Lz;
      } else {
        part->w[i] = -part->w[i];
        part->z[i] = -part->z[i];
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
  FPpart *d_x, *d_y, *d_z;
  FPpart *d_u, *d_v, *d_w;

  cudaMalloc(&d_x, nop * sizeof(FPpart));
  cudaMalloc(&d_y, nop * sizeof(FPpart));
  cudaMalloc(&d_z, nop * sizeof(FPpart));
  cudaMalloc(&d_u, nop * sizeof(FPpart));
  cudaMalloc(&d_v, nop * sizeof(FPpart));
  cudaMalloc(&d_w, nop * sizeof(FPpart));

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
  cudaMemcpy(d_x, part->x, nop * sizeof(FPpart), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, part->y, nop * sizeof(FPpart), cudaMemcpyHostToDevice);
  cudaMemcpy(d_z, part->z, nop * sizeof(FPpart), cudaMemcpyHostToDevice);
  cudaMemcpy(d_u, part->u, nop * sizeof(FPpart), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, part->v, nop * sizeof(FPpart), cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, part->w, nop * sizeof(FPpart), cudaMemcpyHostToDevice);

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
  temp_part.x = d_x;
  temp_part.y = d_y;
  temp_part.z = d_z;
  temp_part.u = d_u;
  temp_part.v = d_v;
  temp_part.w = d_w;

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
  cudaMemcpy(part->x, d_x, nop * sizeof(FPpart), cudaMemcpyDeviceToHost);
  cudaMemcpy(part->y, d_y, nop * sizeof(FPpart), cudaMemcpyDeviceToHost);
  cudaMemcpy(part->z, d_z, nop * sizeof(FPpart), cudaMemcpyDeviceToHost);
  cudaMemcpy(part->u, d_u, nop * sizeof(FPpart), cudaMemcpyDeviceToHost);
  cudaMemcpy(part->v, d_v, nop * sizeof(FPpart), cudaMemcpyDeviceToHost);
  cudaMemcpy(part->w, d_w, nop * sizeof(FPpart), cudaMemcpyDeviceToHost);

  // 9. Free device memory
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_w);
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
    ix = 2 + int(floor((part->x[i] - grd->xStart) * grd->invdx));
    iy = 2 + int(floor((part->y[i] - grd->yStart) * grd->invdy));
    iz = 2 + int(floor((part->z[i] - grd->zStart) * grd->invdz));

    // distances from node
    xi[0] = part->x[i] - grd->XN[ix - 1][iy][iz];
    eta[0] = part->y[i] - grd->YN[ix][iy - 1][iz];
    zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
    xi[1] = grd->XN[ix][iy][iz] - part->x[i];
    eta[1] = grd->YN[ix][iy][iz] - part->y[i];
    zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];

    // calculate the weights for different nodes
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          weight[ii][jj][kk] =
              part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;

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
          temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    ////////////////////////////
    // add current density - Jy
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    ////////////////////////////
    // add current density - Jz
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    ////////////////////////////
    // add pressure pxx
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    ////////////////////////////
    // add pressure pxy
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    /////////////////////////////
    // add pressure pxz
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    /////////////////////////////
    // add pressure pyy
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    /////////////////////////////
    // add pressure pyz
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    /////////////////////////////
    // add pressure pzz
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
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
