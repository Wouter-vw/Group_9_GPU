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

__global__ void mover_PC_kernel(
    FPpart* d_x, FPpart* d_y, FPpart* d_z, FPpart* d_u, FPpart* d_v,
    FPpart* d_w, FPpart* d_XN_flat, FPpart* d_YN_flat, FPpart* d_ZN_flat,
    FPfield* d_Ex_flat, FPfield* d_Ey_flat, FPfield* d_Ez_flat,
    FPfield* d_Bxn_flat, FPfield* d_Byn_flat, FPfield* d_Bzn_flat,
    FPpart d_invVOL, FPpart d_xStart, FPpart d_yStart, FPpart d_zStart,
    FPpart d_invdx, FPpart d_invdy, FPpart d_invdz, FPpart d_Lx, FPpart d_Ly,
    FPpart d_Lz, FPpart dt_sub_cycling, FPpart dto2, FPpart qomdt2,
    bool d_PERIODICX, bool d_PERIODICY, bool d_PERIODICZ, int d_nxn, int d_nyn,
    int d_nzn, int d_nop, int d_NiterMover) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  FPpart omdtsq, denom, ut, vt, wt, udotb;

  // local (to the particle) electric and magnetic field
  FPfield Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

  // interpolation densities
  int ix, iy, iz;
  FPfield weight[2][2][2];
  FPfield xi[2], eta[2], zeta[2];

  // intermediate particle position and velocity
  FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

  if (i < d_nop) {
    xptilde = d_x[i];
    yptilde = d_y[i];
    zptilde = d_z[i];

    for (int inner = 0; inner < d_NiterMover; inner++) {
      // Interpolation G-->P
      ix = 2 + int((d_x[i] - d_xStart) * d_invdx);
      iy = 2 + int((d_y[i] - d_yStart) * d_invdy);
      iz = 2 + int((d_z[i] - d_zStart) * d_invdz);

      // Check indixing
      xi[0] =
          d_x[i] - d_XN_flat[(ix - 1) * (d_nyn * d_nzn) + (iy)*d_nzn + (iz)];
      eta[0] = d_y[i] - d_YN_flat[ix * d_nyn * d_nzn + (iy - 1) * d_nzn + (iz)];
      zeta[0] = d_z[i] - d_ZN_flat[ix * d_nyn * d_nzn + (iy)*d_nzn + (iz - 1)];

      xi[1] = d_XN_flat[ix * d_nyn * d_nzn + (iy)*d_nzn + (iz)] - d_x[i];
      eta[1] = d_YN_flat[ix * d_nyn * d_nzn + (iy)*d_nzn + (iz)] - d_y[i];
      zeta[1] = d_ZN_flat[ix * d_nyn * d_nzn + (iy)*d_nzn + (iz)] - d_z[i];

      for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 2; jj++) {
          for (int kk = 0; kk < 2; kk++) {
            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * d_invVOL;
          }
        }
      }

      Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

      for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 2; jj++) {
          for (int kk = 0; kk < 2; kk++) {
            // Check idx index
            int idx = (ix - ii) * d_nyn * d_nzn + (iy - jj) * d_nzn + (iz - kk);
            Exl += weight[ii][jj][kk] * d_Ex_flat[idx];
            Eyl += weight[ii][jj][kk] * d_Ey_flat[idx];
            Ezl += weight[ii][jj][kk] * d_Ez_flat[idx];
            Bxl += weight[ii][jj][kk] * d_Bxn_flat[idx];
            Byl += weight[ii][jj][kk] * d_Byn_flat[idx];
            Bzl += weight[ii][jj][kk] * d_Bzn_flat[idx];
          }
        }
      }

      // Particle motion equations
      omdtsq = qomdt2 * qomdt2 * (Bxl * Bxl + Byl * Byl + Bzl * Bzl);
      denom = 1.0 / (1.0 + omdtsq);

      ut = d_u[i] + qomdt2 * Exl;
      vt = d_v[i] + qomdt2 * Eyl;
      wt = d_w[i] + qomdt2 * Ezl;
      udotb = ut * Bxl + vt * Byl + wt * Bzl;

      uptilde =
          (ut + qomdt2 * (vt * Bzl - wt * Byl + qomdt2 * udotb * Bxl)) * denom;
      vptilde =
          (vt + qomdt2 * (wt * Bxl - ut * Bzl + qomdt2 * udotb * Byl)) * denom;
      wptilde =
          (wt + qomdt2 * (ut * Byl - vt * Bxl + qomdt2 * udotb * Bzl)) * denom;

      d_x[i] = xptilde + uptilde * dto2;
      d_y[i] = yptilde + vptilde * dto2;
      d_z[i] = zptilde + wptilde * dto2;
    }

    // Update the final position and velocity
    d_u[i] = 2.0 * uptilde - d_u[i];
    d_v[i] = 2.0 * vptilde - d_v[i];
    d_w[i] = 2.0 * wptilde - d_w[i];
    // update position
    d_x[i] = xptilde + uptilde * dt_sub_cycling;
    d_y[i] = yptilde + vptilde * dt_sub_cycling;
    d_z[i] = zptilde + wptilde * dt_sub_cycling;

    // Boundary conditions
    if (d_x[i] > d_Lx) {
      if (d_PERIODICX) {
        d_x[i] -= d_Lx;
      } else {
        d_u[i] = -d_u[i];
        d_x[i] = 2 * d_Lx - d_x[i];
      }
    }
    if (d_x[i] < 0) {
      if (d_PERIODICX) {
        d_x[i] += d_Lx;
      } else {
        d_u[i] = -d_u[i];
        d_x[i] = -d_x[i];
      }
    }
    if (d_y[i] > d_Ly) {
      if (d_PERIODICY) {
        d_y[i] -= d_Ly;
      } else {
        d_v[i] = -d_v[i];
        d_y[i] = 2 * d_Ly - d_y[i];
      }
    }
    if (d_y[i] < 0) {
      if (d_PERIODICY) {
        d_y[i] += d_Ly;
      } else {
        d_v[i] = -d_v[i];
        d_y[i] = -d_y[i];
      }
    }
    if (d_z[i] > d_Lz) {
      if (d_PERIODICZ) {
        d_z[i] -= d_Lz;
      } else {
        d_w[i] = -d_w[i];
        d_z[i] = 2 * d_Lz - d_z[i];
      }
    }
    if (d_z[i] < 0) {
      if (d_PERIODICZ) {
        d_z[i] += d_Lz;
      } else {
        d_w[i] = -d_w[i];
        d_z[i] = -d_z[i];
      }
    }
  }
}

int mover_PC_GPU(struct particles* part, struct EMfield* field,
                 struct grid* grd, struct parameters* param) {
  // print species and subcycling
  // std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " -
  // species " << part->species_ID << " ***" << std::endl;

  // auxiliary variables
  FPpart dt_sub_cycling = (FPpart)param->dt / ((double)part->n_sub_cycles);
  FPpart dto2 = .5 * dt_sub_cycling, qomdt2 = part->qom * dto2 / param->c;

  // corresponds to XN, YN etc. also Ex, Bxn COPY for slimpicity
  int nxn = grd->nxn;
  int nyn = grd->nyn;
  int nzn = grd->nzn;
  int nop = part->nop;

  // Device pointers -> Free at the end
  FPpart *d_x, *d_y, *d_z;
  FPpart *d_u, *d_v, *d_w;
  FPpart *d_XN_flat, *d_YN_flat, *d_ZN_flat;
  FPfield *d_Ex_flat, *d_Ey_flat, *d_Ez_flat, *d_Bxn_flat, *d_Byn_flat,
      *d_Bzn_flat;

  // Allocate memory on the device
  cudaMalloc(&d_x, nop * sizeof(FPpart));
  cudaMalloc(&d_y, nop * sizeof(FPpart));
  cudaMalloc(&d_z, nop * sizeof(FPpart));
  cudaMalloc(&d_u, nop * sizeof(FPpart));
  cudaMalloc(&d_v, nop * sizeof(FPpart));
  cudaMalloc(&d_w, nop * sizeof(FPpart));
  cudaMalloc(&d_XN_flat, nxn * nyn * nzn * sizeof(FPpart));
  cudaMalloc(&d_YN_flat, nxn * nyn * nzn * sizeof(FPpart));
  cudaMalloc(&d_ZN_flat, nxn * nyn * nzn * sizeof(FPpart));
  cudaMalloc(&d_Ex_flat, nxn * nyn * nzn * sizeof(FPfield));
  cudaMalloc(&d_Ey_flat, nxn * nyn * nzn * sizeof(FPfield));
  cudaMalloc(&d_Ez_flat, nxn * nyn * nzn * sizeof(FPfield));
  cudaMalloc(&d_Bxn_flat, nxn * nyn * nzn * sizeof(FPfield));
  cudaMalloc(&d_Byn_flat, nxn * nyn * nzn * sizeof(FPfield));
  cudaMalloc(&d_Bzn_flat, nxn * nyn * nzn * sizeof(FPfield));

  // Copy data from host to device
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

  // Initialize number of threads
  int threadsPerBlock = 256;
  int blocksPerGrid = (part->nop + threadsPerBlock - 1) / threadsPerBlock;

  // Loop over all sub cycles. Wait untill finished for a number of points
  // before you go to the next
  for (int i_sub = 0; i_sub < part->n_sub_cycles; i_sub++) {
    mover_PC_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_x, d_y, d_z, d_u, d_v, d_w, d_XN_flat, d_YN_flat, d_ZN_flat,
        d_Ex_flat, d_Ey_flat, d_Ez_flat, d_Bxn_flat, d_Byn_flat, d_Bzn_flat,
        grd->invVOL, grd->xStart, grd->yStart, grd->zStart, grd->invdx,
        grd->invdy, grd->invdz, param->Lx, param->Ly, param->Lz, dt_sub_cycling,
        dto2, qomdt2, param->PERIODICX, param->PERIODICY, param->PERIODICZ, nxn,
        nyn, nzn, nop, part->NiterMover);
    cudaDeviceSynchronize();
  }

  // Copy back to CPU
  cudaMemcpy(part->x, d_x, nop * sizeof(FPpart), cudaMemcpyDeviceToHost);
  cudaMemcpy(part->y, d_y, nop * sizeof(FPpart), cudaMemcpyDeviceToHost);
  cudaMemcpy(part->z, d_z, nop * sizeof(FPpart), cudaMemcpyDeviceToHost);
  cudaMemcpy(part->u, d_u, nop * sizeof(FPpart), cudaMemcpyDeviceToHost);
  cudaMemcpy(part->v, d_v, nop * sizeof(FPpart), cudaMemcpyDeviceToHost);
  cudaMemcpy(part->w, d_w, nop * sizeof(FPpart), cudaMemcpyDeviceToHost);

  // Free memory
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
