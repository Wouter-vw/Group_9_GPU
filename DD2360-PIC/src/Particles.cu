#include <cuda.h>
#include <cuda_runtime.h>

#include "Alloc.h"
#include "Particles.h"

FPpart* flattenArray(FPpart*** array3D, int nx, int ny, int nz) {
  // Allocate memory for the 1D array
  FPpart* array1D = (FPpart*)malloc(nx * ny * nz * sizeof(FPpart));

  // Flatten the 3D array into the 1D array
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      for (int k = 0; k < nz; k++) {
        int index = i * (ny * nz) + j * nz + k;
        array1D[index] = array3D[i][j][k];
      }
    }
  }

  return array1D;  // Return the flattened array
}

void unflattenArray(FPpart* array1D, FPpart*** array3D, int nx, int ny,
                    int nz) {
  // Populate the pre-allocated 3D array with values from the 1D array
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      for (int k = 0; k < nz; k++) {
        int index = i * (ny * nz) + j * nz + k;
        array3D[i][j][k] = array1D[index];
      }
    }
  }
}

FPfield* flattenArrayfield(FPfield*** array3D, int nx, int ny, int nz) {
  // Allocate memory for the 1D array
  FPfield* array1D = (FPfield*)malloc(nx * ny * nz * sizeof(FPfield));

  // Flatten the 3D array into the 1D array
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      for (int k = 0; k < nz; k++) {
        int index = i * (ny * nz) + j * nz + k;
        array1D[index] = array3D[i][j][k];
      }
    }
  }

  return array1D;  // Return the flattened array
}

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
  if (param->qom[is] < 0) {  // electrons
    part->NiterMover = param->NiterMover;
    part->n_sub_cycles = param->n_sub_cycles;
  } else {  // ions: only one iteration
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

/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd,
             struct parameters* param) {
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
    // move each particle with new fields
    for (int i = 0; i < part->nop; i++) {
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
        uptilde = (ut + qomdt2 * (vt * Bzl - wt * Byl + qomdt2 * udotb * Bxl)) *
                  denom;
        vptilde = (vt + qomdt2 * (wt * Bxl - ut * Bzl + qomdt2 * udotb * Byl)) *
                  denom;
        wptilde = (wt + qomdt2 * (ut * Byl - vt * Bxl + qomdt2 * udotb * Bzl)) *
                  denom;
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
        if (param->PERIODICX == true) {  // PERIODIC
          part->x[i] = part->x[i] - grd->Lx;
        } else {  // REFLECTING BC
          part->u[i] = -part->u[i];
          part->x[i] = 2 * grd->Lx - part->x[i];
        }
      }

      if (part->x[i] < 0) {
        if (param->PERIODICX == true) {  // PERIODIC
          part->x[i] = part->x[i] + grd->Lx;
        } else {  // REFLECTING BC
          part->u[i] = -part->u[i];
          part->x[i] = -part->x[i];
        }
      }

      // Y-DIRECTION: BC particles
      if (part->y[i] > grd->Ly) {
        if (param->PERIODICY == true) {  // PERIODIC
          part->y[i] = part->y[i] - grd->Ly;
        } else {  // REFLECTING BC
          part->v[i] = -part->v[i];
          part->y[i] = 2 * grd->Ly - part->y[i];
        }
      }

      if (part->y[i] < 0) {
        if (param->PERIODICY == true) {  // PERIODIC
          part->y[i] = part->y[i] + grd->Ly;
        } else {  // REFLECTING BC
          part->v[i] = -part->v[i];
          part->y[i] = -part->y[i];
        }
      }

      // Z-DIRECTION: BC particles
      if (part->z[i] > grd->Lz) {
        if (param->PERIODICZ == true) {  // PERIODIC
          part->z[i] = part->z[i] - grd->Lz;
        } else {  // REFLECTING BC
          part->w[i] = -part->w[i];
          part->z[i] = 2 * grd->Lz - part->z[i];
        }
      }

      if (part->z[i] < 0) {
        if (param->PERIODICZ == true) {  // PERIODIC
          part->z[i] = part->z[i] + grd->Lz;
        } else {  // REFLECTING BC
          part->w[i] = -part->w[i];
          part->z[i] = -part->z[i];
        }
      }

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

  // Flatten XN, YN, ZN, E and B -> Free at the end
  FPpart* XN_flat = flattenArray(grd->XN, nxn, nyn, nzn);
  FPpart* YN_flat = flattenArray(grd->YN, nxn, nyn, nzn);
  FPpart* ZN_flat = flattenArray(grd->ZN, nxn, nyn, nzn);
  FPfield* Ex_flat = flattenArrayfield(field->Ex, nxn, nyn, nzn);
  FPfield* Ey_flat = flattenArrayfield(field->Ey, nxn, nyn, nzn);
  FPfield* Ez_flat = flattenArrayfield(field->Ez, nxn, nyn, nzn);
  FPfield* Bxn_flat = flattenArrayfield(field->Bxn, nxn, nyn, nzn);
  FPfield* Byn_flat = flattenArrayfield(field->Byn, nxn, nyn, nzn);
  FPfield* Bzn_flat = flattenArrayfield(field->Bzn, nxn, nyn, nzn);

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
  cudaMemcpy(d_XN_flat, XN_flat, nxn * nyn * nzn * sizeof(FPpart),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_YN_flat, YN_flat, nxn * nyn * nzn * sizeof(FPpart),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_ZN_flat, ZN_flat, nxn * nyn * nzn * sizeof(FPpart),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_Ex_flat, Ex_flat, nxn * nyn * nzn * sizeof(FPfield),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_Ey_flat, Ey_flat, nxn * nyn * nzn * sizeof(FPfield),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_Ez_flat, Ez_flat, nxn * nyn * nzn * sizeof(FPfield),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_Bxn_flat, Bxn_flat, nxn * nyn * nzn * sizeof(FPfield),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_Byn_flat, Byn_flat, nxn * nyn * nzn * sizeof(FPfield),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_Bzn_flat, Bzn_flat, nxn * nyn * nzn * sizeof(FPfield),
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

  free(XN_flat);
  free(YN_flat);
  free(ZN_flat);
  free(Ex_flat);
  free(Ey_flat);
  free(Ez_flat);
  free(Bxn_flat);
  free(Byn_flat);
  free(Bzn_flat);

  return 0;
}

__device__ void call_weight(int ix, int iy, int iz, FPpart x, FPpart y,
                            FPpart z, FPpart u, FPpart v, FPpart w, FPpart q,
                            FPpart* d_XN_flat, FPpart* d_YN_flat,
                            FPpart* d_ZN_flat, FPpart d_invVOL,
                            FPpart weight[2][2][2], int d_nyn, int d_nzn) {
  FPpart xi[2], eta[2], zeta[2];

  // Compute offsets
  xi[0] = x - d_XN_flat[(ix - 1) * (d_nyn * d_nzn) + (iy)*d_nzn + (iz)];
  eta[0] = y - d_YN_flat[ix * d_nyn * d_nzn + (iy - 1) * d_nzn + (iz)];
  zeta[0] = z - d_ZN_flat[ix * d_nyn * d_nzn + (iy)*d_nzn + (iz - 1)];

  int index = ix * d_nyn * d_nzn + (iy)*d_nzn + (iz);
  xi[1] = d_XN_flat[index] - x;
  eta[1] = d_YN_flat[index] - y;
  zeta[1] = d_ZN_flat[index] - z;

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

__global__ void calculate_weight(FPpart* d_x, FPpart* d_y, FPpart* d_z,
                                 FPpart* d_q, FPpart* d_XN_flat,
                                 FPpart* d_YN_flat, FPpart* d_ZN_flat,
                                 FPpart* d_weight, FPpart d_invVOL,
                                 FPpart d_xStart, FPpart d_yStart,
                                 FPpart d_zStart, FPpart d_invdx,
                                 FPpart d_invdy, FPpart d_invdz, int d_nxn,
                                 int d_nyn, int d_nzn, int d_nop) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < d_nop) {
    // arrays needed for interpolation
    // FPpart temp[2][2][2];

    FPpart x = d_x[i];
    FPpart y = d_y[i];
    FPpart z = d_z[i];
    FPpart q = d_q[i];

    // index of the cell
    int ix, iy, iz;

    // determine cell: can we change to int()? is it faster?
    ix = 2 + int(floor((x - d_xStart) * d_invdx));
    iy = 2 + int(floor((y - d_yStart) * d_invdy));
    iz = 2 + int(floor((z - d_zStart) * d_invdz));

    // FPpart weight[2][2][2];

    FPpart xi[2], eta[2], zeta[2];

    // Compute offsets
    xi[0] = x - d_XN_flat[(ix - 1) * (d_nyn * d_nzn) + (iy)*d_nzn + (iz)];
    eta[0] = y - d_YN_flat[ix * d_nyn * d_nzn + (iy - 1) * d_nzn + (iz)];
    zeta[0] = z - d_ZN_flat[ix * d_nyn * d_nzn + (iy)*d_nzn + (iz - 1)];

    int index = ix * d_nyn * d_nzn + (iy)*d_nzn + (iz);
    xi[1] = d_XN_flat[index] - x;
    eta[1] = d_YN_flat[index] - y;
    zeta[1] = d_ZN_flat[index] - z;

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

__global__ void interpP2G_kernel_rhon(FPpart* d_x, FPpart* d_y, FPpart* d_z,
                                      FPpart* d_rhon_flat, FPpart* d_weight,
                                      FPpart d_invVOL, FPpart d_xStart,
                                      FPpart d_yStart, FPpart d_zStart,
                                      FPpart d_invdx, FPpart d_invdy,
                                      FPpart d_invdz, int d_nxn, int d_nyn,
                                      int d_nzn, int d_nop) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < d_nop) {
    // arrays needed for interpolation
    // FPpart temp[2][2][2];

    FPpart x = d_x[i];
    FPpart y = d_y[i];
    FPpart z = d_z[i];

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

__global__ void interpP2G_kernel_Jx(FPpart* d_x, FPpart* d_y, FPpart* d_z,
                                    FPpart* d_u, FPpart* d_Jx_flat,
                                    FPpart* d_weight, FPpart d_invVOL,
                                    FPpart d_xStart, FPpart d_yStart,
                                    FPpart d_zStart, FPpart d_invdx,
                                    FPpart d_invdy, FPpart d_invdz, int d_nxn,
                                    int d_nyn, int d_nzn, int d_nop) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < d_nop) {
    // arrays needed for interpolation
    // FPpart temp[2][2][2];

    FPpart x = d_x[i];
    FPpart y = d_y[i];
    FPpart z = d_z[i];
    FPpart u = d_u[i];

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

__global__ void interpP2G_kernel_Jy(FPpart* d_x, FPpart* d_y, FPpart* d_z,
                                    FPpart* d_v, FPpart* d_Jy_flat,
                                    FPpart* d_weight, FPpart d_invVOL,
                                    FPpart d_xStart, FPpart d_yStart,
                                    FPpart d_zStart, FPpart d_invdx,
                                    FPpart d_invdy, FPpart d_invdz, int d_nxn,
                                    int d_nyn, int d_nzn, int d_nop) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < d_nop) {
    // arrays needed for interpolation
    // FPpart temp[2][2][2];

    FPpart x = d_x[i];
    FPpart y = d_y[i];
    FPpart z = d_z[i];
    FPpart v = d_v[i];

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

__global__ void interpP2G_kernel_Jz(FPpart* d_x, FPpart* d_y, FPpart* d_z,
                                    FPpart* d_w, FPpart* d_Jz_flat,
                                    FPpart* d_weight, FPpart d_invVOL,
                                    FPpart d_xStart, FPpart d_yStart,
                                    FPpart d_zStart, FPpart d_invdx,
                                    FPpart d_invdy, FPpart d_invdz, int d_nxn,
                                    int d_nyn, int d_nzn, int d_nop) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < d_nop) {
    // arrays needed for interpolation
    // FPpart temp[2][2][2];

    FPpart x = d_x[i];
    FPpart y = d_y[i];
    FPpart z = d_z[i];
    FPpart w = d_w[i];

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

__global__ void interpP2G_kernel_pxx(FPpart* d_x, FPpart* d_y, FPpart* d_z,
                                     FPpart* d_u, FPpart* d_pxx_flat,
                                     FPpart* d_weight, FPpart d_invVOL,
                                     FPpart d_xStart, FPpart d_yStart,
                                     FPpart d_zStart, FPpart d_invdx,
                                     FPpart d_invdy, FPpart d_invdz, int d_nxn,
                                     int d_nyn, int d_nzn, int d_nop) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < d_nop) {
    // arrays needed for interpolation
    // FPpart temp[2][2][2];

    FPpart x = d_x[i];
    FPpart y = d_y[i];
    FPpart z = d_z[i];
    FPpart u = d_u[i];

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

__global__ void interpP2G_kernel_pxy(
    FPpart* d_x, FPpart* d_y, FPpart* d_z, FPpart* d_u, FPpart* d_v,
    FPpart* d_pxy_flat, FPpart* d_weight, FPpart d_invVOL, FPpart d_xStart,
    FPpart d_yStart, FPpart d_zStart, FPpart d_invdx, FPpart d_invdy,
    FPpart d_invdz, int d_nxn, int d_nyn, int d_nzn, int d_nop) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < d_nop) {
    // arrays needed for interpolation
    // FPpart temp[2][2][2];

    FPpart x = d_x[i];
    FPpart y = d_y[i];
    FPpart z = d_z[i];
    FPpart u = d_u[i];
    FPpart v = d_v[i];

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

__global__ void interpP2G_kernel_pxz(
    FPpart* d_x, FPpart* d_y, FPpart* d_z, FPpart* d_u, FPpart* d_w,
    FPpart* d_pxz_flat, FPpart* d_weight, FPpart d_invVOL, FPpart d_xStart,
    FPpart d_yStart, FPpart d_zStart, FPpart d_invdx, FPpart d_invdy,
    FPpart d_invdz, int d_nxn, int d_nyn, int d_nzn, int d_nop) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < d_nop) {
    // arrays needed for interpolation
    // FPpart temp[2][2][2];

    FPpart x = d_x[i];
    FPpart y = d_y[i];
    FPpart z = d_z[i];
    FPpart u = d_u[i];
    FPpart w = d_w[i];

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

__global__ void interpP2G_kernel_pyy(FPpart* d_x, FPpart* d_y, FPpart* d_z,
                                     FPpart* d_v, FPpart* d_pyy_flat,
                                     FPpart* d_weight, FPpart d_invVOL,
                                     FPpart d_xStart, FPpart d_yStart,
                                     FPpart d_zStart, FPpart d_invdx,
                                     FPpart d_invdy, FPpart d_invdz, int d_nxn,
                                     int d_nyn, int d_nzn, int d_nop) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < d_nop) {
    // arrays needed for interpolation
    // FPpart temp[2][2][2];

    FPpart x = d_x[i];
    FPpart y = d_y[i];
    FPpart z = d_z[i];
    FPpart v = d_v[i];

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

__global__ void interpP2G_kernel_pyz(
    FPpart* d_x, FPpart* d_y, FPpart* d_z, FPpart* d_v, FPpart* d_w,
    FPpart* d_pyz_flat, FPpart* d_weight, FPpart d_invVOL, FPpart d_xStart,
    FPpart d_yStart, FPpart d_zStart, FPpart d_invdx, FPpart d_invdy,
    FPpart d_invdz, int d_nxn, int d_nyn, int d_nzn, int d_nop) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < d_nop) {
    // arrays needed for interpolation
    // FPpart temp[2][2][2];

    FPpart x = d_x[i];
    FPpart y = d_y[i];
    FPpart z = d_z[i];
    FPpart v = d_v[i];
    FPpart w = d_w[i];

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

__global__ void interpP2G_kernel_pzz(FPpart* d_x, FPpart* d_y, FPpart* d_z,
                                     FPpart* d_w, FPpart* d_pzz_flat,
                                     FPpart* d_weight, FPpart d_invVOL,
                                     FPpart d_xStart, FPpart d_yStart,
                                     FPpart d_zStart, FPpart d_invdx,
                                     FPpart d_invdy, FPpart d_invdz, int d_nxn,
                                     int d_nyn, int d_nzn, int d_nop) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < d_nop) {
    // arrays needed for interpolation
    // FPpart temp[2][2][2];

    FPpart x = d_x[i];
    FPpart y = d_y[i];
    FPpart z = d_z[i];
    FPpart w = d_w[i];

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
    FPpart* d_x, FPpart* d_y, FPpart* d_z, FPpart* d_u, FPpart* d_v,
    FPpart* d_w, FPpart* d_q, FPpart* d_XN_flat, FPpart* d_YN_flat,
    FPpart* d_ZN_flat, FPpart* d_rhon_flat, FPpart* d_Jx_flat,
    FPpart* d_Jy_flat, FPpart* d_Jz_flat, FPpart* d_pxx_flat,
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

    FPpart x = d_x[i];
    FPpart y = d_y[i];
    FPpart z = d_z[i];
    FPpart u = d_u[i];
    FPpart v = d_v[i];
    FPpart w = d_w[i];
    FPpart q = d_q[i];

    // index of the cell
    int ix, iy, iz;

    // determine cell: can we change to int()? is it faster?
    ix = 2 + int(floor((x - d_xStart) * d_invdx));
    iy = 2 + int(floor((y - d_yStart) * d_invdy));
    iz = 2 + int(floor((z - d_zStart) * d_invdz));

    // Check indixing
    xi[0] = x - d_XN_flat[(ix - 1) * (d_nyn * d_nzn) + (iy)*d_nzn + (iz)];
    eta[0] = y - d_YN_flat[ix * d_nyn * d_nzn + (iy - 1) * d_nzn + (iz)];
    zeta[0] = z - d_ZN_flat[ix * d_nyn * d_nzn + (iy)*d_nzn + (iz - 1)];

    int index = ix * d_nyn * d_nzn + (iy)*d_nzn + (iz);
    xi[1] = d_XN_flat[index] - x;
    eta[1] = d_YN_flat[index] - y;
    zeta[1] = d_ZN_flat[index] - z;

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
//     FPpart* d_x, FPpart* d_y, FPpart* d_z,
//     FPpart* d_u, FPpart* d_v, FPpart* d_w,
//     FPpart* d_q,
//     FPpart* d_XN_flat, FPpart* d_YN_flat, FPpart* d_ZN_flat,
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

//         FPpart x = d_x[i];
//         FPpart y = d_y[i];
//         FPpart z = d_z[i];
//         FPpart u = d_u[i];
//         FPpart v = d_v[i];
//         FPpart w = d_w[i];
//         FPpart q = d_q[i];

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
//         xi[1] = d_XN_flat[index] - x;
//         eta[1] = d_YN_flat[index] - y;
//         zeta[1] = d_ZN_flat[index] - z;

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

// This is Async V2
void interpP2G_GPU_V2(struct particles* part, struct interpDensSpecies* ids,
                      struct grid* grd) {
  // corresponds to XN, YN etc. also Ex, Bxn COPY for slimpicity
  int nxn = grd->nxn;
  int nyn = grd->nyn;
  int nzn = grd->nzn;
  int nop = part->nop;

  // Flatten XN, YN, ZN, rhon, J and p -> Free at the end
  FPpart* XN_flat = flattenArray(grd->XN, nxn, nyn, nzn);
  FPpart* YN_flat = flattenArray(grd->YN, nxn, nyn, nzn);
  FPpart* ZN_flat = flattenArray(grd->ZN, nxn, nyn, nzn);
  FPpart* rhon_flat = flattenArray(ids->rhon, nxn, nyn, nzn);
  FPpart* Jx_flat = flattenArray(ids->Jx, nxn, nyn, nzn);
  FPpart* Jy_flat = flattenArray(ids->Jy, nxn, nyn, nzn);
  FPpart* Jz_flat = flattenArray(ids->Jz, nxn, nyn, nzn);
  FPpart* pxx_flat = flattenArray(ids->pxx, nxn, nyn, nzn);
  FPpart* pxy_flat = flattenArray(ids->pxy, nxn, nyn, nzn);
  FPpart* pxz_flat = flattenArray(ids->pxz, nxn, nyn, nzn);
  FPpart* pyy_flat = flattenArray(ids->pyy, nxn, nyn, nzn);
  FPpart* pyz_flat = flattenArray(ids->pyz, nxn, nyn, nzn);
  FPpart* pzz_flat = flattenArray(ids->pzz, nxn, nyn, nzn);

  int nStreams = 10;
  cudaStream_t stream[nStreams];
  for (int i = 0; i < nStreams; ++i) cudaStreamCreate(&stream[i]);

  // Device pointers -> Free at the end
  FPpart *d_x, *d_y, *d_z;
  FPpart *d_u, *d_v, *d_w;
  FPpart* d_q;
  FPpart* d_weight;
  FPpart *d_XN_flat, *d_YN_flat, *d_ZN_flat;
  FPpart *d_rhon_flat, *d_Jx_flat, *d_Jy_flat, *d_Jz_flat;
  FPpart *d_pxx_flat, *d_pxy_flat, *d_pxz_flat, *d_pyy_flat, *d_pyz_flat,
      *d_pzz_flat;

  // Allocate memory on the device
  cudaMalloc(&d_x, nop * sizeof(FPpart));
  cudaMalloc(&d_y, nop * sizeof(FPpart));
  cudaMalloc(&d_z, nop * sizeof(FPpart));
  cudaMalloc(&d_u, nop * sizeof(FPpart));
  cudaMalloc(&d_v, nop * sizeof(FPpart));
  cudaMalloc(&d_w, nop * sizeof(FPpart));
  cudaMalloc(&d_q, nop * sizeof(FPpart));
  cudaMalloc(&d_weight, 8 * nop * sizeof(FPpart));

  cudaMalloc(&d_XN_flat, nxn * nyn * nzn * sizeof(FPpart));
  cudaMalloc(&d_YN_flat, nxn * nyn * nzn * sizeof(FPpart));
  cudaMalloc(&d_ZN_flat, nxn * nyn * nzn * sizeof(FPpart));
  cudaMalloc(&d_rhon_flat, nxn * nyn * nzn * sizeof(FPpart));
  cudaMalloc(&d_Jx_flat, nxn * nyn * nzn * sizeof(FPpart));
  cudaMalloc(&d_Jy_flat, nxn * nyn * nzn * sizeof(FPpart));
  cudaMalloc(&d_Jz_flat, nxn * nyn * nzn * sizeof(FPpart));
  cudaMalloc(&d_pxx_flat, nxn * nyn * nzn * sizeof(FPpart));
  cudaMalloc(&d_pxy_flat, nxn * nyn * nzn * sizeof(FPpart));
  cudaMalloc(&d_pxz_flat, nxn * nyn * nzn * sizeof(FPpart));
  cudaMalloc(&d_pyy_flat, nxn * nyn * nzn * sizeof(FPpart));
  cudaMalloc(&d_pyz_flat, nxn * nyn * nzn * sizeof(FPpart));
  cudaMalloc(&d_pzz_flat, nxn * nyn * nzn * sizeof(FPpart));

  int currentStreamSize = nxn * nyn * nzn * sizeof(FPpart);

  // Copy data from host to device

  cudaMemcpyAsync(d_x, part->x, nop * sizeof(FPpart), cudaMemcpyHostToDevice,
                  stream[0]);
  cudaMemcpyAsync(d_y, part->y, nop * sizeof(FPpart), cudaMemcpyHostToDevice,
                  stream[1]);
  cudaMemcpyAsync(d_z, part->z, nop * sizeof(FPpart), cudaMemcpyHostToDevice,
                  stream[2]);

  cudaMemcpyAsync(d_q, part->q, nop * sizeof(FPpart), cudaMemcpyHostToDevice,
                  stream[6]);
  cudaMemcpyAsync(d_XN_flat, XN_flat, nxn * nyn * nzn * sizeof(FPpart),
                  cudaMemcpyHostToDevice, stream[7]);
  cudaMemcpyAsync(d_YN_flat, YN_flat, nxn * nyn * nzn * sizeof(FPpart),
                  cudaMemcpyHostToDevice, stream[8]);
  cudaMemcpyAsync(d_ZN_flat, ZN_flat, nxn * nyn * nzn * sizeof(FPpart),
                  cudaMemcpyHostToDevice, stream[9]);

  for (int i = 0; i < nStreams; ++i) cudaStreamSynchronize(stream[i]);

  int threadsPerBlock = 256;
  int blocksPerGrid = (part->nop + threadsPerBlock - 1) / threadsPerBlock;

  calculate_weight<<<blocksPerGrid, threadsPerBlock, 0, stream[5]>>>(
      d_x, d_y, d_z, d_q, d_XN_flat, d_YN_flat, d_ZN_flat, d_weight,
      grd->invVOL, grd->xStart, grd->yStart, grd->zStart, grd->invdx,
      grd->invdy, grd->invdz, nxn, nyn, nzn, nop);

  cudaMemcpyAsync(d_u, part->u, nop * sizeof(FPpart), cudaMemcpyHostToDevice,
                  stream[3]);
  cudaMemcpyAsync(d_v, part->v, nop * sizeof(FPpart), cudaMemcpyHostToDevice,
                  stream[4]);
  cudaMemcpyAsync(d_w, part->w, nop * sizeof(FPpart), cudaMemcpyHostToDevice,
                  stream[5]);

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

  interpP2G_kernel_rhon<<<blocksPerGrid, threadsPerBlock, 0, stream[0]>>>(
      d_x, d_y, d_z, d_rhon_flat, d_weight, grd->invVOL, grd->xStart,
      grd->yStart, grd->zStart, grd->invdx, grd->invdy, grd->invdz, nxn, nyn,
      nzn, nop);
  interpP2G_kernel_Jx<<<blocksPerGrid, threadsPerBlock, 0, stream[1]>>>(
      d_x, d_y, d_z, d_u, d_Jx_flat, d_weight, grd->invVOL, grd->xStart,
      grd->yStart, grd->zStart, grd->invdx, grd->invdy, grd->invdz, nxn, nyn,
      nzn, nop);
  interpP2G_kernel_Jy<<<blocksPerGrid, threadsPerBlock, 0, stream[2]>>>(
      d_x, d_y, d_z, d_v, d_Jy_flat, d_weight, grd->invVOL, grd->xStart,
      grd->yStart, grd->zStart, grd->invdx, grd->invdy, grd->invdz, nxn, nyn,
      nzn, nop);
  interpP2G_kernel_Jz<<<blocksPerGrid, threadsPerBlock, 0, stream[3]>>>(
      d_x, d_y, d_z, d_w, d_Jz_flat, d_weight, grd->invVOL, grd->xStart,
      grd->yStart, grd->zStart, grd->invdx, grd->invdy, grd->invdz, nxn, nyn,
      nzn, nop);
  interpP2G_kernel_pxx<<<blocksPerGrid, threadsPerBlock, 0, stream[4]>>>(
      d_x, d_y, d_z, d_u, d_pxx_flat, d_weight, grd->invVOL, grd->xStart,
      grd->yStart, grd->zStart, grd->invdx, grd->invdy, grd->invdz, nxn, nyn,
      nzn, nop);
  interpP2G_kernel_pxy<<<blocksPerGrid, threadsPerBlock, 0, stream[5]>>>(
      d_x, d_y, d_z, d_u, d_v, d_pxy_flat, d_weight, grd->invVOL, grd->xStart,
      grd->yStart, grd->zStart, grd->invdx, grd->invdy, grd->invdz, nxn, nyn,
      nzn, nop);
  interpP2G_kernel_pxz<<<blocksPerGrid, threadsPerBlock, 0, stream[6]>>>(
      d_x, d_y, d_z, d_u, d_w, d_pxz_flat, d_weight, grd->invVOL, grd->xStart,
      grd->yStart, grd->zStart, grd->invdx, grd->invdy, grd->invdz, nxn, nyn,
      nzn, nop);
  interpP2G_kernel_pyy<<<blocksPerGrid, threadsPerBlock, 0, stream[7]>>>(
      d_x, d_y, d_z, d_v, d_pyy_flat, d_weight, grd->invVOL, grd->xStart,
      grd->yStart, grd->zStart, grd->invdx, grd->invdy, grd->invdz, nxn, nyn,
      nzn, nop);
  interpP2G_kernel_pyz<<<blocksPerGrid, threadsPerBlock, 0, stream[8]>>>(
      d_x, d_y, d_z, d_v, d_w, d_pyz_flat, d_weight, grd->invVOL, grd->xStart,
      grd->yStart, grd->zStart, grd->invdx, grd->invdy, grd->invdz, nxn, nyn,
      nzn, nop);
  interpP2G_kernel_pzz<<<blocksPerGrid, threadsPerBlock, 0, stream[9]>>>(
      d_x, d_y, d_z, d_w, d_pzz_flat, d_weight, grd->invVOL, grd->xStart,
      grd->yStart, grd->zStart, grd->invdx, grd->invdy, grd->invdz, nxn, nyn,
      nzn, nop);

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

  // Write to original array and unflatten
  unflattenArray(rhon_flat, ids->rhon, nxn, nyn, nzn);
  unflattenArray(Jx_flat, ids->Jx, nxn, nyn, nzn);
  unflattenArray(Jy_flat, ids->Jy, nxn, nyn, nzn);
  unflattenArray(Jz_flat, ids->Jz, nxn, nyn, nzn);
  unflattenArray(pxx_flat, ids->pxx, nxn, nyn, nzn);
  unflattenArray(pxy_flat, ids->pxy, nxn, nyn, nzn);
  unflattenArray(pxz_flat, ids->pxz, nxn, nyn, nzn);
  unflattenArray(pyy_flat, ids->pyy, nxn, nyn, nzn);
  unflattenArray(pyz_flat, ids->pyz, nxn, nyn, nzn);
  unflattenArray(pzz_flat, ids->pzz, nxn, nyn, nzn);

  // Free cuda arrays
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_w);
  cudaFree(d_q);
  cudaFree(d_XN_flat);
  cudaFree(d_YN_flat);
  cudaFree(d_ZN_flat);
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

  // Free the arrays
  free(XN_flat);
  free(YN_flat);
  free(ZN_flat);
  free(rhon_flat);
  free(Jx_flat);
  free(Jy_flat);
  free(Jz_flat);
  free(pxx_flat);
  free(pxy_flat);
  free(pxz_flat);
  free(pyy_flat);
  free(pyz_flat);
  free(pzz_flat);

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
//         cudaMemcpyAsync(&d_x[offset], &part->x[offset], currentChunkSize *
//         sizeof(FPpart), cudaMemcpyHostToDevice, streams[i]);
//         cudaMemcpyAsync(&d_y[offset], &part->y[offset], currentChunkSize *
//         sizeof(FPpart), cudaMemcpyHostToDevice, streams[i]);
//         cudaMemcpyAsync(&d_z[offset], &part->z[offset], currentChunkSize *
//         sizeof(FPpart), cudaMemcpyHostToDevice, streams[i]);
//         cudaMemcpyAsync(&d_u[offset], &part->u[offset], currentChunkSize *
//         sizeof(FPpart), cudaMemcpyHostToDevice, streams[i]);
//         cudaMemcpyAsync(&d_v[offset], &part->v[offset], currentChunkSize *
//         sizeof(FPpart), cudaMemcpyHostToDevice, streams[i]);
//         cudaMemcpyAsync(&d_w[offset], &part->w[offset], currentChunkSize *
//         sizeof(FPpart), cudaMemcpyHostToDevice, streams[i]);
//         cudaMemcpyAsync(&d_q[offset], &part->q[offset], currentChunkSize *
//         sizeof(FPpart), cudaMemcpyHostToDevice, streams[i]);

//         // Launch kernel
//         int blocksPerGrid = (currentChunkSize + threadsPerBlock - 1) /
//         threadsPerBlock; interpP2G_kernel<<<blocksPerGrid, threadsPerBlock,
//         0, streams[i]>>>(
//             &d_x[offset], &d_y[offset], &d_z[offset],
//             &d_u[offset], &d_v[offset], &d_w[offset],
//             &d_q[offset],
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

// GPU Mover Time / Cycle   (s) = 0.250327
//    CPU Mover Time / Cycle   (s) = 2.89693