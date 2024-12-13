#include "Particles.h"

#include "Alloc.h"
// #include <cuda.h>
// #include <cuda_runtime.h>

/** allocate particle arrays */
void Particles::allocate(struct Parameters* param, int is) {
  // set species ID
  species_ID = is;
  // number of particles
  nop = param->np[is];
  // maximum number of particles
  npmax = param->npMax[is];

  // choose a different number of mover iterations for ions and electrons
  if (param->qom[is] < 0) {  // electrons
    NiterMover = param->NiterMover;
    n_sub_cycles = param->n_sub_cycles;
  } else {  // ions: only one iteration
    NiterMover = 1;
    n_sub_cycles = 1;
  }

  // particles per cell
  npcelx = param->npcelx[is];
  npcely = param->npcely[is];
  npcelz = param->npcelz[is];
  npcel = npcelx * npcely * npcelz;

  // cast it to required precision
  qom = (FPpart)param->qom[is];

  long npmax = npmax;

  // initialize drift and thermal velocities
  // drift
  u0 = (FPpart)param->u0[is];
  v0 = (FPpart)param->v0[is];
  w0 = (FPpart)param->w0[is];
  // thermal
  uth = (FPpart)param->uth[is];
  vth = (FPpart)param->vth[is];
  wth = (FPpart)param->wth[is];

  //////////////////////////////
  /// ALLOCATION PARTICLE ARRAYS
  //////////////////////////////

  particles = new Particle[npmax];
}

void Particles::deallocate() { delete[] particles; }
void Particle::update(ParticleSettings* part, Grid* grd, Parameters* param,
                      EMfield* field) {
  FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
  int ix, iy, iz;
  xptilde = x;
  yptilde = y;
  zptilde = z;

  // auxiliary variables
  FPpart dt_sub_cycling = (FPpart)param->dt / ((double)part->nSubCycles);
  FPpart dto2 = .5 * dt_sub_cycling, qomdt2 = part->qom * dto2 / param->c;
  FPpart omdtsq, denom, ut, vt, wt, udotb;

  // local (to the particle) electric and magnetic field
  FPfield Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

  // interpolation densities
  FPfield weight[2][2][2];
  FPfield xi[2], eta[2], zeta[2];

  // intermediate particle position and velocity

  // calculate the average velocity iteratively
  for (int innter = 0; innter < part->nIterMover; innter++) {
    // interpolation G-->P
    ix = 2 + int((x - grd->xStart) * grd->invdx);
    iy = 2 + int((y - grd->yStart) * grd->invdy);
    iz = 2 + int((z - grd->zStart) * grd->invdz);

    // calculate weights
    xi[0] = x - grd->XN[ix - 1][iy][iz];
    eta[0] = y - grd->YN[ix][iy - 1][iz];
    zeta[0] = z - grd->ZN[ix][iy][iz - 1];
    xi[1] = grd->XN[ix][iy][iz] - x;
    eta[1] = grd->YN[ix][iy][iz] - y;
    zeta[1] = grd->ZN[ix][iy][iz] - z;
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
    ut = u + qomdt2 * Exl;
    vt = v + qomdt2 * Eyl;
    wt = w + qomdt2 * Ezl;
    udotb = ut * Bxl + vt * Byl + wt * Bzl;
    // solve the velocity equation
    uptilde =
        (ut + qomdt2 * (vt * Bzl - wt * Byl + qomdt2 * udotb * Bxl)) * denom;
    vptilde =
        (vt + qomdt2 * (wt * Bxl - ut * Bzl + qomdt2 * udotb * Byl)) * denom;
    wptilde =
        (wt + qomdt2 * (ut * Byl - vt * Bxl + qomdt2 * udotb * Bzl)) * denom;
    // update position
    x = xptilde + uptilde * dto2;
    y = yptilde + vptilde * dto2;
    z = zptilde + wptilde * dto2;

  }  // end of iteration
  // update the final position and velocity
  u = 2.0 * uptilde - u;
  v = 2.0 * vptilde - v;
  w = 2.0 * wptilde - w;
  x = xptilde + uptilde * dt_sub_cycling;
  y = yptilde + vptilde * dt_sub_cycling;
  z = zptilde + wptilde * dt_sub_cycling;

  //////////
  //////////
  ////////// BC

  // X-DIRECTION: BC particles
  if (x > grd->Lx) {
    if (param->PERIODICX == true) {  // PERIODIC
      x = x - grd->Lx;
    } else {  // REFLECTING BC
      u = -u;
      x = 2 * grd->Lx - x;
    }
  }

  if (x < 0) {
    if (param->PERIODICX == true) {  // PERIODIC
      x = x + grd->Lx;
    } else {  // REFLECTING BC
      u = -u;
      x = -x;
    }
  }

  // Y-DIRECTION: BC particles
  if (y > grd->Ly) {
    if (param->PERIODICY == true) {  // PERIODIC
      y = y - grd->Ly;
    } else {  // REFLECTING BC
      v = -v;
      y = 2 * grd->Ly - y;
    }
  }

  if (y < 0) {
    if (param->PERIODICY == true) {  // PERIODIC
      y = y + grd->Ly;
    } else {  // REFLECTING BC
      v = -v;
      y = -y;
    }
  }

  // Z-DIRECTION: BC particles
  if (z > grd->Lz) {
    if (param->PERIODICZ == true) {  // PERIODIC
      z = z - grd->Lz;
    } else {  // REFLECTING BC
      w = -w;
      z = 2 * grd->Lz - z;
    }
  }

  if (z < 0) {
    if (param->PERIODICZ == true) {  // PERIODIC
      z = z + grd->Lz;
    } else {  // REFLECTING BC
      w = -w;
      z = -z;
    }
  }
}

/** particle mover */
int mover_PC(struct Particles* part, struct EMfield* field, struct Grid* grd,
             struct Parameters* param) {
  // print species and subcycling
  std::cout << "***  MOVER with SUBCYCLYING " << param->n_sub_cycles
            << " - species " << part->species_ID << " ***" << std::endl;

  std::cout << "N sub cycles: " << part->n_sub_cycles << std::endl;
  std::cout << "N OP: " << part->nop << std::endl;
  std::cout << "N iter mover: " << part->NiterMover << std::endl;
  // start subcycling

  ParticleSettings* settings = new ParticleSettings(
      part->qom, part->NiterMover, part->nop, part->n_sub_cycles);
  for (int i_sub = 0; i_sub < part->n_sub_cycles; i_sub++) {
    // move each particle with new fields
    for (int i = 0; i < part->nop; i++) {
      part->particles[i].update(settings, grd, param, field);
    }  // end of subcycling
  }  // end of one particle

  std::cout << "delete settings\n";
  delete settings;
  return (0);  // exit succcesfully
}  // end of the mover

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct Particles* part, struct interpDensSpecies* ids,
               struct Grid* grd) {
  // arrays needed for interpolation
  FPpart weight[2][2][2];
  FPpart temp[2][2][2];
  FPpart xi[2], eta[2], zeta[2];

  // index of the cell
  int ix, iy, iz;

  for (long long i = 0; i < part->nop; i++) {
    // determine cell: can we change to int()? is it faster?
    ix = 2 + int(floor((part->particles[i].x - grd->xStart) * grd->invdx));
    iy = 2 + int(floor((part->particles[i].y - grd->yStart) * grd->invdy));
    iz = 2 + int(floor((part->particles[i].z - grd->zStart) * grd->invdz));

    // distances from node
    xi[0] = part->particles[i].x - grd->XN[ix - 1][iy][iz];
    eta[0] = part->particles[i].y - grd->YN[ix][iy - 1][iz];
    zeta[0] = part->particles[i].z - grd->ZN[ix][iy][iz - 1];
    xi[1] = grd->XN[ix][iy][iz] - part->particles[i].x;
    eta[1] = grd->YN[ix][iy][iz] - part->particles[i].y;
    zeta[1] = grd->ZN[ix][iy][iz] - part->particles[i].z;

    // calculate the weights for different nodes
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          weight[ii][jj][kk] =
              part->particles[i].q * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;

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
          temp[ii][jj][kk] = part->particles[i].u * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    ////////////////////////////
    // add current density - Jy
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] = part->particles[i].v * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    ////////////////////////////
    // add current density - Jz
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] = part->particles[i].w * weight[ii][jj][kk];
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
              part->particles[i].u * part->particles[i].u * weight[ii][jj][kk];
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
              part->particles[i].u * part->particles[i].v * weight[ii][jj][kk];
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
              part->particles[i].u * part->particles[i].w * weight[ii][jj][kk];
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
              part->particles[i].v * part->particles[i].v * weight[ii][jj][kk];
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
              part->particles[i].v * part->particles[i].w * weight[ii][jj][kk];
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
              part->particles[i].w * part->particles[i].w * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->pzz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
  }
}
