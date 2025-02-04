#ifndef PARTICLES_H
#define PARTICLES_H

#include <math.h>

#include "Alloc.h"
#include "EMfield.h"
#include "Grid.h"
#include "InterpDensSpecies.h"
#include "Parameters.h"
#include "PrecisionTypes.h"
#include <nvtx3/nvtx3.hpp>

struct Particle {
  // position
  FPpart x, y, z;
  // velocity
  FPpart u, v, w;
  // charge
  /** q must have precision of interpolated quantities: typically double. Not
  used in mover */
  FPinterp q;
};

struct particles {
  /** species ID: 0, 1, 2 , ... */
  int species_ID;

  /** maximum number of particles of this species on this domain. used for
   * memory allocation */
  long npmax;
  /** number of particles of this species on this domain */
  long nop;

  /** Electron and ions have different number of iterations: ions moves slower
   * than ions */
  int NiterMover;
  /** number of particle of subcycles in the mover */
  int n_sub_cycles;

  /** number of particles per cell */
  int npcel;
  /** number of particles per cell - X direction */
  int npcelx;
  /** number of particles per cell - Y direction */
  int npcely;
  /** number of particles per cell - Z direction */
  int npcelz;

  /** charge over mass ratio */
  FPpart qom;

  /* drift and thermal velocities for this species */
  FPpart u0, v0, w0;
  FPpart uth, vth, wth;

  // /** particle arrays: 1D arrays[npmax] */
  Particle *data;
};

/** allocate particle arrays */
void particle_allocate(struct parameters *, struct particles *, int);

/** deallocate */
void particle_deallocate(struct particles *);

/** particle mover */
int mover_PC(struct particles *, struct EMfield *, struct grid *,
             struct parameters *);

int mover_PC_GPU(struct particles *, struct EMfield *, struct grid *,
                 struct parameters *);

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles *, struct interpDensSpecies *, struct grid *);

void interpP2G_GPU(struct particles *, struct interpDensSpecies *,
                   struct grid *);
void interpP2G_GPU_sync(struct particles *, struct interpDensSpecies *,
                   struct grid *);

#endif
