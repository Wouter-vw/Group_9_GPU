#pragma once

#include "Alloc.h"
#include "Grid.h"

/** structure with field information */
struct EMfield {
  // field arrays: 4D arrays

  /* Electric field defined on nodes: last index is component */
  FPfield*** Ex;
  FPfield* Ex_flat;
  FPfield*** Ey;
  FPfield* Ey_flat;
  FPfield*** Ez;
  FPfield* Ez_flat;
  /* Magnetic field defined on nodes: last index is component */
  FPfield*** Bxn;
  FPfield* Bxn_flat;
  FPfield*** Byn;
  FPfield* Byn_flat;
  FPfield*** Bzn;
  FPfield* Bzn_flat;

  /** allocate electric and magnetic field */
  void allocate(Grid*);

  /** deallocate electric and magnetic field */
  void deallocate(Grid*);
};

/** allocate electric and magnetic field */
void field_allocate(struct Grid*, struct EMfield*);

/** deallocate electric and magnetic field */
void field_deallocate(struct Grid*, struct EMfield*);
