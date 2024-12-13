#include "EMfield.h"

void EMfield::allocate(Grid* grid) {
  // E on nodes
  Ex = newArr3<FPfield>(&Ex_flat, grid->nxn, grid->nyn, grid->nzn);
  Ey = newArr3<FPfield>(&Ey_flat, grid->nxn, grid->nyn, grid->nzn);
  Ez = newArr3<FPfield>(&Ez_flat, grid->nxn, grid->nyn, grid->nzn);
  // B on nodes
  Bxn = newArr3<FPfield>(&Bxn_flat, grid->nxn, grid->nyn, grid->nzn);
  Byn = newArr3<FPfield>(&Byn_flat, grid->nxn, grid->nyn, grid->nzn);
  Bzn = newArr3<FPfield>(&Bzn_flat, grid->nxn, grid->nyn, grid->nzn);
}

/** deallocate electric and magnetic field */
void EMfield::deallocate(Grid* grid) {
  // E deallocate 3D arrays
  delArr3(Ex, grid->nxn, grid->nyn);
  delArr3(Ey, grid->nxn, grid->nyn);
  delArr3(Ez, grid->nxn, grid->nyn);

  // B deallocate 3D arrays
  delArr3(Bxn, grid->nxn, grid->nyn);
  delArr3(Byn, grid->nxn, grid->nyn);
  delArr3(Bzn, grid->nxn, grid->nyn);
}