#include "EMfield_aux.h"

/** allocate electric and magnetic field */
void EMfield_aux::allocate(Grid* grd) {
  // Electrostatic potential
  Phi = newArr3<FPfield>(&Phi_flat, grd->nxc, grd->nyc, grd->nzc);

  // allocate 3D arrays
  Exth = newArr3<FPfield>(&Exth_flat, grd->nxn, grd->nyn, grd->nzn);
  Eyth = newArr3<FPfield>(&Eyth_flat, grd->nxn, grd->nyn, grd->nzn);
  Ezth = newArr3<FPfield>(&Ezth_flat, grd->nxn, grd->nyn, grd->nzn);

  // B on centers
  Bxc = newArr3<FPfield>(&Bxc_flat, grd->nxc, grd->nyc, grd->nzc);
  Byc = newArr3<FPfield>(&Byc_flat, grd->nxc, grd->nyc, grd->nzc);
  Bzc = newArr3<FPfield>(&Bzc_flat, grd->nxc, grd->nyc, grd->nzc);
}

/** deallocate */
void EMfield_aux::deallocate(Grid* grd) {
  // Eth
  delArr3(Exth, grd->nxn, grd->nyn);
  delArr3(Eyth, grd->nxn, grd->nyn);
  delArr3(Ezth, grd->nxn, grd->nyn);

  // Bc
  delArr3(Bxc, grd->nxc, grd->nyc);
  delArr3(Byc, grd->nxc, grd->nyc);
  delArr3(Bzc, grd->nxc, grd->nyc);

  // Phi
  delArr3(Phi, grd->nxc, grd->nyc);
}
