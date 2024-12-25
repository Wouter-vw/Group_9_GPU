#include "EMfield.h"

/** allocate electric and magnetic field */
void field_allocate(struct grid* grd, struct EMfield* field) {
  // E on nodes
  field->electricField = newArr3<Vec3<FPfield>>(&field->electricField_flat,
                                                grd->nxn, grd->nyn, grd->nzn);
  // B on nodes
  field->Bxn = newArr3<FPfield>(&field->Bxn_flat, grd->nxn, grd->nyn, grd->nzn);
  field->Byn = newArr3<FPfield>(&field->Byn_flat, grd->nxn, grd->nyn, grd->nzn);
  field->Bzn = newArr3<FPfield>(&field->Bzn_flat, grd->nxn, grd->nyn, grd->nzn);
}

/** deallocate electric and magnetic field */
void field_deallocate(struct grid* grd, struct EMfield* field) {
  // E deallocate 3D arrays
  delArr3(field->electricField, grd->nxn, grd->nyn);

  // B deallocate 3D arrays
  delArr3(field->Bxn, grd->nxn, grd->nyn);
  delArr3(field->Byn, grd->nxn, grd->nyn);
  delArr3(field->Bzn, grd->nxn, grd->nyn);
}
