#include "EMfield.h"

/** allocate electric and magnetic field */
void field_allocate(struct grid* grd, struct EMfield* field) {
  // E on nodes
  field->electricField = newArr3<Vec3<FPfield>>(&field->electricField_flat,
                                                grd->nxn, grd->nyn, grd->nzn);
  // B on nodes
  field->magneticField = newArr3<Vec3<FPfield>>(&field->magneticField_flat,
                                                grd->nxn, grd->nyn, grd->nzn);
}

/** deallocate electric and magnetic field */
void field_deallocate(struct grid* grd, struct EMfield* field) {
  // E deallocate 3D arrays
  delArr3(field->electricField, grd->nxn, grd->nyn);

  // B deallocate 3D arrays
  delArr3(field->magneticField, grd->nxn, grd->nyn);
}