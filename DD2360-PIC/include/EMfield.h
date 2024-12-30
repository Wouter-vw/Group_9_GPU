#ifndef EMFIELD_H
#define EMFIELD_H

#include <nvtx3/nvtx3.hpp>

#include "Alloc.h"
#include "Grid.h"

/** structure with field information */
struct EMfield {
    // field arrays: 4D arrays

  /* Electric field defined on nodes: last index is component */
    // originally Ex, Ey, Ez
    Vec3<FPfield> ***electricField;
    Vec3<FPfield> *electricField_flat;
    /* Magnetic field defined on nodes: last index is component */
    // originally Bx, By, Bz
    Vec3<FPfield> ***magneticField;
    Vec3<FPfield> *magneticField_flat;
};

/** allocate electric and magnetic field */
void field_allocate(struct grid*, struct EMfield*);

/** deallocate electric and magnetic field */
void field_deallocate(struct grid*, struct EMfield*);

#endif