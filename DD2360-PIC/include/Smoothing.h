#ifndef SMOOTHING_H
#define SMOOTHING_H

#include "PrecisionTypes.h"
#include "Parameters.h"
#include "Grid.h"

/** Smmoth Interpolation Quantity defined on Center */
void smoothInterpScalarC(FPinterp***, Grid*, Parameters*);

/** Smmoth Interpolation Quantity defined on Nodes */
void smoothInterpScalarN(FPinterp***, Grid*, Parameters*);

#endif
