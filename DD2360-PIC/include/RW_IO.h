#ifndef RW_IO_H
#define RW_IO_H

#include "ConfigFile.h"
#include "input_array.h"

/** read the inputfile given via the command line */
void readInputFile(struct Parameters*, int, char**);

/** Print Simulation Parameters */
void printParameters(struct Parameters*);

/** Save Simulation Parameters */
void saveParameters(struct Parameters*);

void VTK_Write_Vectors(int, struct Grid*, struct EMfield*);

void VTK_Write_Scalars(int, struct Grid*, struct interpDensSpecies*, struct interpDensNet*);

#endif
