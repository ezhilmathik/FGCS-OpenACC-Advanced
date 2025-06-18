#ifndef FOLD3DPMM_H
#define FOLD3DPMM_H

#include <stdint.h>

#define MAXT 10e15*1.0
#define EDGE_GODUNOV_2D 1.0 
#define CBD
#define FOLD

#define UNROLLED

//#define SINGLE
#ifdef SINGLE
#define _DOUBLE_ float
#else
#define _DOUBLE_ double
#endif


//_DOUBLE_ DX, dy, dz;
//_DOUBLE_ DXYP, DXZP, DYZP, DXYZP;

////////////////////////////////////////////////////////////////////////////////
// input parameter for the 3 directions 
///////////////////////////////////////////////////////////////////////////////
#define _nx 128
#define _ny 128
#define _nz 128
#define HEIGHT 130 // N+2-> ghost node

#ifndef sign
#define sign(a) (a > 0) ? 1 : -1
#endif

// define total length of the 1 direction 
// _nx+2 => INT

#define _xmin 0.
#define _xmax 10.

#define _ymin 0.
#define _ymax 10.

#define _zmin 0.
#define _zmax 10.


#define _axx -0.35
#define _ayy  0.4
#define _azz  0.7
#define _Fc 1.1


////////////////////////////////////////////////////////////////////////////////
// CPU initialization helper function below
////////////////////////////////////////////////////////////////////////////////
_DOUBLE_ InitFoldFromPoint(_DOUBLE_ F, _DOUBLE_ ax,_DOUBLE_ ay,_DOUBLE_ az, 
			   _DOUBLE_ x, _DOUBLE_ y, _DOUBLE_ z);

_DOUBLE_ ImpSurfF(_DOUBLE_ x, _DOUBLE_ y, _DOUBLE_ z);


void InitNode(_DOUBLE_ x, _DOUBLE_ y, _DOUBLE_ z, int i, int j, int k, _DOUBLE_* T);


void updateVoxel(_DOUBLE_* T, int i, int j, int k, 
		 _DOUBLE_ xp, _DOUBLE_ yp, _DOUBLE_ zp, 
		 _DOUBLE_ dxx, _DOUBLE_ dyy, _DOUBLE_ dzz);

_DOUBLE_ x(int i, _DOUBLE_ DX);

_DOUBLE_ y(int j, _DOUBLE_ dy);

_DOUBLE_ z(int k, _DOUBLE_ dz);

////////////////////////////////////////////////////////////////////////////////
// CPU initialization starts from here
////////////////////////////////////////////////////////////////////////////////
#ifdef __cplusplus
extern "C" void ImplicitInitialiser(_DOUBLE_* T, int nx, int ny, int nz, 
				    _DOUBLE_ dxx,_DOUBLE_ dyy, _DOUBLE_ dzz);
#else
void ImplicitInitialiser(_DOUBLE_* T, int nx, int ny, int nz, 
			 _DOUBLE_ dxx,_DOUBLE_ dyy, _DOUBLE_ dzz);
#endif

#endif



