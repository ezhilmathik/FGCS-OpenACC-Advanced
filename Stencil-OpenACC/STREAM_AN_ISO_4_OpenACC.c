//-*-c++-*-
/*
  ============================================================================
  Name        : 3DparallelMarching.c
  Author      : Tor Gillberg           - seial c code
  Author      : Ezhilmathi Krishnasamy - parallel(CUDA) version for 4-GPU
  model       : Advanced GPU model (using streams)
  Description : Fold3DPMM.c ( has initialization implementaiton functions)
  Description : Fold3DPMM.h ( has input parameter for the grid size and etc,.)
  ( mimimun length shuould be >= 64; NOTE -> _nx=_ny=_nz should be equal)
  nvcc -O3 mi.cu
  or for _DOUBLE_ precision
  nvcc -arch sm_20 -O3
  optional flag to prefer larger shared memory and smaller L1 cache (suggestion from Mohammed)
  --ptxas-options=-v

  compilation : nvcc -arch=sm_35 -O3 -maxrregcount=64 xxxxxx.cu Fold3dPMM.c -Xcompiler -fopenmp
  ============================================================================
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <openacc.h>

// #define VTK_PRI // vtk for paraview 
//#define PRI
#include "Fold3dPMM.h"

#ifndef fmin
#define fmin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#ifndef sign
#define sign(a) (a > 0) ? 1 : -1
#endif

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

#pragma acc routine(FoldEdge2Points) seq
void FoldEdge2Points(double* tnew, const double st, const double xt, 
		     const double F, const double ax, const double ay, 
		     const double az, const double dxx, const double dzz)
{  
#ifdef OPTTEST
  if(ax > 0.0 && xt > st) 
    {
      return;
    }
#else
  if(*tnew < fmin(xt, st) || (ax > 0.0 && xt > st)) 
    {
      return;
    }
#endif
  
  double ga = F*F - ay*ay; 
  double dtx = (xt - st) / dxx;
  double c = (1.0 - dtx*ax);
  double sqrp = ga * (c*c - dtx * dtx * (ga - az*az));

  if(0.0 < sqrp)
    {
      c = (-az*c + sqrt(sqrp)) / (ga - az*az);
      sqrp = st + dzz*c;
      
      if(sqrp < *tnew && sqrp > fmin(st, xt)) 
	{
	  ga = sqrt((dtx * dtx + c*c) * F*F / ga);

	  double xe = -dzz * (F * dtx+ax * ga) / (F*c + az*ga);

	  xe = -dzz * (dtx+ax * ga) / (c+az * ga);
	  if (0.0 < xe && xe < dxx ) 
	    {
	      *tnew = sqrp;
	    }
	}
    }
  
  if(*tnew > xt) 
    {
      c = (-dxx*ax + dzz*az);
      ga = dxx*dxx + dzz*dzz;
      sqrp = (F*F - ax*ax - ay*ay - az*az) * ga + c*c;
      if(sqrp > 0.0) 
	{
	  c = c + sqrt(sqrp);
	  if (c > 0.0) 
	    {
	      *tnew = fmin(*tnew, xt + ga/c);
	    }
	}
    }
}


#pragma acc routine(FoldSurf3Points) seq
void FoldSurf3Points(double* tnew, const double st, 
		     const double xt, const double yt, const double F, 
		     const double ax, const double ay, const double az, 
		     const double dxx, const double dyy, const double dzz, 
		     const double dxz)
{

#ifdef OPTTEST

  if((ax > 0.0 && xt > st) || (ay > 0.0 && yt > xt)) 
    {
      return;
    }
  
#else
  
  if(*tnew < fmin(st, fmin(xt, yt)) || (ax > 0.0 && xt > st) || (ay > 0.0 && yt > xt)) 
    {
      return;
    }
  
#endif

  double dtx = (xt - st) / dxx;
  double dty = (yt - xt) / dyy;
  double dtxdtysq = dtx*dtx + dty*dty;
  double ga = F*F - az*az;
  double al = 1.0 - ax*dtx - ay*dty;
  double sqrp = al*al - dtxdtysq*ga;
  double xe, ye;

  if(sqrp > 0.0) 
    {
      al = (-az*al + F*sqrt(sqrp))/ga;
      ga = st + dzz*al;
      if(ga < *tnew && ga > fmin(fmin(st, xt), yt)) 
	{
	  sqrp = sqrt(dtxdtysq + al*al);
	  dtxdtysq = -dzz/(F*al + az*sqrp);
	  xe = dtxdtysq * (F*dtx + ax*sqrp);
	  ye = dtxdtysq * (F*dty + ay*sqrp);
	  if(0.0 <= fmin(xe, ye) && ye*dxx <= xe*dyy && xe <= dxx) 
	    {
	      *tnew = ga;
	    }
	}
    }
  if(*tnew > fmin(xt, yt) && (!(ax*dzz + az*dxx > 0.0 && xt > yt))) 
    {
      FoldEdge2Points(tnew, xt, yt, F, ay, (ax*dzz + az*dxx) / dxz,
		      (-ax*dxx + az*dzz) / dxz, dyy, dxz);
    }
}

//*******************************************//
// the Sweeps computaiton starts from here   //
// with 2 sweeps in each 3-direction         //
//*******************************************//


// sweeps from top to bottom in X - direction
void function_x_down(double *T, int i,   
		     double dzz, double dxx, double dyy, 
		     double dxy, double dxz, double dyz, 
		     double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  
  int j;//= (blockDim.y * blockIdx.y + threadIdx.y) + 1+1;
  int k;// = (blockDim.x * blockIdx.x + threadIdx.x) + 1;
  
  //if ((j >= 2 && j < WIDTH-2) && (k >= 1 && k < HEIGHT-1))
  //  {
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T) async(stream) 
#pragma acc loop collapse(2) private (j, k, ay, ax, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn) 
  for(j = 2; j < WIDTH-2; ++j)
    {
      for(k = 1; k < HEIGHT-1; ++k)
	{ 	  
	  tnew = fabs((T[(i+1) + HEIGHT * (j + WIDTH * k)]));
	  st =  fabs((T[i + HEIGHT * (j + WIDTH * k)]));
	  xt  = fabs((T[i + HEIGHT * (j + WIDTH * (k+1))]));
	  yt  = fabs((T[i + HEIGHT * ((j+1) + WIDTH * k)]));
	  xnt = fabs((T[i + HEIGHT * (j + WIDTH * (k-1))]));
	  ynt = fabs((T[i + HEIGHT * ((j-1) + WIDTH * k)]));
	  txm = fabs((T[i + HEIGHT * ((j-1) + WIDTH * (k+1))]));
	  txy = fabs((T[i + HEIGHT * ((j+1) + WIDTH * (k+1))]));
	  tym = fabs((T[i + HEIGHT * ((j+1) + WIDTH * (k-1))]));
	  txnyn = fabs((T[i + HEIGHT * ((j-1) + WIDTH * (k-1))]));
  
	  ay = _ayy; ax = _azz; az = _axx;	  
  
	  if (T[(i+1) + HEIGHT * (j + WIDTH * k)] < 0)
	    {
	      ax *= -1; 
	      ay *= -1; 
	      az *= -1;
	    }
  
	  if(st < tnew)
	    {
	      double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;
	      if (sqrp > 0.0)
		{
		  sqrp = az*dzz + sqrt(sqrp);
		  if(sqrp > 0.0) 
		    {
		      tnew = fmin(tnew, st + dzz*dzz/sqrp);
		    }
		}
	    }
#ifdef UNROLLED
	  if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	    {
	      FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	      FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	      FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	      FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	      FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	      FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	    }
#else
	  if(tnew > fmin(st, xt) || tnew > txy)
	    {
	      FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	    }
	  if(tnew > fmin(st, xt) || tnew > txm)
	    {
	      FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, xt)) 
		{ 
		  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
		}
	    }
	  if(tnew > fmin(st,yt) || tnew > txy)
	    {
	      FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, txy)) 
		{
		  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
				  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
		}
	    }
	  if(tnew > fmin(st, yt) || tnew > tym)
	    {
	      FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, yt))
		{
		  FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
		}
	    }
	  if(tnew > fmin(st,xnt) || tnew > tym)
	    {
	      FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, tym)) 
		{
		  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
				  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
		}
	    }
	  if(tnew > fmin(st, xnt) || tnew > txnyn)
	    {
	      FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, xnt))
		{
		  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
		}
	    }
	  if(tnew > fmin(st, ynt) || tnew > txnyn)
	    {
	      FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, txnyn)) 
		{
		  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
				  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
		}
	    }
	  if(tnew > fmin(st, ynt) || tnew > txm)
	    {
	      FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, ynt)) 
		{
		  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
		}
	      if(tnew > fmin(st, txm)) 
		{
		  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
				  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
		}
	    }
  
#endif
  
	  T[(i+1) + HEIGHT * (j + WIDTH * k)] = (sign(T[(i+1) + HEIGHT * (j + WIDTH * k)]) * 1.0) * tnew;
	} // IF  
    }
}



void function_x_t_down(double *T, int i,   
		       double dzz, double dxx, double dyy, 
		       double dxy, double dxz, double dyz, 
		       double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
{
	
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 

  unsigned int j;//  = (blockDim.y * blockIdx.y + threadIdx.y) + 1;
  unsigned int k;//  = (blockDim.x * blockIdx.x + threadIdx.x) + 1;

  //if ((j >= 1 && j < WIDTH-2) && (k >= 1 && k < HEIGHT-1))
  //  { 
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T) async(stream) 
#pragma acc loop collapse(2) private (j, k, ay, ax, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn) 
  for(j = 1; j < WIDTH-2; ++j)
    {
      for(k = 1; k < HEIGHT-1; ++k)
	{ 
	  tnew = fabs((T[(i+1) + HEIGHT * (j + WIDTH * k)]));
	  st =  fabs((T[i + HEIGHT * (j + WIDTH * k)]));
	  xt  = fabs((T[i + HEIGHT * (j + WIDTH * (k+1))]));
	  yt  = fabs((T[i + HEIGHT * ((j+1) + WIDTH * k)]));
	  xnt = fabs((T[i + HEIGHT * (j + WIDTH * (k-1))]));
	  ynt = fabs((T[i + HEIGHT * ((j-1) + WIDTH * k)]));
	  txm = fabs((T[i + HEIGHT * ((j-1) + WIDTH * (k+1))]));
	  txy = fabs((T[i + HEIGHT * ((j+1) + WIDTH * (k+1))]));
	  tym = fabs((T[i + HEIGHT * ((j+1) + WIDTH * (k-1))]));
	  txnyn = fabs((T[i + HEIGHT * ((j-1) + WIDTH * (k-1))]));
      
	  ay = _ayy; ax = _azz; az = _axx;	  
  
	  if (T[(i+1) + HEIGHT * (j + WIDTH * k)] < 0)
	    {
	      ax *= -1; 
	      ay *= -1; 
	      az *= -1;
	    }
  
	  if(st < tnew)
	    {
	      double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;
	      if (sqrp > 0.0)
		{
		  sqrp = az*dzz + sqrt(sqrp);
		  if(sqrp > 0.0) 
		    {
		      tnew = fmin(tnew, st + dzz*dzz/sqrp);
		    }
		}
	    }
#ifdef UNROLLED
	  if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	    {
	      FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	      FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	      FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	      FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	      FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	      FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	    }
#else
	  if(tnew > fmin(st, xt) || tnew > txy)
	    {
	      FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	    }
	  if(tnew > fmin(st, xt) || tnew > txm)
	    {
	      FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, xt)) 
		{ 
		  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
		}
	    }
	  if(tnew > fmin(st,yt) || tnew > txy)
	    {
	      FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, txy)) 
		{
		  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
				  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
		}
	    }
	  if(tnew > fmin(st, yt) || tnew > tym)
	    {
	      FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, yt))
		{
		  FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
		}
	    }
	  if(tnew > fmin(st,xnt) || tnew > tym)
	    {
	      FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, tym)) 
		{
		  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
				  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
		}
	    }
	  if(tnew > fmin(st, xnt) || tnew > txnyn)
	    {
	      FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, xnt))
		{
		  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
		}
	    }
	  if(tnew > fmin(st, ynt) || tnew > txnyn)
	    {
	      FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, txnyn)) 
		{
		  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
				  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
		}
	    }
	  if(tnew > fmin(st, ynt) || tnew > txm)
	    {
	      FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, ynt)) 
		{
		  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
		}
	      if(tnew > fmin(st, txm)) 
		{
		  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
				  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
		}
	    }
  
#endif
  
	  T[(i+1) + HEIGHT * (j + WIDTH * k)] = (sign(T[(i+1) + HEIGHT * (j + WIDTH * k)]) * 1.0) * tnew;
	} // IF  
    }
}


void function_x_b_down(double *T, int i,   
		       double dzz, double dxx, double dyy, 
		       double dxy, double dxz, double dyz, 
		       double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
{
	
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 

  int j;// = (blockDim.y * blockIdx.y + threadIdx.y) + 1 + 1;
  int k;// = (blockDim.x * blockIdx.x + threadIdx.x) + 1;

  //if ((j >= 2 && j < WIDTH-1) && (k >= 1 && k < HEIGHT-1))
  //  { 
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T) async(stream) 
#pragma acc loop collapse(2) private (j, k, ay, ax, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn) 
  for(j = 2; j < WIDTH-1; ++j)
    {
      for(k = 1; k < HEIGHT-1; ++k)
	{ 
	  tnew = fabs((T[(i+1) + HEIGHT * (j + WIDTH * k)]));
	  st =  fabs((T[i + HEIGHT * (j + WIDTH * k)]));
	  xt  = fabs((T[i + HEIGHT * (j + WIDTH * (k+1))]));
	  yt  = fabs((T[i + HEIGHT * ((j+1) + WIDTH * k)]));
	  xnt = fabs((T[i + HEIGHT * (j + WIDTH * (k-1))]));
	  ynt = fabs((T[i + HEIGHT * ((j-1) + WIDTH * k)]));
	  txm = fabs((T[i + HEIGHT * ((j-1) + WIDTH * (k+1))]));
	  txy = fabs((T[i + HEIGHT * ((j+1) + WIDTH * (k+1))]));
	  tym = fabs((T[i + HEIGHT * ((j+1) + WIDTH * (k-1))]));
	  txnyn = fabs((T[i + HEIGHT * ((j-1) + WIDTH * (k-1))]));
      
	  ay = _ayy; ax = _azz; az = _axx;	  
  
	  if (T[(i+1) + HEIGHT * (j + WIDTH * k)] < 0)
	    {
	      ax *= -1; 
	      ay *= -1; 
	      az *= -1;
	    }
  
	  if(st < tnew)
	    {
	      double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;
	      if (sqrp > 0.0)
		{
		  sqrp = az*dzz + sqrt(sqrp);
		  if(sqrp > 0.0) 
		    {
		      tnew = fmin(tnew, st + dzz*dzz/sqrp);
		    }
		}
	    }
#ifdef UNROLLED
	  if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	    {
	      FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	      FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	      FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	      FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	      FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	      FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	    }
#else
	  if(tnew > fmin(st, xt) || tnew > txy)
	    {
	      FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	    }
	  if(tnew > fmin(st, xt) || tnew > txm)
	    {
	      FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, xt)) 
		{ 
		  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
		}
	    }
	  if(tnew > fmin(st,yt) || tnew > txy)
	    {
	      FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, txy)) 
		{
		  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
				  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
		}
	    }
	  if(tnew > fmin(st, yt) || tnew > tym)
	    {
	      FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, yt))
		{
		  FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
		}
	    }
	  if(tnew > fmin(st,xnt) || tnew > tym)
	    {
	      FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, tym)) 
		{
		  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
				  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
		}
	    }
	  if(tnew > fmin(st, xnt) || tnew > txnyn)
	    {
	      FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, xnt))
		{
		  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
		}
	    }
	  if(tnew > fmin(st, ynt) || tnew > txnyn)
	    {
	      FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, txnyn)) 
		{
		  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
				  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
		}
	    }
	  if(tnew > fmin(st, ynt) || tnew > txm)
	    {
	      FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, ynt)) 
		{
		  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
		}
	      if(tnew > fmin(st, txm)) 
		{
		  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
				  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
		}
	    }
  
#endif
  
	  T[(i+1) + HEIGHT * (j + WIDTH * k)] = (sign(T[(i+1) + HEIGHT * (j + WIDTH * k)]) * 1.0) * tnew;
	} // IF  
    }
}


void function_x_1_down(double *T, double *a_1_2, int i,   
		       double dzz, double dxx, double dyy, 
		       double dxy, double dxz, double dyz, 
		       double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
{
	
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 

  int j;// = WIDTH-2;//(blockDim.y * blockIdx.y + threadIdx.y) + 1;
  int k;// = (blockDim.x * blockIdx.x + threadIdx.x) + 1;

  //if( j == (WIDTH-2) && ( k >= 1 && k < (HEIGHT-1)))
  //  {
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T, a_1_2) async(stream) 
#pragma acc loop private (j, k, ay, ax, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn) 
  //  for(j = 1; j < WIDTH-2; ++j)
  //    {
  for(k = 1; k < HEIGHT-1; ++k)
    { 
      j = (WIDTH-2);
      tnew = fabs((T[(i+1) + HEIGHT * (j + WIDTH * k)]));
      st =  fabs((T[i + HEIGHT * (j + WIDTH * k)]));
      xt  = fabs((T[i + HEIGHT * (j + WIDTH * (k+1))]));
      yt  = fabs((T[i + HEIGHT * ((j+1) + WIDTH * k)]));
      xnt = fabs((T[i + HEIGHT * (j + WIDTH * (k-1))]));
      ynt = fabs((T[i + HEIGHT * ((j-1) + WIDTH * k)]));
      txm = fabs((T[i + HEIGHT * ((j-1) + WIDTH * (k+1))]));
      txy = fabs((T[i + HEIGHT * ((j+1) + WIDTH * (k+1))]));
      tym = fabs((T[i + HEIGHT * ((j+1) + WIDTH * (k-1))]));
      txnyn = fabs((T[i + HEIGHT * ((j-1) + WIDTH * (k-1))]));
      
      ay = _ayy; ax = _azz; az = _axx;	  
   
      if (T[(i+1) + HEIGHT * (j + WIDTH * k)] < 0)
	{
	  ax *= -1; 
	  ay *= -1; 
	  az *= -1;
	}
  
      if(st < tnew)
	{
	  double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;
	  if (sqrp > 0.0)
	    {
	      sqrp = az*dzz + sqrt(sqrp);
	      if(sqrp > 0.0) 
		{
		  tnew = fmin(tnew, st + dzz*dzz/sqrp);
		}
	    }
	}
#ifdef UNROLLED
      if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	}
#else
      if(tnew > fmin(st, xt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	}
      if(tnew > fmin(st, xt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xt)) 
	    { 
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st,yt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txy)) 
	    {
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, yt))
	    {
	      FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	    }
	}
      if(tnew > fmin(st,xnt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, tym)) 
	    {
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, xnt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xnt))
	    {
	      FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txnyn)) 
	    {
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, ynt)) 
	    {
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	    }
	  if(tnew > fmin(st, txm)) 
	    {
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
  
#endif
  
      T[(i+1) + HEIGHT * (j + WIDTH * k)] = (sign(T[(i+1) + HEIGHT * (j + WIDTH * k)]) * 1.0) * tnew;
      a_1_2[k-1] = T[(i+1) + HEIGHT * (j + WIDTH * k)];
    }
  
}


void function_x_2_down(double *T, double *a_2_1, int i,   
		       double dzz, double dxx, double dyy, 
		       double dxy, double dxz, double dyz, 
		       double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
{
	
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 

  int j;// = 1;//(blockDim.y * blockIdx.y + threadIdx.y) + 1;
  int k;// = (blockDim.x * blockIdx.x + threadIdx.x) + 1;
 

  //if( j == 1 && ( k >= 1 && k < (HEIGHT-1)))
  //{
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T, a_2_1) async(stream) 
#pragma acc loop private (j,k, ay, ax, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn) 
  //  for(j = 1; j < WIDTH-2; ++j)
  //    {
  for(k = 1; k < HEIGHT-1; ++k)
    {     
      j = 1;
      tnew = fabs((T[(i+1) + HEIGHT * (j + WIDTH * k)]));
      st =  fabs((T[i + HEIGHT * (j + WIDTH * k)]));
      xt  = fabs((T[i + HEIGHT * (j + WIDTH * (k+1))]));
      yt  = fabs((T[i + HEIGHT * ((j+1) + WIDTH * k)]));
      xnt = fabs((T[i + HEIGHT * (j + WIDTH * (k-1))]));
      ynt = fabs((T[i + HEIGHT * ((j-1) + WIDTH * k)]));
      txm = fabs((T[i + HEIGHT * ((j-1) + WIDTH * (k+1))]));
      txy = fabs((T[i + HEIGHT * ((j+1) + WIDTH * (k+1))]));
      tym = fabs((T[i + HEIGHT * ((j+1) + WIDTH * (k-1))]));
      txnyn = fabs((T[i + HEIGHT * ((j-1) + WIDTH * (k-1))]));
      ay = _ayy; ax = _azz; az = _axx;	  
     
      if (T[(i+1) + HEIGHT * (j + WIDTH * k)] < 0)
	{
	  ax *= -1; 
	  ay *= -1; 
	  az *= -1;
	}
  
      if(st < tnew)
	{
	  double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;
	  if (sqrp > 0.0)
	    {
	      sqrp = az*dzz + sqrt(sqrp);
	      if(sqrp > 0.0) 
		{
		  tnew = fmin(tnew, st + dzz*dzz/sqrp);
		}
	    }
	}
#ifdef UNROLLED
      if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	}
#else
      if(tnew > fmin(st, xt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	}
      if(tnew > fmin(st, xt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xt)) 
	    { 
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st,yt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txy)) 
	    {
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, yt))
	    {
	      FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	    }
	}
      if(tnew > fmin(st,xnt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, tym)) 
	    {
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, xnt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xnt))
	    {
	      FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txnyn)) 
	    {
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, ynt)) 
	    {
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	    }
	  if(tnew > fmin(st, txm)) 
	    {
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
  
#endif
  
      T[(i+1) + HEIGHT * (j + WIDTH * k)] = (sign(T[(i+1) + HEIGHT * (j + WIDTH * k)]) * 1.0) * tnew;
      a_2_1[k-1] = T[(i+1) + HEIGHT * (j + WIDTH * k)];

    }
}


// sweeps from bottom to top in X- direction

void function_x_up(double *T, int i,   
		   double dzz, double dxx, double dyy, 
		   double dxy, double dxz, double dyz, 
		   double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
{
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  
  int j;// = (blockDim.y * blockIdx.y + threadIdx.y) + 1+1;
  int k;// = (blockDim.x * blockIdx.x + threadIdx.x) + 1;
  
  //if ((j >= 2 && j < WIDTH-2) && (k >= 1 && k < HEIGHT-1))
 //   { 
        acc_set_device_num(ID, acc_device_nvidia);
    #pragma acc parallel deviceptr(T) async(stream) 
#pragma acc loop collapse(2) private (j,k, ay, ax, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn) 
  for(j = 2; j < WIDTH-2; ++j)
    {
      for(k = 1; k < HEIGHT-1; ++k)
	{ 
      tnew = fabs((T[(i-1) + HEIGHT * (j + WIDTH * k)]));
      st =  fabs((T[i + HEIGHT * (j + WIDTH * k)]));
      xt  = fabs((T[i + HEIGHT * (j + WIDTH * (k-1))]));
      yt  = fabs((T[i + HEIGHT * ((j-1) + WIDTH * k)]));
      xnt = fabs((T[i + HEIGHT * (j + WIDTH * (k+1))]));
      ynt = fabs((T[i + HEIGHT * ((j+1) + WIDTH * k)]));
      txm = fabs((T[i + HEIGHT * ((j+1) + WIDTH * (k-1))]));
      txy = fabs((T[i + HEIGHT * ((j-1) + WIDTH * (k-1))]));
      tym = fabs((T[i + HEIGHT * ((j-1) + WIDTH* (k+1))]));
      txnyn = fabs((T[i + HEIGHT * ((j+1) + WIDTH * (k+1))]));
  
	  ay = - _ayy; 	ax = - _azz;   	az = - _axx;
  
      if (T[(i-1) + HEIGHT * (j + WIDTH * k)] < 0)
	{
	  ax *= -1; 
	  ay *= -1; 
	  az *= -1;
	}
  
      if(st < tnew)
	{
	  double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

	  if (sqrp > 0.0)
	    {
	      sqrp = az*dzz + sqrt(sqrp);

	      if(sqrp > 0.0) 
		{
		  tnew = fmin(tnew, st + dzz*dzz/sqrp);
		}
	    }
	}
#ifdef UNROLLED
      if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
      
	  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	}
#else
      if(tnew > fmin(st, xt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	}
      if(tnew > fmin(st, xt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xt)) 
	    { 
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txy)) 
	    {
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st,yt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, yt))
	    {
	      FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	    }
	}
      if(tnew > fmin(st,xnt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, tym)) 
	    {
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st,xnt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xnt))
	    {
	      FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txnyn)) 
	    {
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, ynt)) 
	    {
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	    }
	  if(tnew > fmin(st, txm)) 
	    {
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
#endif
  
      T[(i-1) + HEIGHT* (j + WIDTH * k)] = (sign(T[(i-1) + HEIGHT * (j + WIDTH * k)])*1.0)*tnew;
    } // IF
}
}




void function_x_t_up(double *T, int i,   
		     double dzz, double dxx, double dyy, 
		     double dxy, double dxz, double dyz, 
		     double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
{
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  
  int j;// = (blockDim.y * blockIdx.y + threadIdx.y) + 1;
  int k;// = (blockDim.x * blockIdx.x + threadIdx.x) + 1;

  //if ((j >= 1 && j < WIDTH-2) && (k >= 1 && k < HEIGHT-1))
  //  { 
        acc_set_device_num(ID, acc_device_nvidia);
    #pragma acc parallel deviceptr(T) async(stream) 
#pragma acc loop collapse(2) private (j,k, ay, ax, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn) 
  for(j = 1; j < WIDTH-2; ++j)
    {
      for(k = 1; k < HEIGHT-1; ++k)
	{ 
      tnew = fabs((T[(i-1) + HEIGHT * (j + WIDTH * k)]));
      st =  fabs((T[i + HEIGHT * (j + WIDTH * k)]));
      xt  = fabs((T[i + HEIGHT * (j + WIDTH * (k-1))]));
      yt  = fabs((T[i + HEIGHT * ((j-1) + WIDTH * k)]));
      xnt = fabs((T[i + HEIGHT * (j + WIDTH * (k+1))]));
      ynt = fabs((T[i + HEIGHT * ((j+1) + WIDTH * k)]));
      txm = fabs((T[i + HEIGHT * ((j+1) + WIDTH * (k-1))]));
      txy = fabs((T[i + HEIGHT * ((j-1) + WIDTH * (k-1))]));
      tym = fabs((T[i + HEIGHT * ((j-1) + WIDTH* (k+1))]));
      txnyn = fabs((T[i + HEIGHT * ((j+1) + WIDTH * (k+1))]));
  
	  ay = - _ayy; 	ax = - _azz;   	az = - _axx;
  
      if (T[(i-1) + HEIGHT * (j + WIDTH * k)] < 0)
	{
	  ax *= -1; 
	  ay *= -1; 
	  az *= -1;
	}
  
      if(st < tnew)
	{
	  double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

	  if (sqrp > 0.0)
	    {
	      sqrp = az*dzz + sqrt(sqrp);

	      if(sqrp > 0.0) 
		{
		  tnew = fmin(tnew, st + dzz*dzz/sqrp);
		}
	    }
	}
#ifdef UNROLLED
      if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
      
	  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	}
#else
      if(tnew > fmin(st, xt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	}
      if(tnew > fmin(st, xt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xt)) 
	    { 
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txy)) 
	    {
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st,yt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, yt))
	    {
	      FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	    }
	}
      if(tnew > fmin(st,xnt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, tym)) 
	    {
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st,xnt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xnt))
	    {
	      FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txnyn)) 
	    {
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, ynt)) 
	    {
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	    }
	  if(tnew > fmin(st, txm)) 
	    {
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
#endif
  
      T[(i-1) + HEIGHT* (j + WIDTH * k)] = (sign(T[(i-1) + HEIGHT * (j + WIDTH * k)])*1.0)*tnew;
    } // IF
}
}




void function_x_b_up(double *T, int i,   
		     double dzz, double dxx, double dyy, 
		     double dxy, double dxz, double dyz, 
		     double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
{
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  
  int j;// = (blockDim.y * blockIdx.y + threadIdx.y) + 1+1;
  int k;// = (blockDim.x * blockIdx.x + threadIdx.x) + 1;
  
  //if ((j >= 2 && j < WIDTH-1) && (k >= 1 && k < HEIGHT-1))
  //  { 
          acc_set_device_num(ID, acc_device_nvidia);
    #pragma acc parallel deviceptr(T) async(stream) 
#pragma acc loop collapse(2) private (j,k, ay, ax, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn) 
  for(j = 2; j < WIDTH-1; ++j)
    {
      for(k = 1; k < HEIGHT-1; ++k)
	{ 

      tnew = fabs((T[(i-1) + HEIGHT * (j + WIDTH * k)]));
      st =  fabs((T[i + HEIGHT * (j + WIDTH * k)]));
      xt  = fabs((T[i + HEIGHT * (j + WIDTH * (k-1))]));
      yt  = fabs((T[i + HEIGHT * ((j-1) + WIDTH * k)]));
      xnt = fabs((T[i + HEIGHT * (j + WIDTH * (k+1))]));
      ynt = fabs((T[i + HEIGHT * ((j+1) + WIDTH * k)]));
      txm = fabs((T[i + HEIGHT * ((j+1) + WIDTH * (k-1))]));
      txy = fabs((T[i + HEIGHT * ((j-1) + WIDTH * (k-1))]));
      tym = fabs((T[i + HEIGHT * ((j-1) + WIDTH* (k+1))]));
      txnyn = fabs((T[i + HEIGHT * ((j+1) + WIDTH * (k+1))]));
  
	  ay = - _ayy; 	ax = - _azz;   	az = - _axx;
  
      if (T[(i-1) + HEIGHT * (j + WIDTH * k)] < 0)
	{
	  ax *= -1; 
	  ay *= -1; 
	  az *= -1;
	}
  
      if(st < tnew)
	{
	  double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

	  if (sqrp > 0.0)
	    {
	      sqrp = az*dzz + sqrt(sqrp);

	      if(sqrp > 0.0) 
		{
		  tnew = fmin(tnew, st + dzz*dzz/sqrp);
		}
	    }
	}
#ifdef UNROLLED
      if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
      
	  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	}
#else
      if(tnew > fmin(st, xt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	}
      if(tnew > fmin(st, xt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xt)) 
	    { 
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txy)) 
	    {
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st,yt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, yt))
	    {
	      FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	    }
	}
      if(tnew > fmin(st,xnt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, tym)) 
	    {
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st,xnt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xnt))
	    {
	      FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txnyn)) 
	    {
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, ynt)) 
	    {
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	    }
	  if(tnew > fmin(st, txm)) 
	    {
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
#endif
  
      T[(i-1) + HEIGHT* (j + WIDTH * k)] = (sign(T[(i-1) + HEIGHT * (j + WIDTH * k)])*1.0)*tnew;
    }// IF
}
}



void function_x_1_up(double *T, double *a_1_2, int i,   
		     double dzz, double dxx, double dyy, 
		     double dxy, double dxz, double dyz, 
		     double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
{
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  
  int j;// = WIDTH-2;
  int k;// = (blockDim.x * blockIdx.x + threadIdx.x) + 1;
  

  //if( j == (WIDTH-2) && ( k >= 1 && k < (HEIGHT-1)))
    //{
            acc_set_device_num(ID, acc_device_nvidia);
    #pragma acc parallel deviceptr(T, a_1_2) async(stream) 
#pragma acc loop private (j,k, ay, ax, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn) 
//  for(j = 2; j < WIDTH-1; ++j)
//    {
      for(k = 1; k < HEIGHT-1; ++k)
	{ 
	j = (WIDTH-2);
      tnew = fabs((T[(i-1) + HEIGHT * (j + WIDTH * k)]));
      st =  fabs((T[i + HEIGHT * (j + WIDTH * k)]));
      xt  = fabs((T[i + HEIGHT * (j + WIDTH * (k-1))]));
      yt  = fabs((T[i + HEIGHT * ((j-1) + WIDTH * k)]));
      xnt = fabs((T[i + HEIGHT * (j + WIDTH * (k+1))]));
      ynt = fabs((T[i + HEIGHT * ((j+1) + WIDTH * k)]));
      txm = fabs((T[i + HEIGHT * ((j+1) + WIDTH * (k-1))]));
      txy = fabs((T[i + HEIGHT * ((j-1) + WIDTH * (k-1))]));
      tym = fabs((T[i + HEIGHT * ((j-1) + WIDTH* (k+1))]));
      txnyn = fabs((T[i + HEIGHT * ((j+1) + WIDTH * (k+1))]));
  
	  ay = - _ayy; 	ax = - _azz;   	az = - _axx;
  
      if (T[(i-1) + HEIGHT * (j + WIDTH * k)] < 0)
	{
	  ax *= -1; 
	  ay *= -1; 
	  az *= -1;
	}
  
      if(st < tnew)
	{
	  double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

	  if (sqrp > 0.0)
	    {
	      sqrp = az*dzz + sqrt(sqrp);

	      if(sqrp > 0.0) 
		{
		  tnew = fmin(tnew, st + dzz*dzz/sqrp);
		}
	    }
	}
#ifdef UNROLLED
      if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
      
	  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	}
#else
      if(tnew > fmin(st, xt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	}
      if(tnew > fmin(st, xt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xt)) 
	    { 
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txy)) 
	    {
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st,yt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, yt))
	    {
	      FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	    }
	}
      if(tnew > fmin(st,xnt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, tym)) 
	    {
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st,xnt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xnt))
	    {
	      FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txnyn)) 
	    {
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, ynt)) 
	    {
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	    }
	  if(tnew > fmin(st, txm)) 
	    {
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
#endif
  
      T[(i-1) + HEIGHT* (j + WIDTH * k)] = (sign(T[(i-1) + HEIGHT * (j + WIDTH * k)])*1.0)*tnew;
      a_1_2[k-1] = T[(i-1) + HEIGHT* (j + WIDTH * k)];
    }
}



void function_x_2_up(double *T, double *a_2_1, int i,   
		     double dzz, double dxx, double dyy, 
		     double dxy, double dxz, double dyz, 
		     double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
{
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  
  int j;// = 1;//(blockDim.y * blockIdx.y + threadIdx.y) + 1;
  int k;// = (blockDim.x * blockIdx.x + threadIdx.x) + 1;
  

  //if( j == 1 && ( k >= 1 && k < (HEIGHT-1)))
   // {
             acc_set_device_num(ID, acc_device_nvidia);
    #pragma acc parallel deviceptr(T, a_2_1) async(stream) 
#pragma acc loop private (j,k, ay, ax, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn) 
//  for(j = 2; j < WIDTH-1; ++j)
 //   {
      for(k = 1; k < HEIGHT-1; ++k)
	{ 
j=1;
      tnew = fabs((T[(i-1) + HEIGHT * (j + WIDTH * k)]));
      st =  fabs((T[i + HEIGHT * (j + WIDTH * k)]));
      xt  = fabs((T[i + HEIGHT * (j + WIDTH * (k-1))]));
      yt  = fabs((T[i + HEIGHT * ((j-1) + WIDTH * k)]));
      xnt = fabs((T[i + HEIGHT * (j + WIDTH * (k+1))]));
      ynt = fabs((T[i + HEIGHT * ((j+1) + WIDTH * k)]));
      txm = fabs((T[i + HEIGHT * ((j+1) + WIDTH * (k-1))]));
      txy = fabs((T[i + HEIGHT * ((j-1) + WIDTH * (k-1))]));
      tym = fabs((T[i + HEIGHT * ((j-1) + WIDTH* (k+1))]));
      txnyn = fabs((T[i + HEIGHT * ((j+1) + WIDTH * (k+1))]));
  
	  ay = - _ayy; 	ax = - _azz;   	az = - _axx;
  
      if (T[(i-1) + HEIGHT * (j + WIDTH * k)] < 0)
	{
	  ax *= -1; 
	  ay *= -1; 
	  az *= -1;
	}
  
      if(st < tnew)
	{
	  double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

	  if (sqrp > 0.0)
	    {
	      sqrp = az*dzz + sqrt(sqrp);

	      if(sqrp > 0.0) 
		{
		  tnew = fmin(tnew, st + dzz*dzz/sqrp);
		}
	    }
	}
#ifdef UNROLLED
      if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
      
	  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	}
#else
      if(tnew > fmin(st, xt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	}
      if(tnew > fmin(st, xt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xt)) 
	    { 
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txy)) 
	    {
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st,yt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, yt))
	    {
	      FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	    }
	}
      if(tnew > fmin(st,xnt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, tym)) 
	    {
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st,xnt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xnt))
	    {
	      FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txnyn)) 
	    {
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, ynt)) 
	    {
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	    }
	  if(tnew > fmin(st, txm)) 
	    {
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
#endif
  
      T[(i-1) + HEIGHT* (j + WIDTH * k)] = (sign(T[(i-1) + HEIGHT * (j + WIDTH * k)])*1.0)*tnew;
      a_2_1[k-1] = T[(i-1) + HEIGHT* (j + WIDTH * k)];
    }
}




// sweeps from bottom to top from Y -direction

void function_y_up(double *T, int j,   
		   double dzz, double dxx, double dyy, 
		   double dxy, double dxz, double dyz, 
		   double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn;
  
  int i;// = (blockDim.y * blockIdx.y + threadIdx.y)+1+1;
  int k;// = (blockDim.x * blockIdx.x + threadIdx.x)+1;  
 
  //if ((i >= 2 && i < WIDTH-2) && (k >= 1 && k < HEIGHT-1))
  //  { 
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T) async(stream) //num_gangs(_nx) num_workers(64)
#pragma acc loop collapse(2) private (i, k, ay, ax, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
  for(i = 2; i < WIDTH-2; ++i)
    {
      for(k = 1; k < HEIGHT-1; ++k)
        {
	  tnew = fabs((T[i + WIDTH * ((j-1) + HEIGHT*k)]));
	  st =  fabs((T[i + WIDTH * (j + HEIGHT * k)]));
	  xt  = fabs((T[(i-1) + WIDTH * (j + HEIGHT * k)]));
	  yt  = fabs((T[i + WIDTH * (j + HEIGHT * (k-1))]));
	  xnt = fabs((T[(i+1) + WIDTH * (j + HEIGHT * k)]));
	  ynt = fabs((T[i + WIDTH * (j + HEIGHT * (k+1))]));
	  txm = fabs((T[(i-1)+ WIDTH * (j + HEIGHT * (k+1))]));
	  txy = fabs((T[(i-1) + WIDTH * (j + HEIGHT * (k-1))]));
	  tym = fabs((T[(i+1) + WIDTH * (j + HEIGHT * (k-1))]));
	  txnyn = fabs((T[(i+1) + WIDTH * (j + HEIGHT * (k+1))]));
  
	  ay = - _azz; ax = - _axx; az = - _ayy;  	

	  if (T[i + WIDTH * ((j-1) + HEIGHT * k)] < 0)
	    {
	      ax *= -1; 
	      ay *= -1; 
	      az *= -1;
	    }
  
	  if(st < tnew)
	    {
	      double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

	      if (sqrp > 0.0)
		{
		  sqrp = az*dzz + sqrt(sqrp);

		  if(sqrp > 0.0) 
		    {
		      tnew = fmin(tnew, st + dzz*dzz/sqrp);
		    }
		}
	    }
#ifdef UNROLLED
	  if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	    {
	      FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	      FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	      FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	      FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
      
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	      FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	      FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
#else
	  if(tnew > fmin(st, xt) || tnew > txy)
	    {
	      FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	    }
	  if(tnew > fmin(st, xt) || tnew > txm)
	    {
	      FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, xt)) 
		{ 
		  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
		}
	    }
	  if(tnew > fmin(st, yt) || tnew > txy)
	    {
	      FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, txy)) 
		{
		  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
				  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
		}
	    }
	  if(tnew > fmin(st, yt) || tnew > tym)
	    {
	      FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, yt))
		{
		  FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
		}
	    }
	  if(tnew > fmin(st, xnt) || tnew > tym)
	    {
	      FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, tym)) 
		{
		  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
				  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
		}
	    }
  
	  if(tnew > fmin(st, xnt) || tnew > txnyn)
	    {
	      FoldSurf3Points(&tnew, st, xnt,txnyn,F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, xnt))
		{
		  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
		}
	    }
	  if(tnew > fmin(st, ynt) || tnew > txnyn)
	    {
	      FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, txnyn)) 
		{
		  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
				  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
		}
	    }
	  if(tnew > fmin(st,ynt) || tnew > txm)
	    {
	      FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, ynt)) 
		{
		  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
		}
	      if(tnew > fmin(st, txm)) 
		{
		  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
				  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
		}
	    }
  
#endif
  
	  T[i + WIDTH * ((j-1) + HEIGHT * k)] = (sign(T[i + WIDTH * ((j-1) + HEIGHT * k)])*1.0)*tnew;
	} // IF
    }
}


void function_y_t_up(double *T, int j,   
		     double dzz, double dxx, double dyy, 
		     double dxy, double dxz, double dyz, 
		     double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn;
  
  int i;// = (blockDim.y * blockIdx.y + threadIdx.y)+1;
  int k;// = (blockDim.x * blockIdx.x + threadIdx.x)+1;  


  //if ((i >= 1 && i < WIDTH-2) && (k >= 1 && k < HEIGHT-1))
  // { 
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T) async(stream) //  num_gangs(_nx) num_workers(64)
#pragma acc loop collapse(2) private (i, k, ay, ax, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
  for(i = 1; i < WIDTH-2; ++i)
    {
      for(k = 1; k < HEIGHT-1; ++k)
        {
  
	  tnew = fabs((T[i + WIDTH * ((j-1) + HEIGHT*k)]));
	  st =  fabs((T[i + WIDTH * (j + HEIGHT * k)]));
	  xt  = fabs((T[(i-1) + WIDTH * (j + HEIGHT * k)]));
	  yt  = fabs((T[i + WIDTH * (j + HEIGHT * (k-1))]));
	  xnt = fabs((T[(i+1) + WIDTH * (j + HEIGHT * k)]));
	  ynt = fabs((T[i + WIDTH * (j + HEIGHT * (k+1))]));
	  txm = fabs((T[(i-1)+ WIDTH * (j + HEIGHT * (k+1))]));
	  txy = fabs((T[(i-1) + WIDTH * (j + HEIGHT * (k-1))]));
	  tym = fabs((T[(i+1) + WIDTH * (j + HEIGHT * (k-1))]));
	  txnyn = fabs((T[(i+1) + WIDTH * (j + HEIGHT * (k+1))]));
  
	  ay = - _azz; ax = - _axx; az = - _ayy;  	
  
	  if (T[i + WIDTH * ((j-1) + HEIGHT * k)] < 0)
	    {
	      ax *= -1; 
	      ay *= -1; 
	      az *= -1;
	    }
  
	  if(st < tnew)
	    {
	      double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

	      if (sqrp > 0.0)
		{
		  sqrp = az*dzz + sqrt(sqrp);

		  if(sqrp > 0.0) 
		    {
		      tnew = fmin(tnew, st + dzz*dzz/sqrp);
		    }
		}
	    }
#ifdef UNROLLED
	  if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	    {
	      FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	      FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	      FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	      FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
      
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	      FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	      FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
#else
	  if(tnew > fmin(st, xt) || tnew > txy)
	    {
	      FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	    }
	  if(tnew > fmin(st, xt) || tnew > txm)
	    {
	      FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, xt)) 
		{ 
		  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
		}
	    }
	  if(tnew > fmin(st, yt) || tnew > txy)
	    {
	      FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, txy)) 
		{
		  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
				  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
		}
	    }
	  if(tnew > fmin(st, yt) || tnew > tym)
	    {
	      FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, yt))
		{
		  FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
		}
	    }
	  if(tnew > fmin(st, xnt) || tnew > tym)
	    {
	      FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, tym)) 
		{
		  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
				  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
		}
	    }
  
	  if(tnew > fmin(st, xnt) || tnew > txnyn)
	    {
	      FoldSurf3Points(&tnew, st, xnt,txnyn,F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, xnt))
		{
		  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
		}
	    }
	  if(tnew > fmin(st, ynt) || tnew > txnyn)
	    {
	      FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, txnyn)) 
		{
		  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
				  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
		}
	    }
	  if(tnew > fmin(st,ynt) || tnew > txm)
	    {
	      FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, ynt)) 
		{
		  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
		}
	      if(tnew > fmin(st, txm)) 
		{
		  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
				  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
		}
	    }
  
#endif
  
	  T[i + WIDTH * ((j-1) + HEIGHT * k)] = (sign(T[i + WIDTH * ((j-1) + HEIGHT * k)])*1.0)*tnew;
	} // IF
    }
}



void function_y_b_up(double *T, int j,   
		     double dzz, double dxx, double dyy, 
		     double dxy, double dxz, double dyz, 
		     double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn;
  
  int i;// = (blockDim.y * blockIdx.y + threadIdx.y)+1+1;
  int k;// = (blockDim.x * blockIdx.x + threadIdx.x)+1;  
 

  //if ((i >= 2 && i < WIDTH-1) && (k >= 1 && k < HEIGHT-1))
  // { 
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T) async(stream) //async(stream)  num_gangs(_nx) num_workers(64)
#pragma acc loop collapse(2) private (i, k, ay, ax, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
  for(i = 2; i < WIDTH-1; ++i)
    {
      for(k = 1; k < HEIGHT-1; ++k)
        {

	  tnew = fabs((T[i + WIDTH * ((j-1) + HEIGHT*k)]));
	  st =  fabs((T[i + WIDTH * (j + HEIGHT * k)]));
	  xt  = fabs((T[(i-1) + WIDTH * (j + HEIGHT * k)]));
	  yt  = fabs((T[i + WIDTH * (j + HEIGHT * (k-1))]));
	  xnt = fabs((T[(i+1) + WIDTH * (j + HEIGHT * k)]));
	  ynt = fabs((T[i + WIDTH * (j + HEIGHT * (k+1))]));
	  txm = fabs((T[(i-1)+ WIDTH * (j + HEIGHT * (k+1))]));
	  txy = fabs((T[(i-1) + WIDTH * (j + HEIGHT * (k-1))]));
	  tym = fabs((T[(i+1) + WIDTH * (j + HEIGHT * (k-1))]));
	  txnyn = fabs((T[(i+1) + WIDTH * (j + HEIGHT * (k+1))]));
  
	  ay = - _azz; ax = - _axx; az = - _ayy;  	
  
	  if (T[i + WIDTH * ((j-1) + HEIGHT * k)] < 0)
	    {
	      ax *= -1; 
	      ay *= -1; 
	      az *= -1;
	    }
  
	  if(st < tnew)
	    {
	      double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

	      if (sqrp > 0.0)
		{
		  sqrp = az*dzz + sqrt(sqrp);

		  if(sqrp > 0.0) 
		    {
		      tnew = fmin(tnew, st + dzz*dzz/sqrp);
		    }
		}
	    }
#ifdef UNROLLED
	  if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	    {
	      FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	      FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	      FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	      FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
      
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	      FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	      FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
#else
	  if(tnew > fmin(st, xt) || tnew > txy)
	    {
	      FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	    }
	  if(tnew > fmin(st, xt) || tnew > txm)
	    {
	      FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, xt)) 
		{ 
		  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
		}
	    }
	  if(tnew > fmin(st, yt) || tnew > txy)
	    {
	      FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, txy)) 
		{
		  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
				  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
		}
	    }
	  if(tnew > fmin(st, yt) || tnew > tym)
	    {
	      FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, yt))
		{
		  FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
		}
	    }
	  if(tnew > fmin(st, xnt) || tnew > tym)
	    {
	      FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, tym)) 
		{
		  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
				  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
		}
	    }
  
	  if(tnew > fmin(st, xnt) || tnew > txnyn)
	    {
	      FoldSurf3Points(&tnew, st, xnt,txnyn,F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, xnt))
		{
		  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
		}
	    }
	  if(tnew > fmin(st, ynt) || tnew > txnyn)
	    {
	      FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, txnyn)) 
		{
		  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
				  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
		}
	    }
	  if(tnew > fmin(st,ynt) || tnew > txm)
	    {
	      FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, ynt)) 
		{
		  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
		}
	      if(tnew > fmin(st, txm)) 
		{
		  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
				  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
		}
	    }
  
#endif
  
	  T[i + WIDTH * ((j-1) + HEIGHT * k)] = (sign(T[i + WIDTH * ((j-1) + HEIGHT * k)])*1.0)*tnew;
	} // IF
    }
}



void function_y_1_up(double *T, double *a_1_2, int j,   
		     double dzz, double dxx, double dyy, 
		     double dxy, double dxz, double dyz, 
		     double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn;
  
  int i;// = WIDTH-2;
  int k;// = (blockDim.x * blockIdx.x + threadIdx.x)+1;  
  

  //if( i == (WIDTH-2) && ( k >= 1 && k < (HEIGHT-1)))
  //  {
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T, a_1_2) async(stream) //async(stream)  num_gangs(_nx) num_workers(64)
#pragma acc loop private (i, k, ay, ax, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
  //  for(i = 1; i < WIDTH-2; ++i)
  //    {
  for(k = 1; k < HEIGHT-1; ++k)
    {
      i=WIDTH-2;
      tnew = fabs((T[i + WIDTH * ((j-1) + HEIGHT*k)]));
      st =  fabs((T[i + WIDTH * (j + HEIGHT * k)]));
      xt  = fabs((T[(i-1) + WIDTH * (j + HEIGHT * k)]));
      yt  = fabs((T[i + WIDTH * (j + HEIGHT * (k-1))]));
      xnt = fabs((T[(i+1) + WIDTH * (j + HEIGHT * k)]));
      ynt = fabs((T[i + WIDTH * (j + HEIGHT * (k+1))]));
      txm = fabs((T[(i-1)+ WIDTH * (j + HEIGHT * (k+1))]));
      txy = fabs((T[(i-1) + WIDTH * (j + HEIGHT * (k-1))]));
      tym = fabs((T[(i+1) + WIDTH * (j + HEIGHT * (k-1))]));
      txnyn = fabs((T[(i+1) + WIDTH * (j + HEIGHT * (k+1))]));
  
      ay = - _azz; ax = - _axx; az = - _ayy;  	
  
      if (T[i + WIDTH * ((j-1) + HEIGHT * k)] < 0)
	{
	  ax *= -1; 
	  ay *= -1; 
	  az *= -1;
	}
  
      if(st < tnew)
	{
	  double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

	  if (sqrp > 0.0)
	    {
	      sqrp = az*dzz + sqrt(sqrp);

	      if(sqrp > 0.0) 
		{
		  tnew = fmin(tnew, st + dzz*dzz/sqrp);
		}
	    }
	}
#ifdef UNROLLED
      if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
      
	  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	}
#else
      if(tnew > fmin(st, xt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	}
      if(tnew > fmin(st, xt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xt)) 
	    { 
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txy)) 
	    {
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, yt))
	    {
	      FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	    }
	}
      if(tnew > fmin(st, xnt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, tym)) 
	    {
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	    }
	}
  
      if(tnew > fmin(st, xnt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, xnt,txnyn,F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xnt))
	    {
	      FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txnyn)) 
	    {
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st,ynt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, ynt)) 
	    {
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	    }
	  if(tnew > fmin(st, txm)) 
	    {
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
  
#endif
  
      T[i + WIDTH * ((j-1) + HEIGHT * k)] = (sign(T[i + WIDTH * ((j-1) + HEIGHT * k)])*1.0)*tnew;
      a_1_2[k-1] = T[i + WIDTH * ((j-1) + HEIGHT * k)];
    }
}



void function_y_2_up(double *T, double *a_2_1, int j,   
		     double dzz, double dxx, double dyy, 
		     double dxy, double dxz, double dyz, 
		     double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn;
  
  int i;// = 1;//(blockDim.y * blockIdx.y + threadIdx.y)+1;
  int k;// = (blockDim.x * blockIdx.x + threadIdx.x)+1;  
  
  //if( i == 1 && ( k >= 1 && k < (HEIGHT-1)))
  // {
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T, a_2_1) async(stream)  //async(stream)  num_gangs(_nx) num_workers(64)
#pragma acc loop private (i, k, ay, ax, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
  // for(i = 1; i < WIDTH-2; ++i)
  //    {
  for(k = 1; k < HEIGHT-1; ++k)
    {
      i=1;
      tnew = fabs((T[i + WIDTH * ((j-1) + HEIGHT*k)]));
      st =  fabs((T[i + WIDTH * (j + HEIGHT * k)]));
      xt  = fabs((T[(i-1) + WIDTH * (j + HEIGHT * k)]));
      yt  = fabs((T[i + WIDTH * (j + HEIGHT * (k-1))]));
      xnt = fabs((T[(i+1) + WIDTH * (j + HEIGHT * k)]));
      ynt = fabs((T[i + WIDTH * (j + HEIGHT * (k+1))]));
      txm = fabs((T[(i-1)+ WIDTH * (j + HEIGHT * (k+1))]));
      txy = fabs((T[(i-1) + WIDTH * (j + HEIGHT * (k-1))]));
      tym = fabs((T[(i+1) + WIDTH * (j + HEIGHT * (k-1))]));
      txnyn = fabs((T[(i+1) + WIDTH * (j + HEIGHT * (k+1))]));
  
      ay = - _azz; ax = - _axx; az = - _ayy;  	
  
      if (T[i + WIDTH * ((j-1) + HEIGHT * k)] < 0)
	{
	  ax *= -1; 
	  ay *= -1; 
	  az *= -1;
	}
  
      if(st < tnew)
	{
	  double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

	  if (sqrp > 0.0)
	    {
	      sqrp = az*dzz + sqrt(sqrp);

	      if(sqrp > 0.0) 
		{
		  tnew = fmin(tnew, st + dzz*dzz/sqrp);
		}
	    }
	}
#ifdef UNROLLED
      if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
      
	  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	}
#else
      if(tnew > fmin(st, xt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	}
      if(tnew > fmin(st, xt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xt)) 
	    { 
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txy)) 
	    {
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, yt))
	    {
	      FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	    }
	}
      if(tnew > fmin(st, xnt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, tym)) 
	    {
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	    }
	}
  
      if(tnew > fmin(st, xnt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, xnt,txnyn,F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xnt))
	    {
	      FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txnyn)) 
	    {
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st,ynt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, ynt)) 
	    {
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	    }
	  if(tnew > fmin(st, txm)) 
	    {
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
  
#endif
  
      T[i + WIDTH * ((j-1) + HEIGHT * k)] = (sign(T[i + WIDTH * ((j-1) + HEIGHT * k)])*1.0)*tnew;
      a_2_1[k-1] = T[i + WIDTH * ((j-1) + HEIGHT * k)];
    }
}


// sweeps from the top to bottom in Y- direction
void function_y_down(double *T, int j,   
		     double dzz, double dxx, double dyy, 
		     double dxy, double dxz, double dyz, 
		     double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  
  int i;// = (blockDim.y * blockIdx.y + threadIdx.y)+1 + 1;
  int k;// = (blockDim.x * blockIdx.x + threadIdx.x)+1;
  

  //if ((i >= 2 && i < WIDTH-2) && (k >= 1 && k < HEIGHT-1))
  //  { 
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T) async(stream) //async(stream)  num_gangs(_nx) num_workers(64)
#pragma acc loop collapse(2) private (i, k, ay, ax, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
  for(i = 2; i < WIDTH-2; ++i)
    {
      for(k = 1; k < HEIGHT-1; ++k)
        {
	  tnew = fabs((T[i + WIDTH * ((j+1) + HEIGHT * k)]));
	  st =  fabs((T[i + WIDTH * (j + HEIGHT * k)]));
	  xt  = fabs((T[(i+1) + WIDTH * (j + HEIGHT * k)]));
	  yt  = fabs((T[i + WIDTH * (j + HEIGHT * (k+1))]));
	  xnt = fabs((T[(i-1) + WIDTH * (j + HEIGHT * k)]));
	  ynt = fabs((T[i + WIDTH * (j + HEIGHT * (k-1))]));
	  txm =  fabs((T[(i+1) + WIDTH * (j + HEIGHT * (k-1))]));
	  txy =  fabs((T[(i+1) + WIDTH * (j + HEIGHT * (k+1))]));
	  tym =  fabs((T[(i-1) + WIDTH * (j + HEIGHT * (k+1))]));
	  txnyn =fabs((T[(i-1) + WIDTH * (j + HEIGHT * (k-1))]));
	  ay = _azz; ax = _axx; az = _ayy;  	
  
	  if (T[i + WIDTH * ((j+1) + HEIGHT * k)] < 0)
	    {
	      ax *= -1; 
	      ay *= -1; 
	      az *= -1;
	    }
  
	  if(st < tnew)
	    {
	      double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

	      if (sqrp > 0.0)
		{
		  sqrp = az*dzz + sqrt(sqrp);

		  if(sqrp > 0.0) 
		    {
		      tnew = fmin(tnew, st + dzz*dzz / sqrp);
		    }
		}
	    }
#ifdef UNROLLED
	  if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	    {
	      FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	      FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	      FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	      FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	      FoldEdge2Points(&tnew,st, yt, F, ay, ax, az, dyy, dzz);
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	      FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
#else
	  if(tnew > fmin(st, xt) || tnew > txy)
	    {
	      FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	    }
	  if(tnew > fmin(st, xt) || tnew > txm)
	    {
	      FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, xt)) 
		{ 
		  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
		}
	    } 
	  if(tnew > fmin(st, yt) || tnew > txy)
	    {
	      FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, txy)) 
		{
		  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy) / dxy, 
				  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
		}
	    }
	  if(tnew > fmin(st, yt) || tnew > tym)
	    {
	      FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, yt))
		{
		  FoldEdge2Points(&tnew,st, yt, F, ay, ax, az, dyy, dzz);
		}
	    }
	  if(tnew > fmin(st, xnt) || tnew > tym)
	    {
	      FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, tym)) 
		{
		  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
				  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
		}
	    }
	  if(tnew > fmin(st, xnt) || tnew > txnyn)
	    {
	      FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, xnt))
		{
		  FoldEdge2Points(&tnew,st, xnt, F, -ax, ay, az, dxx, dzz);
		}
	    }
	  if(tnew > fmin(st,ynt) || tnew > txnyn)
	    {
	      FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, txnyn)) 
		{
		  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
				  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
		}
	    }
	  if(tnew > fmin(st,ynt) || tnew > txm)
	    {
	      FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, ynt)) 
		{
		  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
		}
	      if(tnew>fmin(st, txm)) 
		{
		  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
				  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
		}
	    }
  
#endif
  
	  T[i + WIDTH * ((j+1) + HEIGHT * k)] = (sign(T[i + WIDTH * ((j+1) + HEIGHT * k)])*1.0)*tnew;
	} //IF  
    }
}




void function_y_t_down(double *T, int j,   
		       double dzz, double dxx, double dyy, 
		       double dxy, double dxz, double dyz, 
		       double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  
  int i;// = (blockDim.y * blockIdx.y + threadIdx.y)+1;
  int k;// = (blockDim.x * blockIdx.x + threadIdx.x)+1;
  

  //if ((i >= 1 && i < WIDTH-2) && (k >= 1 && k < HEIGHT-1))
  //  { 
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T) async(stream) //async(stream)  num_gangs(_nx) num_workers(64)
#pragma acc loop collapse(2) private (i,k, ax, ay, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
  for(i = 1; i < WIDTH-2; ++i)
    {
      for(k = 1; k < HEIGHT-1; ++k)
	{

	  tnew = fabs((T[i + WIDTH * ((j+1) + HEIGHT * k)]));
	  st =  fabs((T[i + WIDTH * (j + HEIGHT * k)]));
	  xt  = fabs((T[(i+1) + WIDTH * (j + HEIGHT * k)]));
	  yt  = fabs((T[i + WIDTH * (j + HEIGHT * (k+1))]));
	  xnt = fabs((T[(i-1) + WIDTH * (j + HEIGHT * k)]));
	  ynt = fabs((T[i + WIDTH * (j + HEIGHT * (k-1))]));
	  txm =  fabs((T[(i+1) + WIDTH * (j + HEIGHT * (k-1))]));
	  txy =  fabs((T[(i+1) + WIDTH * (j + HEIGHT * (k+1))]));
	  tym =  fabs((T[(i-1) + WIDTH * (j + HEIGHT * (k+1))]));
	  txnyn =fabs((T[(i-1) + WIDTH * (j + HEIGHT * (k-1))]));
	  ay = _azz; ax = _axx; az = _ayy;  	
  
	  if (T[i + WIDTH * ((j+1) + HEIGHT * k)] < 0)
	    {
	      ax *= -1; 
	      ay *= -1; 
	      az *= -1;
	    }
  
	  if(st < tnew)
	    {
	      double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

	      if (sqrp > 0.0)
		{
		  sqrp = az*dzz + sqrt(sqrp);

		  if(sqrp > 0.0) 
		    {
		      tnew = fmin(tnew, st + dzz*dzz / sqrp);
		    }
		}
	    }
#ifdef UNROLLED
	  if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	    {
	      FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	      FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	      FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	      FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	      FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	      FoldEdge2Points(&tnew,st, yt, F, ay, ax, az, dyy, dzz);
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	      FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
#else
	  if(tnew > fmin(st, xt) || tnew > txy)
	    {
	      FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	    }
	  if(tnew > fmin(st, xt) || tnew > txm)
	    {
	      FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, xt)) 
		{ 
		  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
		}
	    } 
	  if(tnew > fmin(st, yt) || tnew > txy)
	    {
	      FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, txy)) 
		{
		  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy) / dxy, 
				  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
		}
	    }
	  if(tnew > fmin(st, yt) || tnew > tym)
	    {
	      FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, yt))
		{
		  FoldEdge2Points(&tnew,st, yt, F, ay, ax, az, dyy, dzz);
		}
	    }
	  if(tnew > fmin(st, xnt) || tnew > tym)
	    {
	      FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, tym)) 
		{
		  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
				  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
		}
	    }
	  if(tnew > fmin(st, xnt) || tnew > txnyn)
	    {
	      FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, xnt))
		{
		  FoldEdge2Points(&tnew,st, xnt, F, -ax, ay, az, dxx, dzz);
		}
	    }
	  if(tnew > fmin(st,ynt) || tnew > txnyn)
	    {
	      FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, txnyn)) 
		{
		  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
				  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
		}
	    }
	  if(tnew > fmin(st,ynt) || tnew > txm)
	    {
	      FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, ynt)) 
		{
		  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
		}
	      if(tnew>fmin(st, txm)) 
		{
		  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
				  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
		}
	    }
  
#endif
  
	  T[i + WIDTH * ((j+1) + HEIGHT * k)] = (sign(T[i + WIDTH * ((j+1) + HEIGHT * k)])*1.0)*tnew;
	} // IF  
    }
}



void function_y_b_down(double *T, int j,   
		       double dzz, double dxx, double dyy, 
		       double dxy, double dxz, double dyz, 
		       double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  
  int i;// = (blockDim.y * blockIdx.y + threadIdx.y)+1+1;
  int k;// = (blockDim.x * blockIdx.x + threadIdx.x)+1;


  //if ((i >= 2 && i < WIDTH-1) && (k >= 1 && k < HEIGHT-1))
  //  { 
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T) async(stream) //async(stream)  num_gangs(_nx) num_workers(64)
#pragma acc loop collapse(2) private (i,k, ax, ay, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
  for(i = 2; i < WIDTH-1; ++i)
    {
      for(k = 1; k < HEIGHT-1; ++k)
        {

      tnew = fabs((T[i + WIDTH * ((j+1) + HEIGHT * k)]));
      st =  fabs((T[i + WIDTH * (j + HEIGHT * k)]));
      xt  = fabs((T[(i+1) + WIDTH * (j + HEIGHT * k)]));
      yt  = fabs((T[i + WIDTH * (j + HEIGHT * (k+1))]));
      xnt = fabs((T[(i-1) + WIDTH * (j + HEIGHT * k)]));
      ynt = fabs((T[i + WIDTH * (j + HEIGHT * (k-1))]));
      txm =  fabs((T[(i+1) + WIDTH * (j + HEIGHT * (k-1))]));
      txy =  fabs((T[(i+1) + WIDTH * (j + HEIGHT * (k+1))]));
      tym =  fabs((T[(i-1) + WIDTH * (j + HEIGHT * (k+1))]));
      txnyn =fabs((T[(i-1) + WIDTH * (j + HEIGHT * (k-1))]));
	  ay = _azz; ax = _axx; az = _ayy;  	
  
      if (T[i + WIDTH * ((j+1) + HEIGHT * k)] < 0)
	{
	  ax *= -1; 
	  ay *= -1; 
	  az *= -1;
	}
  
      if(st < tnew)
	{
	  double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

	  if (sqrp > 0.0)
	    {
	      sqrp = az*dzz + sqrt(sqrp);

	      if(sqrp > 0.0) 
		{
		  tnew = fmin(tnew, st + dzz*dzz / sqrp);
		}
	    }
	}
#ifdef UNROLLED
      if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew,st, yt, F, ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	}
#else
      if(tnew > fmin(st, xt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	}
      if(tnew > fmin(st, xt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xt)) 
	    { 
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	    }
	} 
      if(tnew > fmin(st, yt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txy)) 
	    {
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy) / dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, yt))
	    {
	      FoldEdge2Points(&tnew,st, yt, F, ay, ax, az, dyy, dzz);
	    }
	}
      if(tnew > fmin(st, xnt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, tym)) 
	    {
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, xnt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xnt))
	    {
	      FoldEdge2Points(&tnew,st, xnt, F, -ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st,ynt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txnyn)) 
	    {
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st,ynt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, ynt)) 
	    {
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	    }
	  if(tnew>fmin(st, txm)) 
	    {
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	    }
	}
  
#endif
  
      T[i + WIDTH * ((j+1) + HEIGHT * k)] = (sign(T[i + WIDTH * ((j+1) + HEIGHT * k)])*1.0)*tnew;
    } // IF 
}
}



void function_y_1_down(double *T, double *a_1_2, int j,   
		       double dzz, double dxx, double dyy, 
		       double dxy, double dxz, double dyz, 
		       double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  
  int i;// = WIDTH-2;
  int k;// = (blockDim.x * blockIdx.x + threadIdx.x)+1;
  //if( i == (WIDTH-2) && ( k >= 1 && k < (HEIGHT-1)))
  //  {
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T, a_1_2) async(stream) //async(stream)  num_gangs(_nx) num_workers(64)
#pragma acc loop private (i,k, ax, ay, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
//  for(i = 1; i < WIDTH-2; ++i)
//    {
      for(k = 1; k < HEIGHT-1; ++k)
        {
i=WIDTH-2;
      tnew = fabs((T[i + WIDTH * ((j+1) + HEIGHT * k)]));
      st =  fabs((T[i + WIDTH * (j + HEIGHT * k)]));
      xt  = fabs((T[(i+1) + WIDTH * (j + HEIGHT * k)]));
      yt  = fabs((T[i + WIDTH * (j + HEIGHT * (k+1))]));
      xnt = fabs((T[(i-1) + WIDTH * (j + HEIGHT * k)]));
      ynt = fabs((T[i + WIDTH * (j + HEIGHT * (k-1))]));
      txm =  fabs((T[(i+1) + WIDTH * (j + HEIGHT * (k-1))]));
      txy =  fabs((T[(i+1) + WIDTH * (j + HEIGHT * (k+1))]));
      tym =  fabs((T[(i-1) + WIDTH * (j + HEIGHT * (k+1))]));
      txnyn =fabs((T[(i-1) + WIDTH * (j + HEIGHT * (k-1))]));
	  ay = _azz; ax = _axx; az = _ayy;  	
  
      if (T[i + WIDTH * ((j+1) + HEIGHT * k)] < 0)
	{
	  ax *= -1; 
	  ay *= -1; 
	  az *= -1;
	}
  
      if(st < tnew)
	{
	  double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

	  if (sqrp > 0.0)
	    {
	      sqrp = az*dzz + sqrt(sqrp);

	      if(sqrp > 0.0) 
		{
		  tnew = fmin(tnew, st + dzz*dzz / sqrp);
		}
	    }
	}
#ifdef UNROLLED
      if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew,st, yt, F, ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	}
#else
      if(tnew > fmin(st, xt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	}
      if(tnew > fmin(st, xt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xt)) 
	    { 
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	    }
	} 
      if(tnew > fmin(st, yt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txy)) 
	    {
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy) / dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, yt))
	    {
	      FoldEdge2Points(&tnew,st, yt, F, ay, ax, az, dyy, dzz);
	    }
	}
      if(tnew > fmin(st, xnt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, tym)) 
	    {
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, xnt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xnt))
	    {
	      FoldEdge2Points(&tnew,st, xnt, F, -ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st,ynt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txnyn)) 
	    {
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st,ynt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, ynt)) 
	    {
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	    }
	  if(tnew>fmin(st, txm)) 
	    {
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	    }
	}
  
#endif
  
      T[i + WIDTH * ((j+1) + HEIGHT * k)] = (sign(T[i + WIDTH * ((j+1) + HEIGHT * k)])*1.0)*tnew;
      a_1_2[k-1] = T[i + WIDTH * ((j+1) + HEIGHT * k)];
    }
}


void function_y_2_down(double *T, double *a_2_1, int j,   
		       double dzz, double dxx, double dyy, 
		       double dxy, double dxz, double dyz, 
		       double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  
  int i;// = 1;
  int k;// = (blockDim.x * blockIdx.x + threadIdx.x)+1;
  
  //if( i == 1 && ( k >= 1 && k < (HEIGHT-1)))
  //  {
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T, a_2_1) async(stream) //async(stream)  num_gangs(_nx) num_workers(64)
#pragma acc loop private (i,k, ax, ay, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
//  for(i = 1; i < WIDTH-2; ++i)
//    {
      for(k = 1; k < HEIGHT-1; ++k)
        {
i=1;
      tnew = fabs((T[i + WIDTH * ((j+1) + HEIGHT * k)]));
      st =  fabs((T[i + WIDTH * (j + HEIGHT * k)]));
      xt  = fabs((T[(i+1) + WIDTH * (j + HEIGHT * k)]));
      yt  = fabs((T[i + WIDTH * (j + HEIGHT * (k+1))]));
      xnt = fabs((T[(i-1) + WIDTH * (j + HEIGHT * k)]));
      ynt = fabs((T[i + WIDTH * (j + HEIGHT * (k-1))]));
      txm =  fabs((T[(i+1) + WIDTH * (j + HEIGHT * (k-1))]));
      txy =  fabs((T[(i+1) + WIDTH * (j + HEIGHT * (k+1))]));
      tym =  fabs((T[(i-1) + WIDTH * (j + HEIGHT * (k+1))]));
      txnyn =fabs((T[(i-1) + WIDTH * (j + HEIGHT * (k-1))]));
	  ay = _azz; ax = _axx; az = _ayy;  	
  
      if (T[i + WIDTH * ((j+1) + HEIGHT * k)] < 0)
	{
	  ax *= -1; 
	  ay *= -1; 
	  az *= -1;
	}
  
      if(st < tnew)
	{
	  double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

	  if (sqrp > 0.0)
	    {
	      sqrp = az*dzz + sqrt(sqrp);

	      if(sqrp > 0.0) 
		{
		  tnew = fmin(tnew, st + dzz*dzz / sqrp);
		}
	    }
	}
#ifdef UNROLLED
      if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew,st, yt, F, ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	}
#else
      if(tnew > fmin(st, xt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	}
      if(tnew > fmin(st, xt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xt)) 
	    { 
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	    }
	} 
      if(tnew > fmin(st, yt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txy)) 
	    {
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy) / dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, yt))
	    {
	      FoldEdge2Points(&tnew,st, yt, F, ay, ax, az, dyy, dzz);
	    }
	}
      if(tnew > fmin(st, xnt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, tym)) 
	    {
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, xnt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xnt))
	    {
	      FoldEdge2Points(&tnew,st, xnt, F, -ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st,ynt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txnyn)) 
	    {
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st,ynt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, ynt)) 
	    {
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	    }
	  if(tnew>fmin(st, txm)) 
	    {
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
	    }
	}
  
#endif
  
      T[i + WIDTH * ((j+1) + HEIGHT * k)] = (sign(T[i + WIDTH * ((j+1) + HEIGHT * k)])*1.0)*tnew;
      a_2_1[k-1] = T[i + WIDTH * ((j+1) + HEIGHT * k)];
    }
}


// sweeps from bottom to top in the Z- direction
void function_z_up(double *T, int k,   
		   double dzz, double dxx, double dyy, 
		   double dxy, double dxz, double dyz, 
		   double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
  
{ 
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  
  int i;// = (blockDim.x * blockIdx.x + threadIdx.x)+1;
  int j;// = (blockDim.y * blockIdx.y + threadIdx.y)+1+1;

 // if ((j >= 2 && j < WIDTH-2) && (i >= 1 && i < HEIGHT-1))
  //  { 
acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T) async(stream) //async(stream)  num_gangs(_nx) num_workers(64)
#pragma acc loop collapse(2) private (i,j, ax, ay, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
  for(j = 2; j < WIDTH-2; ++j)
    {
      for(i = 1; i < HEIGHT-1; ++i)
        {
      tnew = fabs((T[i + HEIGHT * (j + WIDTH * (k+1))]));
      st =  fabs((T[i + HEIGHT * (j + WIDTH * k)]));
      xt  = fabs((T[i + HEIGHT * ((j+1) + WIDTH * k)]));
      yt  = fabs((T[(i+1) + HEIGHT * (j + WIDTH * k)]));
      xnt = fabs((T[i + HEIGHT * ((j-1) + WIDTH * k)]));
      ynt = fabs((T[(i-1) + HEIGHT * (j + WIDTH * k)]));
      txm =  fabs((T[(i-1) + HEIGHT * ((j+1) + WIDTH * k)]));
      txy =  fabs((T[(i+1) + HEIGHT * ((j+1) + WIDTH * k)]));
      tym =  fabs((T[(i+1) + HEIGHT * ((j-1) +WIDTH * k)]));
      txnyn =fabs((T[(i-1) + HEIGHT * ((j-1) + WIDTH * k)]));
      ay = _axx; ax = _ayy; az = _azz;
   
      if (T[i + HEIGHT * (j + WIDTH * (k+1))] < 0)
	{
	  ax *= -1; 
	  ay *= -1; 
	  az *= -1;
	}
      if(st < tnew)
	{
	  double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

	  if (sqrp > 0.0)
	    {
	      sqrp = az*dzz + sqrt(sqrp);

	      if(sqrp > 0.0) 
		{
		  tnew = fmin(tnew, st+dzz * dzz/sqrp);
		}
	    }
	}
#ifdef UNROLLED
      if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	}
#else
      if(tnew > fmin(st, xt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	}
      if(tnew > fmin(st, xt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xt)) 
	    { 
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txy)) 
	    {
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, yt))
	    {
	      FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	    }
	}
      if(tnew > fmin(st, xnt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, tym))
	    {
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, xnt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xnt))
	    {
	      FoldEdge2Points(&tnew,st, xnt, F, -ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txnyn))
	    {
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy) / dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, ynt)) 
	    {
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	    }
	  if(tnew > fmin(st, txm)) 
	    {
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
#endif
      T[i + HEIGHT * (j + WIDTH * (k+1))] = (1.0*sign(T[i + HEIGHT * (j + WIDTH * (k+1))]))*tnew;
    } // IF
}
}



void function_z_t_up(double *T, int k,   
		     double dzz, double dxx, double dyy, 
		     double dxy, double dxz, double dyz, 
		     double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
  
{ 
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  
  int i;// = (blockDim.x * blockIdx.x + threadIdx.x)+1;
  int j;// = (blockDim.y * blockIdx.y + threadIdx.y)+1;


  //if ((j >= 1 && j < WIDTH-2) && (i >= 1 && i < HEIGHT-1))
  //  { 
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T) async(stream) //async(stream)  num_gangs(_nx) num_workers(64)
#pragma acc loop collapse(2) private (i,j, ax, ay, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
  for(j = 1; j < WIDTH-2; ++j)
    {
      for(i = 1; i < HEIGHT-1; ++i)
        {
      tnew = fabs((T[i + HEIGHT * (j + WIDTH * (k+1))]));
      st =  fabs((T[i + HEIGHT * (j + WIDTH * k)]));
      xt  = fabs((T[i + HEIGHT * ((j+1) + WIDTH * k)]));
      yt  = fabs((T[(i+1) + HEIGHT * (j + WIDTH * k)]));
      xnt = fabs((T[i + HEIGHT * ((j-1) + WIDTH * k)]));
      ynt = fabs((T[(i-1) + HEIGHT * (j + WIDTH * k)]));
      txm =  fabs((T[(i-1) + HEIGHT * ((j+1) + WIDTH * k)]));
      txy =  fabs((T[(i+1) + HEIGHT * ((j+1) + WIDTH * k)]));
      tym =  fabs((T[(i+1) + HEIGHT * ((j-1) +WIDTH * k)]));
      txnyn =fabs((T[(i-1) + HEIGHT * ((j-1) + WIDTH * k)]));
	  ay = _axx; ax = _ayy; az = _azz;
   
      if (T[i + HEIGHT * (j + WIDTH * (k+1))] < 0)
	{
	  ax *= -1; 
	  ay *= -1; 
	  az *= -1;
	}
      if(st < tnew)
	{
	  double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

	  if (sqrp > 0.0)
	    {
	      sqrp = az*dzz + sqrt(sqrp);

	      if(sqrp > 0.0) 
		{
		  tnew = fmin(tnew, st+dzz * dzz/sqrp);
		}
	    }
	}
#ifdef UNROLLED
      if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	}
#else
      if(tnew > fmin(st, xt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	}
      if(tnew > fmin(st, xt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xt)) 
	    { 
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txy)) 
	    {
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, yt))
	    {
	      FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	    }
	}
      if(tnew > fmin(st, xnt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, tym))
	    {
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, xnt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xnt))
	    {
	      FoldEdge2Points(&tnew,st, xnt, F, -ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txnyn))
	    {
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy) / dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, ynt)) 
	    {
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	    }
	  if(tnew > fmin(st, txm)) 
	    {
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
#endif
      T[i + HEIGHT * (j + WIDTH * (k+1))] = (1.0*sign(T[i + HEIGHT * (j + WIDTH * (k+1))]))*tnew;
    } //IF
}
}




void function_z_b_up(double *T, int k,   
		     double dzz, double dxx, double dyy, 
		     double dxy, double dxz, double dyz, 
		     double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
  
{ 
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  
  int i;// = (blockDim.x * blockIdx.x + threadIdx.x)+1;
  int j;// = (blockDim.y * blockIdx.y + threadIdx.y)+1+1;

  //if ((j >= 2 && j < WIDTH-1) && (i >= 1 && i < HEIGHT-1))
   // { 
   acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T) async(stream) //async(stream)  num_gangs(_nx) num_workers(64)
#pragma acc loop collapse(2) private (i,j, ax, ay, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
  for(j = 2; j < WIDTH-1; ++j)
    {
      for(i = 1; i < HEIGHT-1; ++i)
        {
      tnew = fabs((T[i + HEIGHT * (j + WIDTH * (k+1))]));
      st =  fabs((T[i + HEIGHT * (j + WIDTH * k)]));
      xt  = fabs((T[i + HEIGHT * ((j+1) + WIDTH * k)]));
      yt  = fabs((T[(i+1) + HEIGHT * (j + WIDTH * k)]));
      xnt = fabs((T[i + HEIGHT * ((j-1) + WIDTH * k)]));
      ynt = fabs((T[(i-1) + HEIGHT * (j + WIDTH * k)]));
      txm =  fabs((T[(i-1) + HEIGHT * ((j+1) + WIDTH * k)]));
      txy =  fabs((T[(i+1) + HEIGHT * ((j+1) + WIDTH * k)]));
      tym =  fabs((T[(i+1) + HEIGHT * ((j-1) +WIDTH * k)]));
      txnyn =fabs((T[(i-1) + HEIGHT * ((j-1) + WIDTH * k)]));
	  ay = _axx; ax = _ayy; az = _azz;
   
      if (T[i + HEIGHT * (j + WIDTH * (k+1))] < 0)
	{
	  ax *= -1; 
	  ay *= -1; 
	  az *= -1;
	}
      if(st < tnew)
	{
	  double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

	  if (sqrp > 0.0)
	    {
	      sqrp = az*dzz + sqrt(sqrp);

	      if(sqrp > 0.0) 
		{
		  tnew = fmin(tnew, st+dzz * dzz/sqrp);
		}
	    }
	}
#ifdef UNROLLED
      if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	}
#else
      if(tnew > fmin(st, xt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	}
      if(tnew > fmin(st, xt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xt)) 
	    { 
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txy)) 
	    {
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, yt))
	    {
	      FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	    }
	}
      if(tnew > fmin(st, xnt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, tym))
	    {
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, xnt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xnt))
	    {
	      FoldEdge2Points(&tnew,st, xnt, F, -ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txnyn))
	    {
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy) / dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, ynt)) 
	    {
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	    }
	  if(tnew > fmin(st, txm)) 
	    {
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
#endif
      T[i + HEIGHT * (j + WIDTH * (k+1))] = (1.0*sign(T[i + HEIGHT * (j + WIDTH * (k+1))]))*tnew;
    } // IF
}
}



void function_z_1_up(double *T, double *a_1_2, int k,   
		     double dzz, double dxx, double dyy, 
		     double dxy, double dxz, double dyz, 
		     double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
  
{ 
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  
  int i;// = (blockDim.x * blockIdx.x + threadIdx.x)+1;
  int j;// = WIDTH-2;

  //if( j == (WIDTH-2) && ( i >= 1 && i < (HEIGHT-1)))
  //  {
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T, a_1_2) async(stream) //async(stream)  num_gangs(_nx) num_workers(64)
#pragma acc loop private (i,j, ax, ay, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
//  for(j = 2; j < WIDTH-2; ++j)
 //   {
      for(i = 1; i < HEIGHT-1; ++i)
        {
        j=WIDTH-2;
      tnew = fabs((T[i + HEIGHT * (j + WIDTH * (k+1))]));
      st =  fabs((T[i + HEIGHT * (j + WIDTH * k)]));
      xt  = fabs((T[i + HEIGHT * ((j+1) + WIDTH * k)]));
      yt  = fabs((T[(i+1) + HEIGHT * (j + WIDTH * k)]));
      xnt = fabs((T[i + HEIGHT * ((j-1) + WIDTH * k)]));
      ynt = fabs((T[(i-1) + HEIGHT * (j + WIDTH * k)]));
      txm =  fabs((T[(i-1) + HEIGHT * ((j+1) + WIDTH * k)]));
      txy =  fabs((T[(i+1) + HEIGHT * ((j+1) + WIDTH * k)]));
      tym =  fabs((T[(i+1) + HEIGHT * ((j-1) +WIDTH * k)]));
      txnyn =fabs((T[(i-1) + HEIGHT * ((j-1) + WIDTH * k)]));
	  ay = _axx; ax = _ayy; az = _azz;
   
      if (T[i + HEIGHT * (j + WIDTH * (k+1))] < 0)
	{
	  ax *= -1; 
	  ay *= -1; 
	  az *= -1;
	}
      if(st < tnew)
	{
	  double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

	  if (sqrp > 0.0)
	    {
	      sqrp = az*dzz + sqrt(sqrp);

	      if(sqrp > 0.0) 
		{
		  tnew = fmin(tnew, st+dzz * dzz/sqrp);
		}
	    }
	}
#ifdef UNROLLED
      if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	}
#else
      if(tnew > fmin(st, xt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	}
      if(tnew > fmin(st, xt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xt)) 
	    { 
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txy)) 
	    {
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, yt))
	    {
	      FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	    }
	}
      if(tnew > fmin(st, xnt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, tym))
	    {
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, xnt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xnt))
	    {
	      FoldEdge2Points(&tnew,st, xnt, F, -ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txnyn))
	    {
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy) / dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, ynt)) 
	    {
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	    }
	  if(tnew > fmin(st, txm)) 
	    {
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
#endif
      T[i + HEIGHT * (j + WIDTH * (k+1))] = (1.0*sign(T[i + HEIGHT * (j + WIDTH * (k+1))]))*tnew;
      a_1_2[i-1] = T[i + HEIGHT * (j + WIDTH * (k+1))];
    }
}



void function_z_2_up(double *T, double *a_2_1, int k,   
		     double dzz, double dxx, double dyy, 
		     double dxy, double dxz, double dyz, 
		     double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
  
{ 
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  
  unsigned int i;// = (blockDim.x * blockIdx.x + threadIdx.x)+1;
  unsigned int j;// = 1;
 // if( j == 1 && ( i >= 1 && i < (HEIGHT-1)))
 //   {
 acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T, a_2_1) async(stream) //async(stream)  num_gangs(_nx) num_workers(64)
#pragma acc loop private (i,j, ax, ay, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
//  for(j = 2; j < WIDTH-2; ++j)
//    {
      for(i = 1; i < HEIGHT-1; ++i)
        {
        j=1;
      tnew = fabs((T[i + HEIGHT * (j + WIDTH * (k+1))]));
      st =  fabs((T[i + HEIGHT * (j + WIDTH * k)]));
      xt  = fabs((T[i + HEIGHT * ((j+1) + WIDTH * k)]));
      yt  = fabs((T[(i+1) + HEIGHT * (j + WIDTH * k)]));
      xnt = fabs((T[i + HEIGHT * ((j-1) + WIDTH * k)]));
      ynt = fabs((T[(i-1) + HEIGHT * (j + WIDTH * k)]));
      txm =  fabs((T[(i-1) + HEIGHT * ((j+1) + WIDTH * k)]));
      txy =  fabs((T[(i+1) + HEIGHT * ((j+1) + WIDTH * k)]));
      tym =  fabs((T[(i+1) + HEIGHT * ((j-1) +WIDTH * k)]));
      txnyn =fabs((T[(i-1) + HEIGHT * ((j-1) + WIDTH * k)]));

      	  ay = _axx; ax = _ayy; az = _azz;

      if (T[i + HEIGHT * (j + WIDTH * (k+1))] < 0)
	{
	  ax *= -1; 
	  ay *= -1; 
	  az *= -1;
	}
      if(st < tnew)
	{
	  double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

	  if (sqrp > 0.0)
	    {
	      sqrp = az*dzz + sqrt(sqrp);

	      if(sqrp > 0.0) 
		{
		  tnew = fmin(tnew, st+dzz * dzz/sqrp);
		}
	    }
	}
#ifdef UNROLLED
      if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	}
#else
      if(tnew > fmin(st, xt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	}
      if(tnew > fmin(st, xt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xt)) 
	    { 
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txy)) 
	    {
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy)/dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, yt))
	    {
	      FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	    }
	}
      if(tnew > fmin(st, xnt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, tym))
	    {
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, xnt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xnt))
	    {
	      FoldEdge2Points(&tnew,st, xnt, F, -ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txnyn))
	    {
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy) / dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, ynt)) 
	    {
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	    }
	  if(tnew > fmin(st, txm)) 
	    {
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
#endif
      T[i + HEIGHT * (j + WIDTH * (k+1))] = (1.0*sign(T[i + HEIGHT * (j + WIDTH * (k+1))]))*tnew;
      a_2_1[i-1] = T[i + HEIGHT * (j + WIDTH * (k+1))];
    }
}



// sweeps from top to down in Z- direction

void function_z_down(double *T, int k,   
		     double dzz, double dxx, double dyy, 
		     double dxy, double dxz, double dyz, 
		     double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  
  unsigned int j;// = (blockDim.y * blockIdx.y + threadIdx.y)+1+1;
  unsigned int i;// = (blockDim.x * blockIdx.x + threadIdx.x)+1;

 // if ((j >= 2 && j < WIDTH-2) && (i >= 1 && i < HEIGHT-1))
 //   { 
  acc_set_device_num(ID, acc_device_nvidia);
 #pragma acc parallel deviceptr(T) async(stream) //async(stream)  num_gangs(_nx) num_workers(64)
#pragma acc loop collapse(2) private (i,j, ax, ay, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
  for(j = 2; j < WIDTH-2; ++j)
    {
      for(i = 1; i < HEIGHT-1; ++i)
        {
        
      tnew = fabs((T[i + HEIGHT * (j + WIDTH * (k-1))]));
      st =  fabs((T[i + HEIGHT * (j + WIDTH * k)]));
      xt  = fabs((T[i + HEIGHT * ((j-1) + WIDTH * k)]));
      yt  = fabs((T[(i-1) + HEIGHT * (j + WIDTH * k)]));
      xnt = fabs((T[i + HEIGHT * ((j+1) + WIDTH * k)]));
      ynt = fabs((T[(i+1) + HEIGHT * (j + WIDTH * k)]));
      txm =  fabs((T[(i+1) + HEIGHT * ((j-1) + WIDTH * k)]));
      txy =  fabs((T[(i-1) + HEIGHT * ((j-1) + WIDTH * k)]));
      tym =  fabs((T[(i-1) + HEIGHT * ((j+1) + WIDTH * k)]));
      txnyn = fabs((T[(i+1) + HEIGHT * ((j+1) + WIDTH * k)]));

      	  ay = -_axx; ax = -_ayy; az = -_azz;	  

      if (T[i + HEIGHT * (j + WIDTH * (k-1))] < 0)
	{
	  ax *= -1; 
	  ay *= -1; 
	  az *= -1;
	}
      if(st < tnew)
	{
	  double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

	  if (sqrp > 0.0)
	    {
	      sqrp = az*dzz + sqrt(sqrp);

	      if(sqrp > 0.0) 
		{
		  tnew = fmin(tnew, st + dzz*dzz/sqrp);
		}
	    }
	}
#ifdef UNROLLED
      if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
      
	  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy) / dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew,st, yt, F, ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy) / dxy, 
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy) / dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy,
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	}
#else
      if(tnew > fmin(st, xt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	}
      if(tnew > fmin(st, xt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xt))
	    { 
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st,yt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txy)) 
	    {
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy) / dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, yt))
	    {
	      FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	    }
	}
      if(tnew > fmin(st,xnt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, tym)) 
	    {
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy) / dxy,
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, xnt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xnt))
	    {
	      FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txnyn)) 
	    {
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy,
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, ynt)) 
	    {
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	    }
	  if(tnew > fmin(st, txm)) 
	    {
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy) / dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
  
#endif
      T[i + HEIGHT * (j + WIDTH * (k-1))] = (1.0 * sign(T[i + HEIGHT * (j + WIDTH * (k-1))])) * tnew;
    } // IF
}
}


void function_z_t_down(double *T, int k,   
		       double dzz, double dxx, double dyy, 
		       double dxy, double dxz, double dyz, 
		       double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  
  unsigned int j;// = (blockDim.y * blockIdx.y + threadIdx.y)+1;
  unsigned int i;// = (blockDim.x * blockIdx.x + threadIdx.x)+1;
  

  //if ((j >= 1 && j < WIDTH-2) && (i >= 1 && i < HEIGHT-1))
  //  { 
   acc_set_device_num(ID, acc_device_nvidia);
 #pragma acc parallel deviceptr(T) async(stream) //async(stream)  num_gangs(_nx) num_workers(64)
#pragma acc loop collapse(2) private (i,j, ax, ay, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
  for(j = 1; j < WIDTH-2; ++j)
   {
      for(i = 1; i < HEIGHT-1; ++i)
        {

      tnew = fabs((T[i + HEIGHT * (j + WIDTH * (k-1))]));
      st =  fabs((T[i + HEIGHT * (j + WIDTH * k)]));
      xt  = fabs((T[i + HEIGHT * ((j-1) + WIDTH * k)]));
      yt  = fabs((T[(i-1) + HEIGHT * (j + WIDTH * k)]));
      xnt = fabs((T[i + HEIGHT * ((j+1) + WIDTH * k)]));
      ynt = fabs((T[(i+1) + HEIGHT * (j + WIDTH * k)]));
      txm =  fabs((T[(i+1) + HEIGHT * ((j-1) + WIDTH * k)]));
      txy =  fabs((T[(i-1) + HEIGHT * ((j-1) + WIDTH * k)]));
      tym =  fabs((T[(i-1) + HEIGHT * ((j+1) + WIDTH * k)]));
      txnyn = fabs((T[(i+1) + HEIGHT * ((j+1) + WIDTH * k)]));
	  ay = -_axx; ax = -_ayy; az = -_azz;	  
  
      if (T[i + HEIGHT * (j + WIDTH * (k-1))] < 0)
	{
	  ax *= -1; 
	  ay *= -1; 
	  az *= -1;
	}
      if(st < tnew)
	{
	  double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

	  if (sqrp > 0.0)
	    {
	      sqrp = az*dzz + sqrt(sqrp);

	      if(sqrp > 0.0) 
		{
		  tnew = fmin(tnew, st + dzz*dzz/sqrp);
		}
	    }
	}
#ifdef UNROLLED
      if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
      
	  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy) / dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew,st, yt, F, ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy) / dxy, 
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy) / dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy,
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	}
#else
      if(tnew > fmin(st, xt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	}
      if(tnew > fmin(st, xt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xt))
	    { 
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st,yt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txy)) 
	    {
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy) / dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, yt))
	    {
	      FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	    }
	}
      if(tnew > fmin(st,xnt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, tym)) 
	    {
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy) / dxy,
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, xnt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xnt))
	    {
	      FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txnyn)) 
	    {
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy,
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, ynt)) 
	    {
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	    }
	  if(tnew > fmin(st, txm)) 
	    {
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy) / dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
  
#endif
      T[i + HEIGHT * (j + WIDTH * (k-1))] = (1.0 * sign(T[i + HEIGHT * (j + WIDTH * (k-1))])) * tnew;
    } // IF
}
}



void function_z_b_down(double *T, int k,   
		       double dzz, double dxx, double dyy, 
		       double dxy, double dxz, double dyz, 
		       double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  
  unsigned int j;// = (blockDim.y * blockIdx.y + threadIdx.y)+1+1;
  unsigned int i;// = (blockDim.x * blockIdx.x + threadIdx.x)+1;

  //if ((j >= 2 && j < WIDTH-1) && (i >= 1 && i < HEIGHT-1))
  //  { 
   acc_set_device_num(ID, acc_device_nvidia);
 #pragma acc parallel deviceptr(T) async(stream) //async(stream)  num_gangs(_nx) num_workers(64)
#pragma acc loop collapse(2) private (i,j, ax, ay, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
 for(j = 2; j < WIDTH-1; ++j)
   {
      for(i = 1; i < HEIGHT-1; ++i)
        {
      tnew = fabs((T[i + HEIGHT * (j + WIDTH * (k-1))]));
      st =  fabs((T[i + HEIGHT * (j + WIDTH * k)]));
      xt  = fabs((T[i + HEIGHT * ((j-1) + WIDTH * k)]));
      yt  = fabs((T[(i-1) + HEIGHT * (j + WIDTH * k)]));
      xnt = fabs((T[i + HEIGHT * ((j+1) + WIDTH * k)]));
      ynt = fabs((T[(i+1) + HEIGHT * (j + WIDTH * k)]));
      txm =  fabs((T[(i+1) + HEIGHT * ((j-1) + WIDTH * k)]));
      txy =  fabs((T[(i-1) + HEIGHT * ((j-1) + WIDTH * k)]));
      tym =  fabs((T[(i-1) + HEIGHT * ((j+1) + WIDTH * k)]));
      txnyn = fabs((T[(i+1) + HEIGHT * ((j+1) + WIDTH * k)]));
	  ay = -_axx; ax = -_ayy; az = -_azz;	  
  
      if (T[i + HEIGHT * (j + WIDTH * (k-1))] < 0)
	{
	  ax *= -1; 
	  ay *= -1; 
	  az *= -1;
	}
      if(st < tnew)
	{
	  double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

	  if (sqrp > 0.0)
	    {
	      sqrp = az*dzz + sqrt(sqrp);

	      if(sqrp > 0.0) 
		{
		  tnew = fmin(tnew, st + dzz*dzz/sqrp);
		}
	    }
	}
#ifdef UNROLLED
      if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
      
	  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy) / dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew,st, yt, F, ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy) / dxy, 
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy) / dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy,
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	}
#else
      if(tnew > fmin(st, xt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	}
      if(tnew > fmin(st, xt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xt))
	    { 
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st,yt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txy)) 
	    {
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy) / dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, yt))
	    {
	      FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	    }
	}
      if(tnew > fmin(st,xnt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, tym)) 
	    {
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy) / dxy,
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, xnt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xnt))
	    {
	      FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txnyn)) 
	    {
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy,
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, ynt)) 
	    {
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	    }
	  if(tnew > fmin(st, txm)) 
	    {
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy) / dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
  
#endif
      T[i + HEIGHT * (j + WIDTH * (k-1))] = (1.0 * sign(T[i + HEIGHT * (j + WIDTH * (k-1))])) * tnew;
    } //IF
}
}



void function_z_1_down(double *T, double *a_1_2, int k,   
		       double dzz, double dxx, double dyy, 
		       double dxy, double dxz, double dyz, 
		       double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  
  unsigned int j;// = WIDTH-2;
  unsigned int i;// = (blockDim.x * blockIdx.x + threadIdx.x)+1;

  //if( j == (WIDTH-2) && ( i >= 1 && i < (HEIGHT-1)))
   // {
    acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T, a_1_2) async(stream) //async(stream)  num_gangs(_nx) num_workers(64)
#pragma acc loop private (i,j, ax, ay, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
//  for(j = 2; j < WIDTH-2; ++j)
//    {
      for(i = 1; i < HEIGHT-1; ++i)
        {
  j=WIDTH-2;
      tnew = fabs((T[i + HEIGHT * (j + WIDTH * (k-1))]));
      st =  fabs((T[i + HEIGHT * (j + WIDTH * k)]));
      xt  = fabs((T[i + HEIGHT * ((j-1) + WIDTH * k)]));
      yt  = fabs((T[(i-1) + HEIGHT * (j + WIDTH * k)]));
      xnt = fabs((T[i + HEIGHT * ((j+1) + WIDTH * k)]));
      ynt = fabs((T[(i+1) + HEIGHT * (j + WIDTH * k)]));
      txm =  fabs((T[(i+1) + HEIGHT * ((j-1) + WIDTH * k)]));
      txy =  fabs((T[(i-1) + HEIGHT * ((j-1) + WIDTH * k)]));
      tym =  fabs((T[(i-1) + HEIGHT * ((j+1) + WIDTH * k)]));
      txnyn = fabs((T[(i+1) + HEIGHT * ((j+1) + WIDTH * k)]));
	  ay = -_axx; ax = -_ayy; az = -_azz;	  
  
      if (T[i + HEIGHT * (j + WIDTH * (k-1))] < 0)
	{
	  ax *= -1; 
	  ay *= -1; 
	  az *= -1;
	}
      if(st < tnew)
	{
	  double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

	  if (sqrp > 0.0)
	    {
	      sqrp = az*dzz + sqrt(sqrp);

	      if(sqrp > 0.0) 
		{
		  tnew = fmin(tnew, st + dzz*dzz/sqrp);
		}
	    }
	}
#ifdef UNROLLED
      if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
      
	  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy) / dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew,st, yt, F, ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy) / dxy, 
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy) / dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy,
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	}
#else
      if(tnew > fmin(st, xt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	}
      if(tnew > fmin(st, xt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xt))
	    { 
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st,yt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txy)) 
	    {
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy) / dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, yt))
	    {
	      FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	    }
	}
      if(tnew > fmin(st,xnt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, tym)) 
	    {
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy) / dxy,
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, xnt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xnt))
	    {
	      FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txnyn)) 
	    {
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy,
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, ynt)) 
	    {
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	    }
	  if(tnew > fmin(st, txm)) 
	    {
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy) / dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
  
#endif
      T[i + HEIGHT * (j + WIDTH * (k-1))] = (1.0 * sign(T[i + HEIGHT * (j + WIDTH * (k-1))])) * tnew;
      a_1_2[i-1] = T[i + HEIGHT * (j + WIDTH * (k-1))];
    }
}



void function_z_2_down(double *T, double *a_2_1, int k,   
		       double dzz, double dxx, double dyy, 
		       double dxy, double dxz, double dyz, 
		       double F, double ay, double ax, double az, int WIDTH, int ID, int stream)
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  
  unsigned int j;// = 1;
  unsigned int i;// = (blockDim.x * blockIdx.x + threadIdx.x)+1;
 // if( j == 1 && ( i >= 1 && i < (HEIGHT-1)))
 //   {
   acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T, a_2_1) async(stream) //async(stream)  num_gangs(_nx) num_workers(64)
#pragma acc loop private (i,j, ax, ay, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
//  for(j = 2; j < WIDTH-2; ++j)
//    {
      for(i = 1; i < HEIGHT-1; ++i)
        {
        j=1;
      tnew = fabs((T[i + HEIGHT * (j + WIDTH * (k-1))]));
      st =  fabs((T[i + HEIGHT * (j + WIDTH * k)]));
      xt  = fabs((T[i + HEIGHT * ((j-1) + WIDTH * k)]));
      yt  = fabs((T[(i-1) + HEIGHT * (j + WIDTH * k)]));
      xnt = fabs((T[i + HEIGHT * ((j+1) + WIDTH * k)]));
      ynt = fabs((T[(i+1) + HEIGHT * (j + WIDTH * k)]));
      txm =  fabs((T[(i+1) + HEIGHT * ((j-1) + WIDTH * k)]));
      txy =  fabs((T[(i-1) + HEIGHT * ((j-1) + WIDTH * k)]));
      tym =  fabs((T[(i-1) + HEIGHT * ((j+1) + WIDTH * k)]));
      txnyn = fabs((T[(i+1) + HEIGHT * ((j+1) + WIDTH * k)]));

      	  ay = -_axx; ax = -_ayy; az = -_azz;	  

      if (T[i + HEIGHT * (j + WIDTH * (k-1))] < 0)
	{
	  ax *= -1; 
	  ay *= -1; 
	  az *= -1;
	}
      if(st < tnew)
	{
	  double sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

	  if (sqrp > 0.0)
	    {
	      sqrp = az*dzz + sqrt(sqrp);

	      if(sqrp > 0.0) 
		{
		  tnew = fmin(tnew, st + dzz*dzz/sqrp);
		}
	    }
	}
#ifdef UNROLLED
      if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy, txm) && tnew < fmin(txnyn, tym))) 
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
      
	  FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy) / dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew,st, yt, F, ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy) / dxy, 
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy) / dxy, 
			  (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy,
			  (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	}
#else
      if(tnew > fmin(st, xt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, xt,  txy, F,  ax,  ay, az, dxx, dyy, dzz, dxz);
	}
      if(tnew > fmin(st, xt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, xt,  txm, F,  ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xt))
	    { 
	      FoldEdge2Points(&tnew, st, xt, F,  ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st,yt) || tnew > txy)
	{
	  FoldSurf3Points(&tnew, st, yt,  txy, F,  ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txy)) 
	    {
	      FoldEdge2Points(&tnew, st, txy, F, (ax*dxx + ay*dyy) / dxy, 
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, yt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, yt,  tym, F,  ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, yt))
	    {
	      FoldEdge2Points(&tnew, st, yt, F, ay, ax, az, dyy, dzz);
	    }
	}
      if(tnew > fmin(st,xnt) || tnew > tym)
	{
	  FoldSurf3Points(&tnew, st, xnt, tym, F, -ax,  ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, tym)) 
	    {
	      FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy) / dxy,
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, xnt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	  if(tnew > fmin(st, xnt))
	    {
	      FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txnyn)
	{
	  FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, txnyn)) 
	    {
	      FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy,
			      (-ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
      if(tnew > fmin(st, ynt) || tnew > txm)
	{
	  FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	  if(tnew > fmin(st, ynt)) 
	    {
	      FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
	    }
	  if(tnew > fmin(st, txm)) 
	    {
	      FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy) / dxy, 
			      (ax*dyy + ay*dxx) / dxy, az, dxy, dzz);
	    }
	}
  
#endif
      T[i + HEIGHT * (j + WIDTH * (k-1))] = (1.0 * sign(T[i + HEIGHT * (j + WIDTH * (k-1))])) * tnew;
      a_2_1[i-1] = T[i + HEIGHT * (j + WIDTH * (k-1))];
    }
}


//*****************************************//
// boundary layer data transfer            //
// between GPU starts from here            //
//*****************************************//


void x_copy_from_1(int dk, double *a_1_2, double *c_T1, 
		   int i, int block, int width, int bot, int ID, int stream)
{
  int k; //= blockIdx.x * blockDim.x + threadIdx.x;
 // if( k >= 0 && k < block-2)
   acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_1_2, c_T1) async(stream)
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)
    {
      a_1_2[k] = c_T1[(i+dk+1) + block * (bot + width * (k+1))] ;  
    }
}


void x_copy_from_2(int dk, double *a_2_1, double *a_2_2, double *c_T2, 
		   int i, int block, int width, int top, int bot, int ID, int stream)
{  
  int k;// = blockIdx.x * blockDim.x + threadIdx.x;
 // if( k >= 0 && k < block-2)
   acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_2_1, a_2_2, c_T2) async(stream)
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)
    {
      a_2_1[k] = c_T2[(i+dk+1) + block * (top + width * (k+1))];
      a_2_2[k] = c_T2[(i+dk+1) + block * (bot + width * (k+1))];
    }
}


void x_copy_from_3(int dk, double *a_3_1,double *a_3_2, double *c_T3, 
		   int i, int block, int width, int top, int bot, int ID, int stream)
{
  int k; //= blockIdx.x * blockDim.x + threadIdx.x;
 // if( k >= 0 && k < block-2)
   acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_3_1, a_3_2, c_T3) async(stream)
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)
    {
      a_3_1[k] = c_T3[(i+dk+1) + block * (top + width * (k+1))]; 
      a_3_2[k] = c_T3[(i+dk+1) + block * (bot + width * (k+1))]; 
    }
}


void x_copy_from_4(int dk, double *a_4_1, double *c_T4, 
		   int i, int block, int width,int top, int ID, int stream)
{ 
  int k; //= blockIdx.x * blockDim.x + threadIdx.x;
 // if( k >= 0 && k < block-2)
   acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_4_1, c_T4) async(stream)
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)
    {
      a_4_1[k] = c_T4[(i+dk+1) + block * (top + width * (k+1))]; 
    }
}


void x_copy_to_1(int dk, double *a_2_1, double *c_T1, 
		 int i, int block, int width, int bot, int ID, int stream)
{  
  int k; //= blockIdx.x * blockDim.x + threadIdx.x;
 // if( k >= 0 && k < block-2)
   acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_2_1, c_T1) async(stream)
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)
    {
      c_T1[(i+dk+1) + block * (bot + width * (k+1))]  =  a_2_1[k];  
    }
}


void x_copy_to_2(int dk, double *a_1_2, double *c_T2, 
		 int i, int block, int width, int bot, int ID, int stream)
{
  int k; //= blockIdx.x * blockDim.x + threadIdx.x;
 // if( k >= 0 && k < block-2)
   acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_1_2, c_T2) async(stream)
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)
    {
      c_T2[(i+dk+1) + block * (0 + width * (k+1))]  = a_1_2[k];
    }
}


void x_copy_to_2_(int dk, double *a_3_1, double *c_T2, 
		  int i, int block, int width, int bot, int ID, int stream)
{
  int k; //= blockIdx.x * blockDim.x + threadIdx.x;
 // if( k >= 0 && k < block-2)
   acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_3_1, c_T2) async(stream)
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)
    {
      c_T2[(i+dk+1) + block * (bot + width * (k+1))]  = a_3_1[k];
    }
}



void x_copy_to_3(int dk, double *a_2_2, double *c_T3, 
		 int i, int block, int width, int bot, int ID, int stream)
{
  int k; //= blockIdx.x * blockDim.x + threadIdx.x;
 // if( k >= 0 && k < block-2)
   acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_2_2, c_T3) async(stream)
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)
    {
      c_T3[(i+dk+1) + block * (0 + width * (k+1))]  =  a_2_2[k];
    }
}


void x_copy_to_3_(int dk, double *a_4_1, double *c_T3, 
		  int i, int block, int width, int bot, int ID, int stream)
{
  int k; //= blockIdx.x * blockDim.x + threadIdx.x;
 // if( k >= 0 && k < block-2)
   acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_4_1, c_T3) async(stream)
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)

    {
      c_T3[(i+dk+1) + block * (bot + width * (k+1))]  =  a_4_1[k];  
    }
}



void x_copy_to_4(int dk, double *a_3_2,double *c_T4, 
		 int i, int block, int width, int bot, int ID, int stream)
{
  int k; //= blockIdx.x * blockDim.x + threadIdx.x;
 // if( k >= 0 && k < block-2)
   acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_3_2, c_T4) async(stream)
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)

    {
      c_T4[(i+dk+1) + block * (0 + width * (k+1))]  =   a_3_2[k]; 
    }
}


void z_copy_from_1(int dk, double *a_1_2, double *c_T1, 
		   int k, int block, int width, int bot, int ID, int stream)
{  
  int i; //= blockIdx.x * blockDim.x + threadIdx.x;
  //if( i >= 0 && i < block-2)
    acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_1_2, c_T1) async(stream)
#pragma acc loop private (i)
  for(i = 0; i < block-2; ++i)

    {
      a_1_2[i] = c_T1[(i+1) + block * (bot + width * (k+dk+1))] ;  
    }
}


void z_copy_from_2(int dk, double *a_2_1, double *a_2_2, double *c_T2, 
		   int k, int block, int width, int top, int bot, int ID, int stream)
{
  int i; //= blockIdx.x * blockDim.x + threadIdx.x;
  //if( i >= 0 && i < block-2)
    acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_2_1, a_2_2, c_T2) async(stream)
#pragma acc loop private (i)
  for(i = 0; i < block-2; ++i)

    {
      a_2_1[i] = c_T2[(i+1) + block *(top + width * (k+dk+1))];
      a_2_2[i] = c_T2[(i+1) + block *(bot + width * (k+dk+1))];
    }
}


void z_copy_from_3(int dk, double *a_3_1,double *a_3_2, double *c_T3, 
		   int k, int block, int width, int top, int bot, int ID, int stream)
{
  int i; //= blockIdx.x * blockDim.x + threadIdx.x;
  //if( i >= 0 && i < block-2)
    acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_3_1, a_3_2, c_T3) async(stream)
#pragma acc loop private (i)
  for(i = 0; i < block-2; ++i)

    {
      a_3_1[i] = c_T3[(i+1) + block * (top + width * (k+dk+1))]; 
      a_3_2[i] = c_T3[(i+1) + block * (bot + width * (k+dk+1))];   
    }
}


void z_copy_from_4(int dk, double *a_4_1, double *c_T4, 
		   int k, int block, int width, int top, int ID, int stream)
{ 
  int i; //= blockIdx.x * blockDim.x + threadIdx.x;
  //if( i >= 0 && i < block-2)
    acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_4_1, c_T4) async(stream)
#pragma acc loop private (i)
  for(i = 0; i < block-2; ++i)

    {
      a_4_1[i] = c_T4[(i+1) + block * (top + width * (k+dk+1))];   
    }
}


void z_copy_to_1(int dk, double *a_2_1, double *c_T1, 
		 int k, int block, int width, int bot, int ID, int stream)
{  
  int i; //= blockIdx.x * blockDim.x + threadIdx.x;
  //if( i >= 0 && i < block-2)
      acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_2_1, c_T1) async(stream)
#pragma acc loop private (i)
  for(i = 0; i < block-2; ++i)
    {
      c_T1[(i+1) + block * (bot + width * (k+dk+1))]  =  a_2_1[i];  
    }
}


void z_copy_to_2(int dk, double *a_1_2, double *c_T2, 
		 int k, int block, int width, int bot, int ID, int stream)
{ 
  int i; //= blockIdx.x * blockDim.x + threadIdx.x;
  //if( i >= 0 && i < block-2)
        acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_1_2, c_T2) async(stream)
#pragma acc loop private (i)
  for(i = 0; i < block-2; ++i)
    {
      c_T2[(i+1) + block * (0 + width * (k+dk+1))]  = a_1_2[i];
    }
}


void z_copy_to_2_(int dk, double *a_3_1, double *c_T2, 
		  int k, int block, int width, int bot, int ID, int stream)
{ 
  int i; //= blockIdx.x * blockDim.x + threadIdx.x;
  //if( i >= 0 && i < block-2)
        acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_3_1, c_T2) async(stream)
#pragma acc loop private (i)
  for(i = 0; i < block-2; ++i)
    {
      c_T2[(i+1) + block * (bot + width * (k+dk+1))]  = a_3_1[i];  
    }
}




void z_copy_to_3(int dk, double *a_2_2, double *c_T3, 
		 int k, int block, int width, int bot, int ID, int stream)
{
  int i; //= blockIdx.x * blockDim.x + threadIdx.x;
  //if( i >= 0 && i < block-2)
        acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_2_2, c_T3) async(stream)
#pragma acc loop private (i)
  for(i = 0; i < block-2; ++i)
    {
      c_T3[(i+1)+ block * (0 + width * (k+dk+1))]  =  a_2_2[i];
    }
}


void z_copy_to_3_(int dk, double *a_4_1, double *c_T3, 
		  int k, int block, int width, int bot, int ID, int stream)
{
  int i; //= blockIdx.x * blockDim.x + threadIdx.x;
  //if( i >= 0 && i < block-2)
        acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_4_1, c_T3) async(stream)
#pragma acc loop private (i)
  for(i = 0; i < block-2; ++i)
    {
      c_T3[(i+1)+ block * (bot + width * (k+dk+1))]  =  a_4_1[i];
    }
}



void z_copy_to_4(int dk, double *a_3_2,double *c_T4, 
		 int k, int block, int width, int bot, int ID, int stream)
{
  int i; //= blockIdx.x * blockDim.x + threadIdx.x;
  //if( i >= 0 && i < block-2)
        acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_3_2, c_T4) async(stream)
#pragma acc loop private (i)
  for(i = 0; i < block-2; ++i)
    {
      c_T4[(i+1) + block * (0 + width * (k+dk+1))]  =   a_3_2[i];
    }
}



void y_copy_from_1(int dk, double *a_1_2, double *c_T1, 
		   int j, int block, int width, int bot, int ID, int stream)
{  
  int k; //= blockIdx.x * blockDim.x + threadIdx.x;
 // if( k >= 0 && k < block-2)
       acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_1_2, c_T1) async(stream)
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)
    {
      a_1_2[k] = c_T1[bot + width * ((j+dk+1) + block * (k+1))] ;  
    }
}



void y_copy_from_2(int dk, double *a_2_1, double *a_2_2, double *c_T2, 
		   int j, int block, int width, int top, int bot, int ID, int stream)
{
  
  int k; //= blockIdx.x * blockDim.x + threadIdx.x;
 // if( k >= 0 && k < block-2)
        acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_2_1, a_2_2, c_T2) async(stream)
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)
    {
      a_2_1[k] = c_T2[top + width * ((j+dk+1) + block * (k+1))];
      a_2_2[k] = c_T2[bot + width * ((j+dk+1) + block * (k+1))];
    }
}



void y_copy_from_3(int dk, double *a_3_1,double *a_3_2, double *c_T3, 
		   int j, int block, int width, int top, int bot, int ID, int stream)
{
  
  int k; //= blockIdx.x * blockDim.x + threadIdx.x;
 // if( k >= 0 && k < block-2)
        acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_3_1, a_3_2, c_T3) async(stream)
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)
    {
      a_3_1[k] = c_T3[top + width * ((j+dk+1) + block * (k+1))]; 
      a_3_2[k] = c_T3[bot + width * ((j+dk+1) + block * (k+1))]; 
    }
}



void y_copy_from_4(int dk, double *a_4_1, double *c_T4, 
		   int j, int block, int width, int top, int ID, int stream)
{  
  int k; //= blockIdx.x * blockDim.x + threadIdx.x;
 // if( k >= 0 && k < block-2)
        acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_4_1, c_T4) async(stream)
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)
    {
      a_4_1[k] = c_T4[top + width * ((j+dk+1) + block * (k+1))]; 
    }
}



void y_copy_to_1(int dk, double *a_2_1, double *c_T1, 
		 int j, int block, int width, int bot, int ID, int stream)
{
  int k; //= blockIdx.x * blockDim.x + threadIdx.x;
 // if( k >= 0 && k < block-2)
        acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_2_1, c_T1) async(stream)
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)
    {
      c_T1[bot + width * ((j+dk+1) + block * (k+1))]  =  a_2_1[k];  
    }
}



void y_copy_to_2(int dk, double *a_1_2, double *c_T2, 
		 int j, int block, int width, int bot, int ID, int stream)
{
  
  int k; //= blockIdx.x * blockDim.x + threadIdx.x;
 // if( k >= 0 && k < block-2)
        acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_1_2, c_T2) async(stream)
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)
    {
      c_T2[0 + width * ((j+dk+1) + block * (k+1))]  = a_1_2[k];
    }
}



void y_copy_to_2_(int dk, double *a_3_1,double *c_T2, 
		  int j, int block, int width, int bot, int ID, int stream)
{
  
  int k; //= blockIdx.x * blockDim.x + threadIdx.x;
 // if( k >= 0 && k < block-2)
        acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_3_1, c_T2) async(stream)
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)
    {
      c_T2[bot + width * ((j+dk+1) + block * (k+1))]  = a_3_1[k];
    }
}



void y_copy_to_3(int dk, double *a_2_2, double *c_T3, 
		 int j, int block, int width, int bot, int ID, int stream)
{
  int k; //= blockIdx.x * blockDim.x + threadIdx.x;
 // if( k >= 0 && k < block-2)
        acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_2_2, c_T3) async(stream)
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)
    {
      c_T3[ 0 + width *((j+dk+1) + block * (k+1))]  =  a_2_2[k];
    }
}



void y_copy_to_3_(int dk, double *a_4_1, double *c_T3, 
		  int j, int block, int width, int bot, int ID, int stream)
{
  int k; //= blockIdx.x * blockDim.x + threadIdx.x;
 // if( k >= 0 && k < block-2)
        acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_4_1, c_T3) async(stream)
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)
    {
      c_T3[bot + width * ((j+dk+1) + block * (k+1))]  =  a_4_1[k];
    }
}



void y_copy_to_4(int dk, double *a_3_2,double *c_T4, 
		 int j, int block, int width, int bot, int ID, int stream)
{  
  int k; //= blockIdx.x * blockDim.x + threadIdx.x;
 // if( k >= 0 && k < block-2)
          acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_3_2, c_T4) async(stream)
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)
    {
      c_T4[0 + width * ((j+dk+1) + block * (k+1))]  =   a_3_2[k];
    }
}



//************************************//
// copy the partined data to          //
// Y- direction for the 3-direction   //
// sweep; from X or Z - direction     //    
// partions                           //
//************************************//  
void from_k_to_y_direction_T1(double *T1,double *T1_1, double *T1_2, 
			      double *T1_3, double *T1_4,
			      int block, int width, int breadth, int ID)//, int stream)
{
 
  int i, j, k;
  
  //i = blockIdx.x * blockDim.x + threadIdx.x;
  //j = blockIdx.y * blockDim.y + threadIdx.y;
  //k = blockIdx.z * blockDim.z + threadIdx.z;
  //if( (i >= 0 && i < width) && ( j >= 0 && j < width ) && ( k >= 0 && k < block))
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T1, T1_1, T1_2, T1_3, T1_4) //async(stream) 
#pragma acc loop collapse(3) private(i, j, k)
  //#pragma omp target is_device_ptr(T1, T1_1, T1_2) device(ID) private(i, j, k)
  for(i = 0; i < width; ++i){
    for(j = 0; j < width; ++j){
      for(k = 0; k < block; ++k){  
	if(j < breadth)
	  { 
	    T1_1[i + width * (j + width * k)] = T1[i + block * (j + width * k)];
	    T1_2[i + width * (j + breadth * k)] = T1[i + breadth + block * (j + width * k)];
	    T1_3[i + width * (j + breadth * k)] = T1[i + (breadth*2) + block * (j + width * k)];
	    T1_4[i + width * (j + breadth * k)] = T1[i + (breadth*3) + block * (j + width * k)];	
	  }
	else
	  { 
	    T1_1[i + width * (j + width * k)] = T1[i + block * (j + width * k)];      
	  }
      }
    }
  }
}


void from_k_to_y_direction_T2(double *T2, double *T2_1, double *T2_2, 
			      double *T2_3, double *T2_4, 
			      int block, int width, int breadth, int ID)//, int stream)
{
  
  int i, j, k;
  
  //i = blockIdx.x * blockDim.x + threadIdx.x;
  //j = blockIdx.y * blockDim.y + threadIdx.y;
  //k = blockIdx.z * blockDim.z + threadIdx.z;
  //if( (i >= 0 && i < width) && ( j >= 0 && j < width ) && ( k >= 0 && k < block))
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T2, T2_1, T2_2, T2_3, T2_4) //async(stream) 
#pragma acc loop collapse(3) private(i, j, k)
  //#pragma omp target is_device_ptr(T1, T1_1, T1_2) device(ID) private(i, j, k)
  for(i = 0; i < width; ++i){
    for(j = 0; j < width; ++j){
      for(k = 0; k < block; ++k){  
	if(j < breadth)
	  {
	    
	    T2_1[i + width * (j + breadth * k)] = T2[i + block * (j + 2 + width * k)];
	    T2_2[i + width * (j + width * k)] = T2[breadth + i + block * (j + width * k)];
	    T2_3[i + width * (j + breadth * k)] = T2[(breadth*2) + i + block * (j + width * k)];
	    T2_4[i + width * (j + breadth * k)] = T2[(breadth*3) + i + block * (j + width * k)];
	    
	  }
	else
	  {
	    T2_2[i + width * (j+width * k)] = T2[breadth + i + block * (j+width * k)];
	  }
      }
    }
  }
}


void from_k_to_y_direction_T3(double *T3, double *T3_1, double *T3_2,
			      double *T3_3, double *T3_4,
			      int block, int width, int breadth, int ID)//, int stream)
{
  int i, j, k;
  
  //i = blockIdx.x * blockDim.x + threadIdx.x;
  //j = blockIdx.y * blockDim.y + threadIdx.y;
  //k = blockIdx.z * blockDim.z + threadIdx.z;
  //if( (i >= 0 && i < width) && ( j >= 0 && j < width ) && ( k >= 0 && k < block))
  //  {
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T3, T3_1, T3_2, T3_3, T3_4) //async(stream) 
#pragma acc loop collapse(3) private(i, j, k)
  //#pragma omp target is_device_ptr(T1, T1_1, T1_2) device(ID) private(i, j, k)
  for(i = 0; i < width; ++i){
    for(j = 0; j < width; ++j){
      for(k = 0; k < block; ++k){  
	if(j < breadth)
	  {
	    T3_1[i + width * (j + breadth * k)] = T3[i + block * (j + 2 + width * k)];
	    T3_2[i + width * (j + breadth * k)] = T3[breadth + i + block * (j + 2 + width * k)];
	    T3_3[i + width * (j + width * k)]= T3[(breadth*2) + i + block * (j + width * k)];
	    T3_4[i + width * (j + breadth * k)] = T3[(breadth*3) + i + block * (j + width * k)]; 
	  }
	else
	  {
	    T3_3[i + width * (j + width * k)] = T3[(breadth*2) + i + block * (j + width * k)];    
	  }
      }
    }
  }
}


void from_k_to_y_direction_T4(double *T4, double *T4_1, double *T4_2, 
			      double *T4_3, double *T4_4,
			      int block, int width, int breadth, int ID)//, int stream)
{
  
  int i, j, k;
  
  //i = blockIdx.x * blockDim.x + threadIdx.x;
  //j = blockIdx.y * blockDim.y + threadIdx.y;
  //k = blockIdx.z * blockDim.z + threadIdx.z;
  //if( (i >= 0 && i < width) && ( j >= 0 && j < width ) && ( k >= 0 && k < block))
  // {
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T4, T4_1, T4_2, T4_3, T4_4) //async(stream) 
#pragma acc loop collapse(3) private(i, j, k)
  //#pragma omp target is_device_ptr(T1, T1_1, T1_2) device(ID) private(i, j, k)
  for(i = 0; i < width; ++i){
    for(j = 0; j < width; ++j){
      for(k = 0; k < block; ++k){  
	if(j < breadth)
	  { 
	    T4_1[i + width * (j + breadth * k)] = T4[i + block * (2+ j + width * k)];
	    T4_2[i + width * (j + breadth * k)] = T4[breadth + i + block * (j + 2 + width * k)];
	    T4_3[i + width * (j + breadth * k)] = T4[(breadth*2) + i + block * (2 + j + width * k)];
	    T4_4[i + width * (j + width * k)] = T4[(breadth*3) + i + block * (j + width * k)];
	  }
	else
	  {
	    T4_4[i + width * (j + width * k)] = T4[(breadth*3) + i + block * (j + width * k)]; 
	  }
      }
    }
  }
}


void to_k_to_y_direction_T1(double *T1, double *T1_1, double *T2_1, 
			    double *T3_1, double *T4_1,
			    int block, int width, int breadth, int ID)//, int stream)
{
  int i, j, k;
  
  //i = blockIdx.x * blockDim.x + threadIdx.x;
  //j = blockIdx.y * blockDim.y + threadIdx.y;
  //k = blockIdx.z * blockDim.z + threadIdx.z;
  //if( (i >= 0 && i < width) && ( j >= 0 && j < width ) && ( k >= 0 && k < block))
  // {
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T1, T1_1, T2_1, T3_1, T4_1) //async(stream) 
#pragma acc loop collapse(3) private(i, j, k)
  //#pragma omp target is_device_ptr(T1, T1_1, T1_2) device(ID) private(i, j, k)
  for(i = 0; i < width; ++i){
    for(j = 0; j < width; ++j){
      for(k = 0; k < block; ++k){  
	if(j < breadth)
	  {
	    T1[i + width * (j + block * k)] =  T1_1[ i + width * (j+width * k)];
	    T1[i + width * (j + ((breadth+2)) + block* k)] =  T2_1[i + width * (j+breadth * k)];
	    T1[i + width * (j + ((breadth+breadth+2)) + block * k)] =  T3_1[i + width * (j+breadth * k)];
	    T1[i + width * (j + ((breadth+breadth+breadth+2)) + block * k)] =  T4_1[i + width * (j+breadth * k)];
	  }
	else
	  {
	    T1[i + width *(j + block * k)] =  T1_1[i + width * (j + width * k)];
	  }
	
      }
    }
  }
}


void to_k_to_y_direction_T2(double *T2,double *T1_2,double *T2_2, 
			    double *T3_2,double *T4_2,
			    int block, int width, int breadth, int ID)//, int stream)
{
  int i, j, k;
  
  //i = blockIdx.x * blockDim.x + threadIdx.x;
  //j = blockIdx.y * blockDim.y + threadIdx.y;
  //k = blockIdx.z * blockDim.z + threadIdx.z;
  //if( (i >= 0 && i < width) && ( j >= 0 && j < width ) && ( k >= 0 && k < block))
  //  { 
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T2, T1_2, T2_2, T3_2, T4_2) //async(stream) 
#pragma acc loop collapse(3) private(i, j, k)
  //#pragma omp target is_device_ptr(T1, T1_1, T1_2) device(ID) private(i, j, k)
  for(i = 0; i < width; ++i){
    for(j = 0; j < width; ++j){
      for(k = 0; k < block; ++k){  
	if(j < breadth)
	  {
	    T2[i + width * (j + block * k)] =  T1_2[i + width * (j + breadth * k)];
	    T2[i + width * (j + breadth+block * k)] =  T2_2[i + width * (j + width * k)];
	    T2[i + width * (j + (breadth+breadth+2) + block * k)] =  T3_2[i + width * (j + breadth * k)];
	    T2[i + width * (j + (breadth+breadth+breadth) + 2 + block * k)] =  T4_2[i + width * (j + breadth * k)]; 
	  }
	else
	  {
	    T2[i + width * (j + breadth + block * k)] =  T2_2[i + width * (j + width * k)];
	  }
      }
    }
  }
}


void to_k_to_y_direction_T3(double *T3, double *T1_3, double *T2_3, 
			    double *T3_3, double *T4_3,
			    int block, int width, int breadth, int ID)//, int stream)
{
  int i, j, k;
  
  //i = blockIdx.x * blockDim.x + threadIdx.x;
  //j = blockIdx.y * blockDim.y + threadIdx.y;
  //k = blockIdx.z * blockDim.z + threadIdx.z;
  //if( (i >= 0 && i < width) && ( j >= 0 && j < width ) && ( k >= 0 && k < block))
  //{
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T3, T1_3, T2_3, T3_3, T4_3) //async(stream) 
#pragma acc loop collapse(3) private(i, j, k)
  //#pragma omp target is_device_ptr(T1, T1_1, T1_2) device(ID) private(i, j, k)
  for(i = 0; i < width; ++i){
    for(j = 0; j < width; ++j){
      for(k = 0; k < block; ++k){  
	if(j < breadth)
	  {
	    T3[i + width * (j + block * k)] =  T1_3[i + width * (j+breadth * k)];
	    T3[i + width * (j + breadth+block * k)] =  T2_3[i + width * (j+breadth * k)];
	    T3[i + width * (j + (breadth+breadth) + block * k)] =  T3_3[i + width * (j + width * k)];
	    T3[i + width * (j + (breadth+breadth+breadth+2) + block * k)] =  T4_3[i + width * (j + breadth * k)];
	  }
	else
	  { 
	    T3[i + width * (j + (breadth+breadth) + block * k)] =  T3_3[i + width * (j + width * k)]; 
	  }
      }
    }
  }
}


void to_k_to_y_direction_T4(double *T4, double *T1_4, double *T2_4, 
			    double *T3_4, double *T4_4,
			    int block, int width, int breadth, int ID)//, int stream)
{
  int i, j, k;
  
  //i = blockIdx.x * blockDim.x + threadIdx.x;
  //j = blockIdx.y * blockDim.y + threadIdx.y;
  //k = blockIdx.z * blockDim.z + threadIdx.z;
  //if( (i >= 0 && i < width) && ( j >= 0 && j < width ) && ( k >= 0 && k < block))
  // {
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T4, T1_4, T2_4, T3_4, T4_4) //async(stream) 
#pragma acc loop collapse(3) private(i, j, k)
  //#pragma omp target is_device_ptr(T1, T1_1, T1_2) device(ID) private(i, j, k)
  for(i = 0; i < width; ++i){
    for(j = 0; j < width; ++j){
      for(k = 0; k < block; ++k){ 
	if(j < breadth)
	  {
	    T4[i + width * (j+ block * k)] =  T1_4[i + width * (j + breadth * k)];
	    T4[i + width * (j+ breadth+block * k)] =  T2_4[i + width * (j + breadth * k)];
	    T4[i + width * (j+ breadth+breadth+block * k)] =  T3_4[i + width * (j + breadth * k)];
	    T4[i + width * (j+ breadth+breadth+breadth+block * k)] =  T4_4[i + width * (j + width * k)];
	  }
	else
	  {
	    T4[i + width * (j + (breadth+breadth+breadth) + block * k)] =  T4_4[i + width * (j + width * k)]; 
	  }
      }
    }
  }
}


//*************************************//
// copy the partined data to           //
// x or Z direction for the next fressh//
// new sweep from Y - direction        //    
// partions                            //
//*************************************//  


void from_y_to_k_direction_T11(double *T1, double *T1_1, double *T1_2, 
			       double *T1_3, double *T1_4,
			       int block, int width, int breadth, int ID)//, int stream)
{
  int i, j, k;
  
  //i = blockIdx.x * blockDim.x + threadIdx.x;
  //j = blockIdx.y * blockDim.y + threadIdx.y;
  //k = blockIdx.z * blockDim.z + threadIdx.z;
  //if( (i >= 0 && i < width) && ( j >= 0 && j < width ) && ( k >= 0 && k < block))
  //  {
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T1, T1_1, T1_2, T1_3, T1_4) //async(stream) 
#pragma acc loop collapse(3) private(i, j, k)
  //#pragma omp target is_device_ptr(T1, T1_1, T1_2) device(ID) private(i, j, k)
  for(i = 0; i < width; ++i){
    for(j = 0; j < width; ++j){
      for(k = 0; k < block; ++k){  
	if(i < breadth)
	  {
	    T1_1[i + width * (j + width * k)] = T1[i + width * (j + block * k)];
	    T1_2[i + breadth * (j + width * k)] = T1[i + width * (breadth + j + block * k)];
	    T1_3[i + breadth * (j + width * k)] = T1[i + width * ((breadth*2) + j + block * k)];
	    T1_4[i + breadth * (j + width * k)] = T1[i + width * ((breadth*3) + j + block * k)];       
	  }
	else
	  {
	    T1_1[i + width * (j + width * k)] = T1[i + width * (j + block * k)]; 
	  }
      }
    }
  }
}


void from_y_to_k_direction_T21(double *T2, double *T2_1, double *T2_2, 
			       double  *T2_3, double *T2_4, 
			       int block, int width, int breadth, int ID)//, int stream)
{
  int i, j, k;
  
  //i = blockIdx.x * blockDim.x + threadIdx.x;
  //j = blockIdx.y * blockDim.y + threadIdx.y;
  //k = blockIdx.z * blockDim.z + threadIdx.z;
  //if( (i >= 0 && i < width) && ( j >= 0 && j < width ) && ( k >= 0 && k < block))
  //  {
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T2, T2_1, T2_2, T2_3, T2_4) //async(stream) 
#pragma acc loop collapse(3) private(i, j, k)
  //#pragma omp target is_device_ptr(T1, T1_1, T1_2) device(ID) private(i, j, k)
  for(i = 0; i < width; ++i){
    for(j = 0; j < width; ++j){
      for(k = 0; k < block; ++k){  
	if(i < breadth)
	  {
	    T2_1[i + breadth * (j + width * k)] = T2[i + 2 + width * (j + block * k)];
	    T2_2[i + width * (j + width * k)] = T2[i + width * (breadth + j + block * k)];
	    T2_3[i + breadth * (j + width * k)] = T2[i + width * ((breadth*2) + j + block * k)];
	    T2_4[i + breadth * (j + width * k)] = T2[i + width * ((breadth*3) + j + block * k)];	      
	  }
	else
	  {
	    T2_2[i + width * (j + width * k)] = T2[i + width * (breadth + j + block * k)];
	  }
      }
    }
  }
}


void from_y_to_k_direction_T31(double *T3, double  *T3_1, double  *T3_2,
			       double  *T3_3, double *T3_4,
			       int block, int width, int breadth, int ID)//, int stream)
{
  int i, j, k;
  
  //i = blockIdx.x * blockDim.x + threadIdx.x;
  //j = blockIdx.y * blockDim.y + threadIdx.y;
  //k = blockIdx.z * blockDim.z + threadIdx.z;
  //if( (i >= 0 && i < width) && ( j >= 0 && j < width ) && ( k >= 0 && k < block))
  //  { 
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T3, T3_1, T3_2, T3_3, T3_4) //async(stream) 
#pragma acc loop collapse(3) private(i, j, k)
  //#pragma omp target is_device_ptr(T1, T1_1, T1_2) device(ID) private(i, j, k)
  for(i = 0; i < width; ++i){
    for(j = 0; j < width; ++j){
      for(k = 0; k < block; ++k){  
	if(i < breadth)
	  {
	    T3_1[i + breadth * (j + width * k)] = T3[i + 2 + width * (j + block * k)];
	    T3_2[i + breadth * (j + width * k)] = T3[i + 2 + width * (breadth + j + block * k)];
	    T3_3[i + width * (j + width * k)] = T3[i + width * ((breadth*2) + j + block * k)];
	    T3_4[i + breadth * (j + width * k)] = T3[i + width * ((breadth*3) + j + block * k)]; 
	  }
	else
	  {
	    T3_3[i + width * (j + width * k)] = T3[i + width * ((breadth*2) + j + block * k)]; 
	  }
      }
    }
  }
}


void from_y_to_k_direction_T41(double *T4, double  *T4_1, double *T4_2, 
			       double  *T4_3, double *T4_4,
			       int block, int width, int breadth, int ID)//, int stream)
{
  
  int i, j, k;
  
  //i = blockIdx.x * blockDim.x + threadIdx.x;
  //j = blockIdx.y * blockDim.y + threadIdx.y;
  //k = blockIdx.z * blockDim.z + threadIdx.z;
  //if( (i >= 0 && i < width) && ( j >= 0 && j < width ) && ( k >= 0 && k < block))
  //  {
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T4, T4_1, T4_2, T4_3, T4_4) //async(stream) 
#pragma acc loop collapse(3) private(i, j, k)
  //#pragma omp target is_device_ptr(T1, T1_1, T1_2) device(ID) private(i, j, k)
  for(i = 0; i < width; ++i){
    for(j = 0; j < width; ++j){
      for(k = 0; k < block; ++k){  
	if(i < breadth)
	  {
	    T4_1[i + breadth * (j + width * k)] = T4[i + 2 + width * (j + block * k)];
	    T4_2[i + breadth * (j + width * k)] = T4[i + 2 + width * (breadth + j + block * k)];
	    T4_3[i + breadth * (j + width * k)] = T4[i + 2 + width * ((breadth*2) + j + block * k)];
	    T4_4[i + width * (j + width * k)] = T4[i + width * ((breadth*3) + j + block * k)]; 
	  }
	else
	  {
	    T4_4[i + width * (j + width * k)] = T4[i + width * ((breadth*3) + j + block * k)];
	  }
      }
    }
  }
}


void to_y_to_k_direction_T11(double *T1, double *T1_1, double *T2_1, 
			     double *T3_1, double *T4_1,
			     int block, int width, int breadth, int ID)//, int stream)
{
  int i, j, k;
  
  //i = blockIdx.x * blockDim.x + threadIdx.x;
  //j = blockIdx.y * blockDim.y + threadIdx.y;
  //k = blockIdx.z * blockDim.z + threadIdx.z;
  //if( (i >= 0 && i < width) && ( j >= 0 && j < width ) && ( k >= 0 && k < block))
  //  {
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T1, T1_1, T2_1, T3_1, T4_1) //async(stream) 
#pragma acc loop collapse(3) private(i, j, k)
  //#pragma omp target is_device_ptr(T1, T1_1, T1_2) device(ID) private(i, j, k)
  for(i = 0; i < width; ++i){
    for(j = 0; j < width; ++j){
      for(k = 0; k < block; ++k){  
	if(i < breadth)
	  {
	    T1[i + block * (j + width * k)] =  T1_1[i + width * (j + width * k)]; 
	    T1[breadth + 2 + i + block *  (j + width * k)] = T2_1[i + breadth * (j + width * k)];
	    T1[(breadth*2) + 2 + i + block * (j + width * k)] = T3_1[i + breadth * (j + width * k)]; 
	    T1[(breadth*3) + 2 + i + block * (j + width * k)] = T4_1[i + breadth * (j + width * k)];  
	  }
	else
	  { 
	    T1[i + block * (j + width * k)] = T1_1[i + width * (j + width * k)];
	  }
      }
    }
  }
}



void to_y_to_k_direction_T21(double *T2, double *T1_2, double *T2_2, 
			     double *T3_2,double *T4_2,
			     int block, int width, int breadth, int ID)//, int stream)
{
  int i, j, k;
  
  //i = blockIdx.x * blockDim.x + threadIdx.x;
  //j = blockIdx.y * blockDim.y + threadIdx.y;
  //k = blockIdx.z * blockDim.z + threadIdx.z;
  //if( (i >= 0 && i < width) && ( j >= 0 && j < width ) && ( k >= 0 && k < block))
  //  {
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T2, T1_2, T2_2, T3_2, T4_2) //async(stream) 
#pragma acc loop collapse(3) private(i, j, k)
  //#pragma omp target is_device_ptr(T1, T1_1, T1_2) device(ID) private(i, j, k)
  for(i = 0; i < width; ++i){
    for(j = 0; j < width; ++j){
      for(k = 0; k < block; ++k){  
	if(i < breadth)
	  {
	    T2[i + block * (j + width * k)] =  T1_2[i + breadth * (j + width * k)]; 
	    T2[breadth + i + block * (j + width * k)] = T2_2[i + width * (j + width * k)];
	    T2[(breadth*2) + 2 + i +block * (j + width * k)] = T3_2[i + breadth * (j + width * k)]; 
	    T2[(breadth*3) + 2 + i +block * (j + width * k)] = T4_2[i + breadth * (j + width * k)];  
	  }
  
	else
	  {
	    T2[breadth + i + block * (j + width * k)] = T2_2[i + width * (j + width * k)];
	  }
      } 
    }
  }
}


void to_y_to_k_direction_T31(double *T3, double *T1_3, double *T2_3, 
			     double *T3_3, double *T4_3,
			     int block, int width, int breadth, int ID)//, int stream)
{  
  int i, j, k;

  //i = blockIdx.x * blockDim.x + threadIdx.x;
  //j = blockIdx.y * blockDim.y + threadIdx.y;
  //k = blockIdx.z * blockDim.z + threadIdx.z;
  //if( (i >= 0 && i < width) && ( j >= 0 && j < width ) && ( k >= 0 && k < block))
  //  { 
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T3, T1_3, T2_3, T3_3, T4_3) //async(stream) 
#pragma acc loop collapse(3) private(i, j, k)
  //#pragma omp target is_device_ptr(T1, T1_1, T1_2) device(ID) private(i, j, k)
  for(i = 0; i < width; ++i){
    for(j = 0; j < width; ++j){
      for(k = 0; k < block; ++k){  
	if(i < breadth)
	  {
	    T3[i + block * (j + width * k)] =  T1_3[i + breadth * (j + width * k)]; 
	    T3[breadth + i + block * (j + width * k)] = T2_3[i + breadth * (j + width * k)];
	    T3[(breadth*2) + i + block * (j + width * k)] = T3_3[i + width * (j + width * k)]; 
	    T3[(breadth*3)+ 2 + i + block * (j + width * k)] = T4_3[i + breadth * (j + width * k)];  
	  }
	else
	  {
	    T3[(breadth*2) + i + block * (j + width * k)] = T3_3[i + width * (j + width * k)];
	  }
      } 
    }
  }
}


void to_y_to_k_direction_T41(double *T4, double *T1_4, double *T2_4, 
			     double *T3_4, double *T4_4,
			     int block, int width, int breadth, int ID)//, int stream)
{
  int i, j, k;
  
  //i = blockIdx.x * blockDim.x + threadIdx.x;
  //j = blockIdx.y * blockDim.y + threadIdx.y;
  //k = blockIdx.z * blockDim.z + threadIdx.z;
  //if( (i >= 0 && i < width) && ( j >= 0 && j < width ) && ( k >= 0 && k < block))
  //  { 
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T4, T1_4, T2_4, T3_4, T4_4) //async(stream) 
#pragma acc loop collapse(3) private(i, j, k)
  //#pragma omp target is_device_ptr(T1, T1_1, T1_2) device(ID) private(i, j, k)
  for(i = 0; i < width; ++i){
    for(j = 0; j < width; ++j){
      for(k = 0; k < block; ++k){  
	if(i < breadth)
	  {
	    T4[i + block * (j + width * k)] =  T1_4[i + breadth * (j + width * k)]; 
	    T4[breadth + i + block * (j + width * k)] = T2_4[i + breadth * (j + width * k)];
	    T4[(breadth*2) + i + block * (j + width * k)] = T3_4[i + breadth * (j + width * k)]; 
	    T4[(breadth*3) + i + block * (j + width * k)] = T4_4[i + width * (j + width * k)]; 
	  }
	else    
	  {
	    T4[(breadth*3) + i + block * (j + width * k)] = T4_4[i + width * (j+width * k)]; 
	  }
      }
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
// print the solution to the vtk file for the visualization
////////////////////////////////////////////////////////////////////////////////
void Write_VTK_Structured_Grid(_DOUBLE_ *T, int nx, int ny, int nz, int eb, const char *filename)
{

  FILE *out;
  out = fopen(filename,"w");
  fprintf(out,"# vtk DataFile Version 3.0\n");
  fprintf(out,"Header -- Folding Generalized Distanze or other related.\n");
  fprintf(out,"ASCII\n");
  fprintf(out,"DATASET STRUCTURED_GRID\n");
  fprintf(out,"DIMENSIONS %d %d %d\n", nz, ny, nx);
  fprintf(out,"POINTS %d _DOUBLE_ \n", nx* ny* nz);
  int i,j,k;
  for(i = eb; i < nx+eb; i++) 
    {
      for(j = eb; j < ny+eb; j++) 
	{
	  for(k = eb; k < nz+eb; k++) 
	    {
	      //	      fprintf(out,"%f %f %f\n", _xmin+i * DX, _ymin+j*dy, _zmin+k*dz);
	    }
	}
    }
  fprintf(out,"POINT_DATA %d\n",nx*ny*nz);
  fprintf(out,"SCALARS scalar_name _DOUBLE_ 1\n");
  fprintf(out,"LOOKUP_TABLE default\n");
  for(i = eb; i < nx+eb; i++) 
    {
      for(j = eb; j < ny+eb; j++)
	{
	  for(k = eb; k < nz+eb; k++) 
	    {
	      fprintf(out,"%f\n",T[i+(_nx+2)*(j+(_ny+2)*k)]);
	    }
	}
    }

  fclose(out);
}


////////////////////////////////////////////////////////////////////////////////
// cpu compute time 
////////////////////////////////////////////////////////////////////////////////
double wtime() 
{
  struct timeval tv;
  gettimeofday(&tv,0);
  return (double) tv.tv_sec + 1e-6 * tv.tv_usec;
}


////////////////////////////////////////////////////////////////////////////////
// main programme starts from here 
////////////////////////////////////////////////////////////////////////////////
int main()
{

  int i, j, k;

  _DOUBLE_ DX = ( _xmax - _xmin) / (_nx - 1.0);
  _DOUBLE_ dy = ( _ymax - _ymin) / (_ny - 1.0);
  _DOUBLE_ dz = ( _zmax - _zmin) / (_nz - 1.0);
  
  _DOUBLE_ DXYP = sqrt(DX*DX + dy*dy);
  _DOUBLE_ DXZP = sqrt(DX*DX + dz*dz);
  _DOUBLE_ DYZP = sqrt(dy*dy + dz*dz);
  _DOUBLE_ DXYZP =  sqrt(DX*DX + dy*dy + dz*dz);
   
  // total length
  int block = _nx + 2;
  // sub block matrix with ghost node
  int width = (_nx/4) + 2;
  // for data transfer sub block size
  int breadth = _nx/4;

  
  // memory allocation for the calculation 1D- > 3D
  _DOUBLE_ *restrict T = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * block * block * block);
    
  printf(" Initialization is started\n");
  // compute time starts 
  double start_initT = wtime();

  // Initialize the 3D-array for the further calculation
  // in this case 1D-array will represent the 3D-array 
  ImplicitInitialiser(T, _nx, _ny, _nz, DX, dy, dz);

  // compute time stops
  double end_initT = wtime();

  printf(" Initialization is finished\n");
  printf(" preparation work before go to GPU \n");

  // const velocity
  double F = _Fc;

  // memmory allocation of blocks of data T1, T2. T3, T4 for 4- GPU's
  _DOUBLE_ *restrict T1 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * (width) * block * block);
  _DOUBLE_ *restrict T2 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * (width) * block * block);
  _DOUBLE_ *restrict T3 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * (width) * block * block);
  _DOUBLE_ *restrict T4 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * (width) * block * block);

  // data partion from the initialized T
  for(k = 0 ;  k < block; k++)
    {
      for(j = 0; j < width; j++)
	{
	  for(i = 0; i < block; i++)
	    {
	      T1[i + block * (j + width * k)] = T[i + block * (j + block * k)];
	      T2[i + block * (j + width * k)] = T[i + block * (j + (breadth *1) + block * k)];
	      T3[i + block * (j + width * k)] = T[i + block * (j + (breadth *2) + block * k)];
	      T4[i + block * (j + width * k)] = T[i + block * (j + (breadth *3) + block * k)];
	    }
	}
    }
  
  // data size for sub blocks transfer from CPU-GPU vice-versa
  _DOUBLE_ size = (width) * block * block * sizeof(_DOUBLE_);

  // data size of boundary layer transfer between GPU-GPU
  _DOUBLE_ buff_size = sizeof(_DOUBLE_) * _nx;

  // data size for partial data transfer from GPU-CPU vice-versa
  _DOUBLE_ data_size = sizeof(_DOUBLE_) * breadth * width * block;
  
  _DOUBLE_ *restrict c_T1 = NULL;   // GPU block 1
  _DOUBLE_ *restrict c_T1_1 = NULL; // GPU partial data transfer
  _DOUBLE_ *restrict c_T1_2 = NULL; // GPU partial data transfer
  _DOUBLE_ *restrict c_T1_3 = NULL; // GPU partial data transfer
  _DOUBLE_ *restrict c_T1_4 = NULL; // GPU partial data transfer
  
  _DOUBLE_ *restrict c_T2 = NULL;   // GPU block 2
  _DOUBLE_ *restrict c_T2_1 = NULL; // GPU partial data transfer
  _DOUBLE_ *restrict c_T2_2 = NULL; // GPU partial data transfer
  _DOUBLE_ *restrict c_T2_3 = NULL; // GPU partial data transfer
  _DOUBLE_ *restrict c_T2_4 = NULL; // GPU partial data transfer
  

  _DOUBLE_ *restrict c_T3 = NULL;   // GPU block 3
  _DOUBLE_ *restrict c_T3_1 = NULL; // GPU partial data transfer
  _DOUBLE_ *restrict c_T3_2 = NULL; // GPU partial data transfer
  _DOUBLE_ *restrict c_T3_3 = NULL; // GPU partial data transfer
  _DOUBLE_ *restrict c_T3_4 = NULL; // GPU partial data transfer
  
  _DOUBLE_ *restrict c_T4 = NULL;    // GPU block 4
  _DOUBLE_ *restrict c_T4_1 = NULL; // GPU partial data transfer
  _DOUBLE_ *restrict c_T4_2 = NULL; // GPU partial data transfer
  _DOUBLE_ *restrict c_T4_3 = NULL; // GPU partial data transfer
  _DOUBLE_ *restrict c_T4_4 = NULL; // GPU partial data transfer

  _DOUBLE_ *restrict a_1_2 = NULL; // GPU buff for block 1 boundary from comp
  _DOUBLE_ *restrict a_2_1 = NULL; // GPU buff for block 2 boundary from comp
  _DOUBLE_ *restrict a_2_2 = NULL; // GPU buff for block 2 boundary from comp
  _DOUBLE_ *restrict a_3_1 = NULL; // GPU buff for block 3 boundary from comp
  _DOUBLE_ *restrict a_3_2 = NULL; // GPU buff for block 3 boundary from comp
  _DOUBLE_ *restrict a_4_1 = NULL; // GPU buff for block 4 boundary from comp
  
  _DOUBLE_ *restrict b_1_2 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * _nx); // CPU buff for 1 boundary transfer
  _DOUBLE_ *restrict b_2_1 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * _nx); // CPU buff for 2 boundary transfer
  _DOUBLE_ *restrict b_2_2 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * _nx); // CPU buff for 2 boundary transfer 
  _DOUBLE_ *restrict b_3_1 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * _nx); // CPU buff for 3 boundary transfer 
  _DOUBLE_ *restrict b_3_2 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * _nx); // CPU buff for 3 boundary transfer 
  _DOUBLE_ *restrict b_4_1 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * _nx); // CPU buff for 4 boundary transfer 
  
  // CPU buffer for data transfer  between GPU's
  _DOUBLE_ *restrict d_1_2 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * breadth * block * width);     
  _DOUBLE_ *restrict d_1_3 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * breadth * block * width);     
  _DOUBLE_ *restrict d_1_4 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * breadth * block * width);     

  _DOUBLE_ *restrict d_2_1 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * breadth * block * width);     
  _DOUBLE_ *restrict d_2_3 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * breadth * block * width);     
  _DOUBLE_ *restrict d_2_4 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * breadth * block * width);     

  _DOUBLE_ *restrict d_3_1 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * breadth * block * width);     
  _DOUBLE_ *restrict d_3_2 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * breadth * block * width);     
  _DOUBLE_ *restrict d_3_4 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * breadth * block * width);     

  _DOUBLE_ *restrict d_4_1 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * breadth * block * width);     
  _DOUBLE_ *restrict d_4_2 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * breadth * block * width);     
  _DOUBLE_ *restrict d_4_3 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * breadth * block * width);     
  
  
  // set cudadevice(0) and allocate the memory of its data
  //
  acc_set_device_num(0, acc_device_nvidia);
  c_T1 = acc_malloc((size_t) size );
  c_T1_1 = acc_malloc((size_t) sizeof(_DOUBLE_) * width * block * width);
  c_T1_2 = acc_malloc((size_t) sizeof(_DOUBLE_) * breadth * block * width);
  c_T1_3 = acc_malloc((size_t) sizeof(_DOUBLE_) * breadth * block * width);
  c_T1_4 = acc_malloc((size_t) sizeof(_DOUBLE_) * breadth * block * width);
  a_1_2 = acc_malloc((size_t) sizeof(_DOUBLE_) * _nx);
  acc_memcpy_to_device(c_T1, T1, size);

  // set cudadevice(1) and allocate the memory of its data
  //
  acc_set_device_num(1, acc_device_nvidia);
  c_T2 = acc_malloc((size_t) size );
  c_T2_1 = acc_malloc((size_t) sizeof(_DOUBLE_) * breadth * block * width);
  c_T2_2 = acc_malloc((size_t) sizeof(_DOUBLE_) * width * block * width);
  c_T2_3 = acc_malloc((size_t) sizeof(_DOUBLE_) * breadth * block * width);
  c_T2_4 = acc_malloc((size_t) sizeof(_DOUBLE_) * breadth * block * width);
  a_2_1 = acc_malloc((size_t) sizeof(_DOUBLE_) * _nx);
  a_2_2 = acc_malloc((size_t) sizeof(_DOUBLE_) * _nx);
  acc_memcpy_to_device(c_T2, T2, size);

  // set cudadevice(2) and allocate the memory of its data
  //
  acc_set_device_num(2, acc_device_nvidia);
  c_T3 = acc_malloc((size_t) size );
  c_T3_1 = acc_malloc((size_t) sizeof(_DOUBLE_) * breadth * block * width);
  c_T3_2 = acc_malloc((size_t) sizeof(_DOUBLE_) * breadth * block * width);
  c_T3_3 = acc_malloc((size_t) sizeof(_DOUBLE_) * width * block * width);
  c_T3_4 = acc_malloc((size_t) sizeof(_DOUBLE_) * breadth * block * width);
  a_3_1 = acc_malloc((size_t) sizeof(_DOUBLE_) * _nx);
  a_3_2 = acc_malloc((size_t) sizeof(_DOUBLE_) * _nx);
  acc_memcpy_to_device(c_T3, T3, size);

  // set cudadevice(3) and allocate the memory of its data
  //
  acc_set_device_num(3, acc_device_nvidia);
  c_T4 = acc_malloc((size_t) size );
  c_T4_1 = acc_malloc((size_t) sizeof(_DOUBLE_) * breadth * block * width);
  c_T4_2 = acc_malloc((size_t) sizeof(_DOUBLE_) * breadth * block * width);
  c_T4_3 = acc_malloc((size_t) sizeof(_DOUBLE_) * breadth * block * width);
  c_T4_4 = acc_malloc((size_t) sizeof(_DOUBLE_) * width * block * width);
  a_4_1 = acc_malloc((size_t) sizeof(_DOUBLE_) * _nx);
  acc_memcpy_to_device(c_T4, T4, size);

  int top = 1; // halo offset top most block
  int bot = _nx/4; // halo offset for bottom most block
  
  printf(" preparation work finished before go to GPU \n");   

  printf(" GPU is starts from here \n");

  // sweep count
  int sweep,tot=8;
                                            
  // time count starts for MULTIPLE GPU
  double compute_timer = 0.0;
  double data_timer_1 = 0.0;
  double data_timer_2 = 0.0;
  double direction_1 = 0.0;
  double direction_2 = 0.0;
  double direction_3 = 0.0;
  double direction_4 = 0.0;
  double direction_5 = 0.0;
  double direction_6 = 0.0;
  
  compute_timer -= omp_get_wtime();

  for(sweep = 0; sweep < tot; sweep++)
    {
      printf(" sweep no %d\n", sweep+1);
      direction_1 -=  omp_get_wtime();      
      for(i = 1; i < _nx ; i++)
	{
	  function_x_1_down(c_T1, a_1_2, i, DX, dz, dy, 
			    DYZP, DXZP, DXYP, F, _ayy, _azz, _axx, width, 0, 0);
	  function_x_t_down(c_T1, i, DX, dz, dy, 
			    DYZP, DXZP, DXYP, F, _ayy, _azz, _axx, width, 0, 1);
	  acc_memcpy_from_device_async(b_1_2, a_1_2, buff_size, 0);
	  
	  //	  cudaSetDevice(1);
	  function_x_1_down(c_T2, a_2_2, i, DX, dz, dy, 
			    DYZP, DXZP, DXYP, F, _ayy, _azz, _axx, width, 1, 2); 
	  function_x_2_down(c_T2, a_2_1, i, DX, dz, dy, 
			    DYZP, DXZP, DXYP, F, _ayy, _azz, _axx, width, 1, 3);
	  function_x_down(c_T2, i, DX, dz, dy, 
			  DYZP, DXZP, DXYP, F, _ayy, _azz, _axx, width, 1, 4); 
	  acc_memcpy_from_device_async(b_2_2, a_2_2, buff_size, 2);
	  acc_memcpy_from_device_async(b_2_1, a_2_1, buff_size, 3);
	  
	  //	  cudaSetDevice(2);
	  function_x_1_down(c_T3, a_3_2, i,  DX, dz, dy, 
			    DYZP, DXZP, DXYP, F, _ayy, _azz, _axx, width, 2, 5);
	  function_x_2_down(c_T3, a_3_1, i,  DX, dz, dy, 
			    DYZP, DXZP, DXYP, F, _ayy, _azz, _axx, width, 2, 6);
	  function_x_down(c_T3, i, DX, dz, dy, 
			  DYZP, DXZP, DXYP, F, _ayy, _azz, _axx, width, 2, 7);
	  acc_memcpy_from_device_async(b_3_2, a_3_2, buff_size, 5);
	  acc_memcpy_from_device_async(b_3_1, a_3_1, buff_size, 6);
	  
	  //	  cudaSetDevice(3);
	  function_x_2_down(c_T4, a_4_1, i, DX, dz, dy, 
			    DYZP, DXZP, DXYP, F, _ayy, _azz, _axx, width, 3, 8);
	  function_x_b_down(c_T4, i, DX, dz, dy, 
			    DYZP, DXZP, DXYP, F, _ayy, _azz, _axx, width, 3, 9);
	  acc_memcpy_from_device_async(b_4_1, a_4_1, buff_size, 8);
	  
	  
	  
	  acc_wait_async(0,3);
	  acc_memcpy_to_device_async(a_1_2, b_2_1, buff_size, 0);
	  x_copy_to_1(1, a_1_2, c_T1, i-1, block, width, bot+1, 0, 0);
	  
	  acc_wait_async(3,0);
	  acc_memcpy_to_device_async(a_2_1, b_1_2, buff_size, 3);
	  x_copy_to_2(1, a_2_1, c_T2, i-1, block, width, bot+1, 1, 3);
	  acc_wait_async(2,6);
	  acc_memcpy_to_device_async(a_2_2, b_3_1, buff_size, 2);   
	  x_copy_to_2_(1, a_2_2, c_T2, i-1, block, width, bot+1, 1, 2);
	  
	  acc_wait_async(6,2);
	  acc_memcpy_to_device_async(a_3_1, b_2_2, buff_size, 6);
	  x_copy_to_3(1, a_3_1, c_T3, i-1, block, width, bot+1, 2, 6);
	  acc_wait_async(5,8);
	  acc_memcpy_to_device_async(a_3_2, b_4_1, buff_size, 5);
	  x_copy_to_3_(1, a_3_2, c_T3, i-1, block, width, bot+1, 2, 5);
	  
	  acc_wait_async(8,5);
	  acc_memcpy_to_device_async(a_4_1, b_3_2, buff_size, 8);
	  x_copy_to_4(1, a_4_1, c_T4, i-1, block, width,bot+1, 3, 8);
	  acc_wait_all();
	}
      direction_1 +=  omp_get_wtime();

      direction_2 -=  omp_get_wtime();      
      for(i = _nx; i > 1 ; i--)
	{
	  // halo
	  //	  cudaSetDevice(0);
	  function_x_1_up(c_T1, a_1_2, i, DX, dz, dy, 
			  DYZP, DXZP, DXYP, F, -_ayy, -_azz, -_axx, width, 0, 0);
	  function_x_t_up(c_T1, i, DX, dz, dy, 
			  DYZP, DXZP, DXYP, F, -_ayy, -_azz, -_axx, width, 0, 1);
	  acc_memcpy_from_device_async(b_1_2, a_1_2, buff_size, 0); 

	  //	  cudaSetDevice(1);
	  function_x_1_up(c_T2, a_2_2, i,  DX, dz, dy, 
			  DYZP, DXZP, DXYP, F, -_ayy, -_azz, -_axx, width, 1, 2);
	  function_x_2_up(c_T2, a_2_1, i,  DX, dz, dy, 
			  DYZP, DXZP, DXYP, F, -_ayy, -_azz, -_axx, width, 1, 3);
	  function_x_up(c_T2, i, DX, dz, dy, 
			DYZP, DXZP, DXYP, F, -_ayy, -_azz, -_axx, width, 1, 4);
	  acc_memcpy_from_device_async(b_2_2, a_2_2, buff_size, 2);
	  acc_memcpy_from_device_async(b_2_1, a_2_1, buff_size, 3);

	  //	  cudaSetDevice(2);
	  function_x_1_up(c_T3, a_3_2, i,  DX, dz, dy, 
			  DYZP, DXZP, DXYP, F, -_ayy, -_azz, -_axx, width, 2, 5);
	  function_x_2_up(c_T3, a_3_1, i,  DX, dz, dy, 
			  DYZP, DXZP, DXYP, F, -_ayy, -_azz, -_axx, width, 2, 6);
	  function_x_up(c_T3, i, DX, dz, dy, 
			DYZP, DXZP, DXYP, F, -_ayy, -_azz, -_axx, width, 2, 7);
	  acc_memcpy_from_device_async(b_3_2, a_3_2, buff_size, 5); 
	  acc_memcpy_from_device_async(b_3_1, a_3_1, buff_size, 6);

	  //	  cudaSetDevice(3);
	  function_x_2_up(c_T4, a_4_1, i,  DX, dz, dy, 
			  DYZP, DXZP, DXYP, F, -_ayy, -_azz, -_axx, width, 3, 8);
	  function_x_b_up(c_T4, i, DX, dz, dy, 
			  DYZP, DXZP, DXYP, F, -_ayy, -_azz, -_axx, width, 3, 9);
	  acc_memcpy_from_device_async(b_4_1, a_4_1, buff_size, 8); 

	  acc_wait_async(0,3);
	  acc_memcpy_to_device_async(a_1_2, b_2_1, buff_size, 0);
	  x_copy_to_1(-1, a_1_2, c_T1, i-1, block, width, bot+1, 0, 0);

	  acc_wait_async(3,0);
	  acc_memcpy_to_device_async(a_2_1, b_1_2, buff_size, 3);
	  x_copy_to_2(-1, a_2_1, c_T2, i-1, block, width, bot+1, 1, 3);
	  acc_wait_async(2,6);
	  acc_memcpy_to_device_async(a_2_2, b_3_1, buff_size, 2); 
	  x_copy_to_2_(-1, a_2_2, c_T2, i-1, block, width, bot+1, 1, 2);
	  	  
	  acc_wait_async(6,2);
	  acc_memcpy_to_device_async(a_3_1, b_2_2, buff_size, 6);
	  x_copy_to_3(-1, a_3_1, c_T3, i-1, block, width, bot+1, 2, 6);
	  acc_wait_async(5,8);
	  acc_memcpy_to_device_async(a_3_2, b_4_1, buff_size, 5);
	  x_copy_to_3_(-1, a_3_2, c_T3, i-1, block, width, bot+1, 2, 5);
	  
	  acc_wait_async(8,5);
	  acc_memcpy_to_device_async(a_4_1, b_3_2, buff_size, 8);
	  x_copy_to_4(-1, a_4_1, c_T4, i-1, block, width, bot+1, 3, 8);
	  acc_wait_all();
	}
      direction_2 +=  omp_get_wtime();

      direction_3 -=  omp_get_wtime();      
      for(i = 1; i < _nx; i++)
	{

	  function_z_1_up(c_T1, a_1_2, i, dz, dy, DX, 
			  DXYP, DYZP, DXZP, F, _axx, _ayy, _azz, width, 0, 0);
	  function_z_t_up(c_T1, i, dz, dy, DX, 
			  DXYP, DYZP, DXZP, F, _axx, _ayy, _azz, width, 0, 1);
	  acc_memcpy_from_device_async(b_1_2, a_1_2, buff_size, 0); 
  

	  function_z_1_up(c_T2, a_2_2, i, dz, dy, DX, 
			  DXYP, DYZP, DXZP, F, _axx, _ayy, _azz, width, 1, 2);
	  function_z_2_up(c_T2, a_2_1, i, dz, dy, DX, 
			  DXYP, DYZP, DXZP, F, _axx, _ayy, _azz, width, 1, 3);
	  function_z_up(c_T2, i, dz, dy, DX, 
			DXYP, DYZP, DXZP, F, _axx, _ayy, _azz, width, 1, 4);  
	  acc_memcpy_from_device_async(b_2_2, a_2_2, buff_size, 2);
	  acc_memcpy_from_device_async(b_2_1, a_2_1, buff_size, 3);
	  

	  function_z_1_up(c_T3, a_3_2, i, dz, dy, DX, 
			  DXYP, DYZP, DXZP, F, _axx, _ayy, _azz, width, 2, 5);
	  function_z_2_up(c_T3, a_3_1, i, dz, dy, DX, 
			  DXYP, DYZP, DXZP, F, _axx, _ayy, _azz, width, 2, 6);
	  function_z_up(c_T3, i, dz, dy, DX, 
			DXYP, DYZP, DXZP, F, _axx, _ayy, _azz, width, 2, 7);
	  acc_memcpy_from_device_async(b_3_2, a_3_2, buff_size, 5); 
	  acc_memcpy_from_device_async(b_3_1, a_3_1, buff_size, 6);
	  

	  function_z_2_up(c_T4, a_4_1, i, dz, dy, DX, 
			  DXYP, DYZP, DXZP, F, _axx, _ayy, _azz, width, 3, 8);
	  function_z_b_up(c_T4, i, dz, dy, DX, 
			  DXYP, DYZP, DXZP, F, _axx, _ayy, _azz, width, 3, 9);
	  acc_memcpy_from_device_async(b_4_1, a_4_1, buff_size, 8);


	  acc_wait_async(0,3);
	  acc_memcpy_to_device_async(a_1_2, b_2_1, buff_size, 0);
	  z_copy_to_1(1, a_1_2, c_T1, i-1, block, width, bot+1, 0, 0);

	  acc_wait_async(3,0);
	  acc_memcpy_to_device_async(a_2_1, b_1_2, buff_size, 3);
	  z_copy_to_2(1, a_2_1, c_T2, i-1, block, width, bot+1, 1, 3);
	  acc_wait_async(2,6);
	  acc_memcpy_to_device_async(a_2_2, b_3_1, buff_size, 2);
	  z_copy_to_2_(1, a_2_2, c_T2, i-1, block, width, bot+1, 1, 2);

	  acc_wait_async(6,2);
	  acc_memcpy_to_device_async(a_3_1, b_2_2, buff_size, 6);
	  z_copy_to_3(1, a_3_1, c_T3, i-1, block, width, bot+1, 2, 6);
	  acc_wait_async(5,8);
	  acc_memcpy_to_device_async(a_3_2, b_4_1, buff_size, 5);
	  z_copy_to_3_(1, a_3_2, c_T3, i-1, block, width, bot+1, 2, 5);

	  acc_wait_async(8,5);
	  acc_memcpy_to_device_async(a_4_1, b_3_2, buff_size, 8);
	  z_copy_to_4(1, a_4_1, c_T4, i-1, block, width, bot+1, 3, 8);
	  acc_wait_all();
	} 
      direction_3 +=  omp_get_wtime();
      
      direction_4 -=  omp_get_wtime();           
      for(i  = _ny; i > 1; i--)
	{
	  function_z_1_down(c_T1, a_1_2, i, dz, dy, DX, 
			    DXYP, DYZP, DXZP, F, -_axx, -_ayy, -_azz, width, 0, 0);
	  function_z_t_down(c_T1, i, dz, dy, DX, 
			    DXYP, DYZP, DXZP, F, -_axx, -_ayy, -_azz, width, 0, 1);
	  acc_memcpy_from_device_async(b_1_2, a_1_2, buff_size, 0); 
	  
	  function_z_1_down(c_T2, a_2_2, i, dz, dy, DX, 
			    DXYP, DYZP, DXZP, F, -_axx, -_ayy, -_azz, width, 1, 2);
	  function_z_2_down(c_T2, a_2_1, i, dz, dy, DX, 
			    DXYP, DYZP, DXZP, F, -_axx, -_ayy, -_azz, width, 1, 3);
	  function_z_down(c_T2, i, dz, dy, DX, 
			  DXYP, DYZP, DXZP, F, -_axx, -_ayy, -_azz, width, 1, 4);
	  acc_memcpy_from_device_async(b_2_2, a_2_2, buff_size, 2);
	  acc_memcpy_from_device_async(b_2_1, a_2_1, buff_size, 3);

	  function_z_1_down(c_T3, a_3_2, i, dz, dy, DX, 
			    DXYP, DYZP, DXZP, F, -_axx, -_ayy, -_azz, width, 2, 5);
	  function_z_2_down(c_T3, a_3_1, i, dz, dy, DX, 
			    DXYP, DYZP, DXZP, F, -_axx, -_ayy, -_azz, width, 2, 6);
	  function_z_down(c_T3, i, dz, dy, DX, 
			  DXYP, DYZP, DXZP, F, -_axx, -_ayy, -_azz, width, 2, 7);
	  acc_memcpy_from_device_async(b_3_2, a_3_2, buff_size, 5);
	  acc_memcpy_from_device_async(b_3_1, a_3_1, buff_size, 6);

	  function_z_2_down(c_T4, a_4_1, i, dz, dy, DX, 
			    DXYP, DYZP, DXZP, F, -_axx, -_ayy, -_azz, width, 3, 8);
	  function_z_b_down(c_T4, i, dz, dy, DX, 
			    DXYP, DYZP, DXZP, F, -_axx, -_ayy, -_azz, width, 3, 9);
	  acc_memcpy_from_device_async(b_4_1, a_4_1, buff_size, 8);  

	  acc_wait_async(0,3);
	  acc_memcpy_to_device_async(a_1_2, b_2_1, buff_size, 0);
	  z_copy_to_1(-1, a_1_2, c_T1, i-1, block, width, bot+1, 0, 0);

	  acc_wait_async(3,0);
	  acc_memcpy_to_device_async(a_2_1, b_1_2, buff_size, 3);
	  z_copy_to_2(-1, a_2_1,c_T2, i-1, block, width, bot+1, 1, 3);
	  acc_wait_async(2,6);
	  acc_memcpy_to_device_async(a_2_2, b_3_1, buff_size, 2);
	  z_copy_to_2_(-1, a_2_2,c_T2, i-1, block, width, bot+1, 1, 2);

	  acc_wait_async(6,2);
	  acc_memcpy_to_device_async(a_3_1, b_2_2, buff_size, 6);
	  z_copy_to_3(-1, a_3_1, c_T3, i-1, block, width, bot+1, 2, 6);
	  acc_wait_async(5,8);
	  acc_memcpy_to_device_async(a_3_2, b_4_1, buff_size, 5);
	  z_copy_to_3_(-1, a_3_2, c_T3, i-1, block, width, bot+1, 2, 5);

	  acc_wait_async(8,5);
	  acc_memcpy_to_device_async(a_4_1, b_3_2, buff_size, 8);
	  z_copy_to_4(-1, a_4_1, c_T4, i-1, block, width, bot+1, 3, 8);
	  acc_wait_all();
	}
      direction_4 +=  omp_get_wtime();

      data_timer_1 -=  omp_get_wtime();
      // copy GPU 1 data in to 1 buffers             
      from_k_to_y_direction_T1(c_T1, c_T1_1, c_T1_2, c_T1_3, c_T1_4, block, width, breadth, 0);
      // copy GPU 2 data in to 2 buffers             
      from_k_to_y_direction_T2(c_T2, c_T2_1, c_T2_2, c_T2_3, c_T2_4, block, width, breadth, 1);
      // copy GPU 3 data in to 3 buffers             
      from_k_to_y_direction_T3(c_T3, c_T3_1, c_T3_2, c_T3_3, c_T3_4, block, width, breadth, 2);
      // copy GPU 4 data in to 4 buffers             
      from_k_to_y_direction_T4(c_T4, c_T4_1, c_T4_2, c_T4_3, c_T4_4, block, width, breadth, 3);
      //      acc_wait_all();


      // copied data to CPU only 3 blocks keep 1 block in GPU itself      
      acc_memcpy_from_device(d_1_2, c_T1_2, data_size);
      acc_memcpy_from_device(d_1_3, c_T1_3, data_size);
      acc_memcpy_from_device(d_1_4, c_T1_4, data_size);
      
      // copied data to CPU only 3 blocks keep 1 block in GPU itself      
      acc_memcpy_from_device(d_2_1, c_T2_1, data_size);
      acc_memcpy_from_device(d_2_3, c_T2_3, data_size);
      acc_memcpy_from_device(d_2_4, c_T2_4, data_size);
      
      // copied data to CPU only 3 blocks keep 1 block in GPU itself      
      acc_memcpy_from_device(d_3_1, c_T3_1, data_size);
      acc_memcpy_from_device(d_3_2, c_T3_2, data_size);
      acc_memcpy_from_device(d_3_4, c_T3_4, data_size);
      
      // copied data to CPU only 3 blocks keep 1 block in GPU itself      
      acc_memcpy_from_device(d_4_1, c_T4_1, data_size);
      acc_memcpy_from_device(d_4_2, c_T4_2, data_size);
      acc_memcpy_from_device(d_4_3, c_T4_3, data_size);

      //      acc_wait_all();
      // copied data to GPU from CPU
      //      acc_wait_async(0,1);
      acc_memcpy_to_device(c_T1_2, d_2_1, data_size);
      //      acc_wait_async(0,2);		    
      acc_memcpy_to_device(c_T1_3, d_3_1, data_size);
      //      acc_wait_async(0,3);		   
      acc_memcpy_to_device(c_T1_4, d_4_1, data_size);
      
      // copied data to GPU from CPU
      //      acc_wait_async(1,0);		    
      acc_memcpy_to_device(c_T2_1, d_1_2, data_size);
      //      acc_wait_async(1,2);		    
      acc_memcpy_to_device(c_T2_3, d_3_2, data_size);
      //      acc_wait_async(1,2);		    
      acc_memcpy_to_device(c_T2_4, d_4_2, data_size);
      
      // copied data to GPU from CPU
      //      acc_wait_async(2,0);		    
      acc_memcpy_to_device(c_T3_1, d_1_3, data_size);
      //      acc_wait_async(2,1);		    
      acc_memcpy_to_device(c_T3_2, d_2_3, data_size);
      //      acc_wait_async(2,3);		    
      acc_memcpy_to_device(c_T3_4, d_4_3, data_size);
      
      // copied data to GPU from CPU
      //      acc_wait_async(3,0);		    
      acc_memcpy_to_device(c_T4_1, d_1_4, data_size);
      //      acc_wait_async(3,1);		   
      acc_memcpy_to_device(c_T4_2, d_2_4, data_size);
      //      acc_wait_async(3,2);		    
      acc_memcpy_to_device(c_T4_3, d_3_4, data_size);      
      //      acc_wait_all();
      
      // mutilpe GPU biuffer in to 1 GPU      
      to_k_to_y_direction_T1(c_T1, c_T1_1, c_T1_2, c_T1_3, c_T1_4, block, width, breadth, 0);
      
      // mutilpe GPU biuffer in to 2 GPU      
      to_k_to_y_direction_T2(c_T2, c_T2_1, c_T2_2, c_T2_3, c_T2_4, block, width, breadth, 1);
      
      // mutilpe GPU biuffer in to 3 GPU      
      to_k_to_y_direction_T3(c_T3, c_T3_1, c_T3_2, c_T3_3, c_T3_4, block, width, breadth, 2);
     
      // mutilpe GPU biuffer in to 4 GPU      
      to_k_to_y_direction_T4(c_T4, c_T4_1, c_T4_2, c_T4_3, c_T4_4, block, width, breadth, 3);
      //      acc_wait_all();
      data_timer_1 +=  omp_get_wtime();

      direction_5 -=  omp_get_wtime();
      for(i = _ny; i > 1; i--)
	{
	  function_y_1_up(c_T1, a_1_2, i, dy, DX, dz, 
			  DXZP, DXYP, DYZP, F, -_azz, -_axx, -_ayy, width, 0, 0);
	  function_y_t_up(c_T1, i, dy, DX, dz, 
			  DXZP, DXYP, DYZP, F, -_azz, -_axx, -_ayy, width, 0, 1);
          acc_memcpy_from_device_async(b_1_2, a_1_2, buff_size, 0);


	  function_y_1_up(c_T2, a_2_2, i, dy, DX, dz, 
			  DXZP, DXYP, DYZP, F, -_azz, -_axx, -_ayy, width, 1, 2);
	  function_y_2_up(c_T2, a_2_1, i, dy, DX, dz, 
			  DXZP, DXYP, DYZP, F, -_azz, -_axx, -_ayy, width, 1, 3);
	  function_y_up(c_T2, i, dy, DX, dz, 
			DXZP, DXYP, DYZP, F, -_azz, -_axx, -_ayy, width, 1, 4);
	  acc_memcpy_from_device_async(b_2_2, a_2_2, buff_size, 2);
          acc_memcpy_from_device_async(b_2_1, a_2_1, buff_size, 3);


	  function_y_1_up(c_T3, a_3_2, i, dy, DX, dz, 
			  DXZP, DXYP, DYZP, F, -_azz, -_axx, -_ayy, width, 2, 5);
	  function_y_2_up(c_T3, a_3_1, i, dy, DX, dz, 
			  DXZP, DXYP, DYZP, F, -_azz, -_axx, -_ayy, width, 2, 6);
	  function_y_up(c_T3, i, dy, DX, dz, 
			DXZP, DXYP, DYZP, F, -_azz, -_axx, -_ayy, width, 2, 7);
	  acc_memcpy_from_device_async(b_3_2, a_3_2, buff_size, 5);
          acc_memcpy_from_device_async(b_3_1, a_3_1, buff_size, 6);

	  
	  function_y_2_up(c_T4, a_4_1, i, dy, DX, dz, 
			  DXZP, DXYP, DYZP, F, -_azz, -_axx, -_ayy, width, 3, 8);
	  function_y_b_up(c_T4, i, dy, DX, dz, 
			  DXZP, DXYP, DYZP, F, -_azz, -_axx, -_ayy, width, 3, 9);
	  acc_memcpy_from_device_async(b_4_1, a_4_1, buff_size, 8);

	  acc_wait_async(0,3);
	  acc_memcpy_to_device_async(a_1_2, b_2_1, buff_size, 0);
	  y_copy_to_1(-1, a_1_2, c_T1, i-1, block, width,bot+1, 0, 0);

	  acc_wait_async(3,0);
	  acc_memcpy_to_device_async(a_2_1, b_1_2, buff_size, 3);
	  y_copy_to_2(-1, a_2_1, c_T2, i-1, block, width, bot+1, 1, 3);
	  acc_wait_async(2,6);
	  acc_memcpy_to_device_async(a_2_2, b_3_1, buff_size, 2);
	  y_copy_to_2_(-1, a_2_2, c_T2, i-1, block, width, bot+1, 1, 2);

	  acc_wait_async(6,2);
	  acc_memcpy_to_device_async(a_3_1, b_2_2, buff_size, 6);
	  y_copy_to_3(-1, a_3_1, c_T3, i-1, block, width, bot+1, 2, 6);
	  acc_wait_async(5,8);
	  acc_memcpy_to_device_async(a_3_2, b_4_1, buff_size, 5);
	  y_copy_to_3_(-1, a_3_2, c_T3, i-1, block, width, bot+1, 2, 5);

	  acc_wait_async(8,5);
	  acc_memcpy_to_device_async(a_4_1, b_3_2, buff_size, 8);
	  y_copy_to_4(-1, a_4_1, c_T4, i-1, block, width, bot+1, 3, 8);
	  acc_wait_all();
	}
      direction_5 +=  omp_get_wtime();

      direction_6 -=  omp_get_wtime();
      for(i = 1; i < _ny; i++)
	{
	  // compute halo_pack and inner
	  //	  cudaSetDevice(0);
	  function_y_1_down(c_T1, a_1_2, i, dy, DX, dz, 
			    DXZP, DXYP, DYZP, F, _azz, _axx, _ayy, width, 0, 0);
	  function_y_t_down(c_T1, i, dy, DX, dz, 
			    DXZP, DXYP, DYZP, F, _azz, _axx, _ayy, width, 0, 1);
	  acc_memcpy_from_device_async(b_1_2, a_1_2, buff_size, 0);


	  // compute halo_pack and inner
	  //	  cudaSetDevice(1);
	  function_y_1_down(c_T2, a_2_2, i, dy, DX, dz, 
			    DXZP, DXYP, DYZP, F, _azz, _axx, _ayy, width, 1, 2);
	  function_y_2_down(c_T2, a_2_1, i, dy, DX, dz, 
			    DXZP, DXYP, DYZP, F, _azz, _axx, _ayy, width, 1, 3);
	  function_y_down(c_T2, i, dy, DX, dz, 
			  DXZP, DXYP, DYZP, F, _azz, _axx, _ayy, width, 1, 4);
	  acc_memcpy_from_device_async(b_2_2, a_2_2, buff_size, 2);
	  acc_memcpy_from_device_async(b_2_1, a_2_1, buff_size, 3);

	  
	  // compute halo_pack and inner
	  //	  cudaSetDevice(2);
	  function_y_1_down(c_T3, a_3_2, i, dy, DX, dz, 
			    DXZP, DXYP, DYZP, F, _azz, _axx, _ayy, width, 2, 5);
	  function_y_2_down(c_T3, a_3_1, i, dy, DX, dz, 
			    DXZP, DXYP, DYZP, F, _azz, _axx, _ayy, width, 2, 6);
	  function_y_down(c_T3, i, dy, DX, dz, 
			  DXZP, DXYP, DYZP, F, _azz, _axx, _ayy, width, 2, 7);
	  acc_memcpy_from_device_async(b_3_2, a_3_2, buff_size, 5);
	  acc_memcpy_from_device_async(b_3_1, a_3_1, buff_size, 6);


	  // compute halo_pack and inner
	  //	  cudaSetDevice(3);
	  function_y_2_down(c_T4, a_4_1, i, dy, DX, dz, 
			    DXZP, DXYP, DYZP, F, _azz, _axx, _ayy, width, 3, 8);
	  function_y_b_down(c_T4, i, dy, DX, dz, 
			    DXZP, DXYP, DYZP, F, _azz, _axx, _ayy, width, 3, 9);
	  acc_memcpy_from_device_async(b_4_1, a_4_1, buff_size, 8);


	  acc_wait_async(0,3);
	  acc_memcpy_to_device_async(a_1_2, b_2_1, buff_size, 0);
	  y_copy_to_1(1, a_1_2, c_T1, i-1, block, width, bot+1, 0, 0);

	  acc_wait_async(3,0);
	  acc_memcpy_to_device_async(a_2_1, b_1_2, buff_size, 3);
	  y_copy_to_2(1, a_2_1, c_T2, i-1, block, width, bot+1, 1, 3);
	  acc_wait_async(2,6);
	  acc_memcpy_to_device_async(a_2_2, b_3_1, buff_size, 2);
	  y_copy_to_2_(1, a_2_2, c_T2, i-1, block, width, bot+1, 1, 2);

	  acc_wait_async(6,2);
	  acc_memcpy_to_device_async(a_3_1, b_2_2, buff_size, 6);
	  y_copy_to_3(1, a_3_1, c_T3, i-1, block, width, bot+1, 2, 6);
	  acc_wait_async(5,8);
	  acc_memcpy_to_device_async(a_3_2, b_4_1, buff_size, 5);
	  y_copy_to_3_(1, a_3_2, c_T3, i-1, block, width, bot+1, 2, 5);

	  acc_wait_async(8,5);
	  acc_memcpy_to_device_async(a_4_1, b_3_2, buff_size, 8);
	  y_copy_to_4(1, a_4_1, c_T4, i-1, block, width, bot+1, 3, 8);
	  acc_wait_all();
	}
      direction_6 +=  omp_get_wtime();

      //      data_timer_2 -=  omp_get_wtime();
      // copy GPU 1 data in to 1 buffers       
      //      cudaSetDevice(0);
      from_y_to_k_direction_T11(c_T1, c_T1_1, c_T1_2, 
				c_T1_3, c_T1_4, block, width, breadth, 0);
      // copy GPU 2 data in to 2 buffers       
      //      cudaSetDevice(1);
      from_y_to_k_direction_T21(c_T2, c_T2_1, c_T2_2, 
				c_T2_3, c_T2_4, block, width, breadth, 1);
      // copy GPU 3 data in to 3 buffers       
      //      cudaSetDevice(2);
      from_y_to_k_direction_T31(c_T3, c_T3_1, c_T3_2, 
				c_T3_3, c_T3_4, block, width, breadth, 2);
      // copy GPU 4 data in to 4 buffers       
      //      cudaSetDevice(3);
      from_y_to_k_direction_T41(c_T4, c_T4_1, c_T4_2, 
				c_T4_3, c_T4_4, block, width, breadth, 3);
      //      acc_wait_all();       
      
      // copied GPU data to CPU buffer
      //      cudaSetDevice(0);
      data_timer_2 -=  omp_get_wtime();
      acc_memcpy_from_device(d_1_2, c_T1_2, data_size);
      acc_memcpy_from_device(d_1_3, c_T1_3, data_size);
      acc_memcpy_from_device(d_1_4, c_T1_4, data_size);
      data_timer_2 +=  omp_get_wtime();
      
      // copied GPU data to CPU buffer
      //      cudaSetDevice(1);
      acc_memcpy_from_device(d_2_1, c_T2_1, data_size);
      acc_memcpy_from_device(d_2_3, c_T2_3, data_size);
      acc_memcpy_from_device(d_2_4, c_T2_4, data_size);
      
      // copied GPU data to CPU buffer
      //      cudaSetDevice(2);
      acc_memcpy_from_device(d_3_1, c_T3_1, data_size);
      acc_memcpy_from_device(d_3_2, c_T3_2, data_size);
      acc_memcpy_from_device(d_3_4, c_T3_4, data_size);
      
      // copied GPU data to CPU buffer
      //      cudaSetDevice(3);
      acc_memcpy_from_device(d_4_1, c_T4_1, data_size);
      acc_memcpy_from_device(d_4_2, c_T4_2, data_size);
      acc_memcpy_from_device(d_4_3, c_T4_3, data_size);     

      // copied data from CPU to GPU
      //      cudaSetDevice(0);
      //      acc_wait_async(0,1);
      acc_memcpy_to_device(c_T1_2, d_2_1, data_size);
      //      acc_wait_async(0,2);
      acc_memcpy_to_device(c_T1_3, d_3_1, data_size);
      //      acc_wait_async(0,3);
      acc_memcpy_to_device(c_T1_4, d_4_1, data_size);
      
      // copied data from CPU to GPU
      //      cudaSetDevice(1);
      //      acc_wait_async(1,0);
      acc_memcpy_to_device(c_T2_1, d_1_2, data_size);
      //      acc_wait_async(1,2);
      acc_memcpy_to_device(c_T2_3, d_3_2, data_size);
      //      acc_wait_async(1,3);
      acc_memcpy_to_device(c_T2_4, d_4_2, data_size);
      
      // copied data from CPU to GPU
      //      cudaSetDevice(2);
      //      acc_wait_async(2,0);
      acc_memcpy_to_device(c_T3_1, d_1_3, data_size);
      //      acc_wait_async(2,1);
      acc_memcpy_to_device(c_T3_2, d_2_3, data_size);
      //      acc_wait_async(2,3);
      acc_memcpy_to_device(c_T3_4, d_4_3, data_size);
      
      // copied data from CPU to GPU
      //      cudaSetDevice(3);
      //      acc_wait_async(3,0);
      acc_memcpy_to_device(c_T4_1, d_1_4, data_size);
      //      acc_wait_async(3,1);
      acc_memcpy_to_device(c_T4_2, d_2_4, data_size);
      //      acc_wait_async(3,2);
      acc_memcpy_to_device(c_T4_3, d_3_4, data_size);
      //      acc_wait_all();
      // mutilpe GPU biuffer in to 1 GPU
      //      cudaSetDevice(0);
      to_y_to_k_direction_T11(c_T1, c_T1_1, c_T1_2, 
			      c_T1_3, c_T1_4, block, width, breadth, 0);
      // mutilpe GPU biuffer in to 2 GPU
      //      cudaSetDevice(1);
      to_y_to_k_direction_T21(c_T2, c_T2_1, c_T2_2, 
			      c_T2_3, c_T2_4, block, width, breadth, 1);     
      // mutilpe GPU biuffer in to 3 GPU
      //      cudaSetDevice(2);
      to_y_to_k_direction_T31(c_T3, c_T3_1, c_T3_2, 
			      c_T3_3, c_T3_4, block, width, breadth, 2);
      // mutilpe GPU biuffer in to 4 GPU
      //      cudaSetDevice(3);
      to_y_to_k_direction_T41(c_T4, c_T4_1, c_T4_2, 
			      c_T4_3, c_T4_4, block, width, breadth, 3);
      //      acc_wait_all();
      //      data_timer_2 +=  omp_get_wtime();
    }// 8 sweeps
  
 
  // total time 
  compute_timer += omp_get_wtime();
  
  // copy back all the computed value to CPU form GPU  
  acc_memcpy_from_device(T1, c_T1, size);    
  acc_memcpy_from_device(T2, c_T2, size);    
  acc_memcpy_from_device(T3, c_T3, size);    
  acc_memcpy_from_device(T4, c_T4, size);
  
  // free the GPU buffers of boiundary layer values 
  // and partitioned data 
  acc_free(c_T1);  acc_free(a_1_2);
  acc_free(c_T2);  acc_free(a_2_1); acc_free(a_2_2);
  acc_free(c_T3);  acc_free(a_3_1); acc_free(a_3_2);
  acc_free(c_T4);  acc_free(a_4_1);

  // free the data transfer GPU buffer
  acc_free(c_T1_1); acc_free(c_T1_2); acc_free(c_T1_3); acc_free(c_T1_4);
  acc_free(c_T2_1); acc_free(c_T2_2); acc_free(c_T2_3); acc_free(c_T2_4);
  acc_free(c_T3_1); acc_free(c_T3_2); acc_free(c_T3_3); acc_free(c_T3_4);
  acc_free(c_T4_1); acc_free(c_T4_2); acc_free(c_T4_3); acc_free(c_T4_4);


  // partitioned CPU data into GPU
  for(i = 0 ; i < block; i++)
    {
      for(j = 0 ; j < width-1; j++)
	{
	  for(k = 0 ; k < block; k++)
	    {
	      T[i + block * (j + block * k)] =  T1[i + block * (j + width * k)];
	      T[i + block * (1+j + (_nx/4) + block * k)] = T2[i + block * (j+1 + width * k)];
	      T[i + block * (1+j + ((_nx/4)*2) + block * k)] = T3[i + block * (j+1 + width * k)];
	      T[i + block * (1+j + ((_nx/4)*3) + block * k)] = T4[i + block * (j+1 + width * k)];
	    }
	}
    }
  
  // free the CPU partined data memory and buffer memory 
  free(T1); free(b_1_2);
  free(T2); free(b_2_1); free(b_2_2);
  free(T3); free(b_3_1); free(b_3_2);
  free(T4); free(b_4_1);

  // data tansfer memory 
  free(d_1_2);free(d_1_3);free(d_1_4);
  free(d_2_1);free(d_2_3);free(d_2_4);
  free(d_3_1);free(d_3_2);free(d_3_4);
  free(d_4_1);free(d_4_2);free(d_4_3);

#ifdef PRI
  FILE *init;
  init = fopen("stream_an_iso_4_openacc.txt","w");
  for(i = 0; i < block; i++)
    {
      for(j = 0; j < block; j++)
	{
	  for(k = 0; k < block; k++)
	    {
	      fprintf(init,"%f\n",T[i + block * (j + block * k)]);
	    }
	}
    }
  
  fclose(init);
#endif 
  free(T); 
  printf("*******************************************************************************************\n");
  printf(" Initialised T, in CPU in------------------------------- %3f seconds.\n", end_initT-start_initT);
  printf(" Data partition time 1 --------------------------------- %f seconds.\n", data_timer_1);
  printf(" Data partition time 2 --------------------------------- %f seconds.\n", data_timer_2);
  printf(" Direction sweep 1     --------------------------------- %f seconds.\n", direction_1);
  printf(" Direction sweep 2     --------------------------------- %f seconds.\n", direction_2);
  printf(" Direction sweep 3     --------------------------------- %f seconds.\n", direction_3);
  printf(" Direction sweep 4     --------------------------------- %f seconds.\n", direction_4);
  printf(" Direction sweep 5     --------------------------------- %f seconds.\n", direction_5);
  printf(" Direction sweep 6     --------------------------------- %f seconds.\n", direction_6);  
  printf(" Created grid size:------------------------------------- %d x %d x %d\n", _nx, _ny, _nz);
  printf(" Model is ---------------------------------------------- AN_Isotropic \n");
  printf(" Total sweep is ---------------------------------------- %d \n", tot);
  printf(" Time taken for the 4 GPU is --------------------------- %f seconds\n",compute_timer);
 
#ifdef VTK_PRI
  Write_VTK_Structured_Grid(T, _nx, _ny, _nz, 1, "testIMP.vtk" );
  printf(" Exported full 3D results to VTK format.\n");
#endif

#ifdef UNROLLED
  printf(" UNROLLED defined \n");
#endif

#ifdef OPTTEST
  printf(" OPTTEST defined \n");
#endif
  
#ifdef PRI
  printf(" Solution is written in the file------------------------ stream_an_iso_4_openacc.txt \n");
#endif
  printf("*******************************************************************************************\n");
  return 0;
}






