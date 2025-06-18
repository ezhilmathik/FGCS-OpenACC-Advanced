//-*-c++-*-
/*
  ============================================================================
  Name        : 3DparallelMarching.c
  Author      : Tor Gillberg 
  Version     : N/A
  Copyright   : Your copyright notice
  Description : With OpenMP or mint
  mintTranslator Fold3dPMM.c
  code accessable on svn, at svn co https://svn.simula.no:40081/int/kalkulo/torgi
  nvcc -O3 mi.cu
  or for _DOUBLE_ precision
  nvcc -arch sm_20 -O3
  optional flag to prefer larger shared memory and smaller L1 cache (suggestion from Mohammed)
  --ptxas-options=-v
  ============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
//#define SIGNEDCOMP
#include <openacc.h>
#include <omp.h>
#include "Fold3dPMM.h"

//#define VTK_PRI // print the VTK solution
#define PRI // print the solution 

#ifndef fmin
#define fmin(a, b) (((a)<(b)) ? (a):(b))
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


////////////////////////////////////////////////////////////////////////////////
// FoldEdge2Points 
////////////////////////////////////////////////////////////////////////////////
#pragma acc routine(FoldEdge2Points) seq
void FoldEdge2Points(double* tnew, const double st, const double xt, 
		     const double F, const double ax, const double ay,
		     const double az, const double dxx, const double dzz)
{
  
#ifdef OPTTEST
  if((ax > 0.0 && xt > st)) 
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

      if(sqrp < *tnew && sqrp > fmin(st,xt)) 
	{
	  ga = sqrt((dtx * dtx + c*c)* F*F / ga);
	  
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

////////////////////////////////////////////////////////////////////////////////
// FoldSurf3Points
////////////////////////////////////////////////////////////////////////////////
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
  if(*tnew < fmin(st, fmin(xt,yt)) || (ax > 0.0 && xt > st) || (ay > 0.0 && yt > xt)) 
    {
      return;
    }
#endif
#ifndef UNROLLED
  if((ax >0.0 && xt > st) || (ay > 0.0 && yt > xt));
  {
    return;
  }
#endif
  double dtx = (xt-st) / dxx;
  double dty = (yt-xt) / dyy;
  double dtxdtysq = dtx*dtx + dty*dty;

  double ga = F*F - az*az;
  double al = 1.0 - ax*dtx - ay*dty;
  double sqrp = al*al - dtxdtysq*ga;
  
  if(sqrp > 0.0) 
    {
      al = (-az*al + F*sqrt(sqrp))/ga;
      ga = st + dzz*al;

      double xe,ye;

      if(ga < *tnew && ga > fmin(fmin(st, xt), yt)) 
	{
	  sqrp = sqrt(dtxdtysq + al*al);
	  dtxdtysq = -dzz / (F*al + az*sqrp);
	  xe = dtxdtysq*(F*dtx + ax*sqrp);
	  ye = dtxdtysq*(F*dty + ay*sqrp);
	  if(0.0 <= fmin(xe, ye) && ye*dxx <= xe*dyy && xe <= dxx) 
	    {
	      *tnew = ga;
	    }
	}
    }
  if(*tnew > fmin(xt, yt) && (!(ax*dzz + az*dxx > 0.0 && xt > yt))) 
    {
      FoldEdge2Points(tnew, xt, yt, F, ay, (ax*dzz + az*dxx)/dxz,
		      (-ax*dxx + az*dzz)/dxz, dyy, dxz);
    }
}

////////////////////////////////////////////////////////////////////////////////
// sweeping direction at x-left
////////////////////////////////////////////////////////////////////////////////
void function_x_down(double *T, int i,  
		     double dzz, double dxx, double dyy, 
		     double dxy, double dxz, double dyz, 
		     double F, double ay, double ax, double az)
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn;
  int j,k;
  int grid=ceil(_nx*(_nx/128.)); 
#pragma acc parallel deviceptr(T) num_gangs(grid) vector_length(128)
#pragma acc loop collapse(2) private (j,k,st,az,ay,ax,txm,txy,txnyn,xt,tym,yt,ynt,xnt)    
  for(k = 1; k < _ny+1; ++k)
    {
      for(j = 1; j < _nz+1; ++j)
  	{	 
	  tnew = fabs((T[(i+1) + HEIGHT * (j + HEIGHT * k)]));
	  st =  fabs((T[i + HEIGHT * (j + HEIGHT * k)]));
	  xt  = fabs((T[i + HEIGHT * (j + HEIGHT * (k+1))]));
	  yt  = fabs((T[i + HEIGHT * ((j+1) + HEIGHT * k)]));
	  xnt = fabs((T[i + HEIGHT * (j + HEIGHT * (k-1))]));
	  ynt = fabs((T[i + HEIGHT * ((j-1) + HEIGHT * k)]));
	  txm = fabs((T[i + HEIGHT * ((j-1) + HEIGHT * (k+1))]));
	  txy = fabs((T[i + HEIGHT * ((j+1) + HEIGHT * (k+1))]));
	  tym = fabs((T[i + HEIGHT * ((j+1) + HEIGHT * (k-1))]));
	  txnyn = fabs((T[i + HEIGHT * ((j-1) + HEIGHT * (k-1))]));
	  
	  ay = _ayy; ax = _azz; az = _axx; F=_Fc;
	  if (T[(i+1) + HEIGHT * (j + HEIGHT * k)] < 0)
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
		      tnew = fmin(tnew, st+dzz*dzz/sqrp);
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
				  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
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
	      FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, xnt))
		{
		  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
		}
	    }
	  if(tnew > fmin(st,ynt) || tnew > txnyn)
	    {
	      FoldSurf3Points(&tnew, st, ynt,txnyn,F, -ay, -ax, az, dyy, dxx, dzz, dyz);
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
	      if(tnew > fmin(st, txm)) 
		{
		  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
				  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
		}
	    }
#endif
  
	  T[(i+1) + HEIGHT * (j + HEIGHT * k)] = (sign(T[(i+1) + HEIGHT * (j + HEIGHT * k)]) * 1.0) * tnew; 
	} // IF
    }
}

////////////////////////////////////////////////////////////////////////////////
// sweeping direction at x-right
////////////////////////////////////////////////////////////////////////////////
void function_x_up(double *T, int i,  
		   double dzz, double dxx, double dyy, 
		   double dxy, double dxz, double dyz, 
		   double F, double ay, double ax, double az)
{
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  int j,k;
  int grid=ceil(_nx*(_nx/128.)); 
#pragma acc parallel deviceptr(T) num_gangs(grid) vector_length(128)
#pragma acc loop collapse(2) private (j,k, tnew,st,az,ay,ax,txm,txy,txnyn,xt,tym,yt,ynt,xnt)
  for(j = 1; j < _ny+1; ++j)
    {
      for(k = 1; k < _nz+1; ++k)
  	{
	  tnew = fabs((T[(i-1) + HEIGHT * (j + HEIGHT * k)]));
	  st =  fabs((T[i + HEIGHT * (j + HEIGHT * k)]));
	  xt  = fabs((T[i + HEIGHT * (j + HEIGHT * (k-1))]));
	  yt  = fabs((T[i + HEIGHT * ((j-1) + HEIGHT * k)]));
	  xnt = fabs((T[i + HEIGHT * (j + HEIGHT * (k+1))]));
	  ynt = fabs((T[i + HEIGHT * ((j+1) + HEIGHT * k)]));
	  txm = fabs((T[i + HEIGHT * ((j+1) + HEIGHT * (k-1))]));
	  txy = fabs((T[i + HEIGHT * ((j-1) + HEIGHT * (k-1))]));
	  tym = fabs((T[i + HEIGHT * ((j-1) + HEIGHT* (k+1))]));
	  txnyn = fabs((T[i + HEIGHT * ((j+1) + HEIGHT * (k+1))]));

	  ay = - _ayy; 	ax = - _azz;   	az = - _axx; F=_Fc;
	  if (T[(i-1) + HEIGHT * (j + HEIGHT * k)] < 0)
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
		      tnew = fmin(tnew,st+dzz*dzz/sqrp);
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
				  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
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
				  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
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
				  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
		}
	    }
#endif

	  T[(i-1) + HEIGHT* (j + HEIGHT * k)] = (sign(T[(i-1) + HEIGHT * (j + HEIGHT * k)]) * 1.0) * tnew;
	} // IF
    }
}


////////////////////////////////////////////////////////////////////////////////
// sweeping direction at y-right
////////////////////////////////////////////////////////////////////////////////
void function_y_up(double *T, int j,  
		   double dzz, double dxx, double dyy, 
		   double dxy, double dxz, double dyz, 
		   double F, double ay, double ax, double az)
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  int i,k;
  int grid=ceil(_nx*(_nx/128.)); 
#pragma acc parallel deviceptr(T) num_gangs(grid) vector_length(128)
#pragma acc loop collapse(2) private (i,k, tnew,st,az,ay,ax,txm,txy,txnyn,xt,tym,yt,ynt,xnt)  
  for(i = 1; i < _nx+1; ++i)
    {
      for(k = 1; k < _nz+1; ++k)
 	{
	  tnew = fabs((T[i+HEIGHT*((j-1)+HEIGHT*k)]));
	  st =  fabs((T[i + HEIGHT * (j + HEIGHT * k)]));
	  xt  = fabs((T[(i-1) + HEIGHT * (j + HEIGHT * k)]));
	  yt  = fabs((T[i + HEIGHT * (j + HEIGHT * (k-1))]));
	  xnt = fabs((T[(i+1) + HEIGHT * (j + HEIGHT * k)]));
	  ynt = fabs((T[i + HEIGHT * (j + HEIGHT * (k+1))]));
	  txm = fabs((T[(i-1)+ HEIGHT * (j + HEIGHT * (k+1))]));
	  txy = fabs((T[(i-1) + HEIGHT * (j + HEIGHT * (k-1))]));
	  tym = fabs((T[(i+1) + HEIGHT * (j + HEIGHT * (k-1))]));
	  txnyn = fabs((T[(i+1) + HEIGHT * (j + HEIGHT * (k+1))]));

	  ay = - _azz; ax = - _axx; az = - _ayy;  	F=_Fc;
	  if (T[i+HEIGHT*((j-1)+HEIGHT*k)] < 0)
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
		  sqrp = az * dzz + sqrt(sqrp);
		  if(sqrp > 0.0) 
		    {
		      tnew = fmin(tnew, st+dzz*dzz/sqrp);
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
				  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
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
				  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
		}
	    }
	  if(tnew > fmin(st, ynt) || tnew > txm)
	    {
	      FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, ynt)) 
		{
		  FoldEdge2Points(&tnew, st, ynt, F, -ay, ax, az, dyy, dzz);
		}
	      if(tnew > fmin(st,txm))
		{
		  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
				  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
		}
	    }
  
#endif
  
	  T[i + HEIGHT * ((j-1) + HEIGHT * k)]=(sign(T[i + HEIGHT * ((j-1) + HEIGHT * k)])*1.0)*tnew;
	} // IF
    }
}

////////////////////////////////////////////////////////////////////////////////
// sweeping direction at y-left
////////////////////////////////////////////////////////////////////////////////
void function_y_down(double *T, int j,  
		     double dzz, double dxx, double dyy, 
		     double dxy, double dxz, double dyz, 
		     double F, double ay, double ax, double az)
  
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn;
  int i,k;
  int grid=ceil(_nx*(_nx/128.)); 
#pragma acc parallel deviceptr(T) num_gangs(grid) vector_length(128)
#pragma acc loop collapse(2) private (i,k, tnew,st,az,ay,ax,txm,txy,txnyn,xt,tym,yt,ynt,xnt)    
  for(i = 1; i < _nx+1; ++i)
    {
      for(k = 1; k < _nz+1; ++k)
 	{
	  tnew = fabs((T[i + HEIGHT * ((j+1) + HEIGHT * k)]));
	  st = fabs((T[i + HEIGHT * (j + HEIGHT * k)]));
	  xt  = fabs((T[(i+1) + HEIGHT * (j + HEIGHT * k)]));
	  yt  = fabs((T[i + HEIGHT * (j + HEIGHT * (k+1))]));
	  xnt = fabs((T[(i-1) + HEIGHT * (j + HEIGHT * k)]));
	  ynt = fabs((T[i + HEIGHT * (j + HEIGHT * (k-1))]));
	  txm = fabs((T[(i+1) + HEIGHT * (j + HEIGHT * (k-1))]));
	  txy = fabs((T[(i+1) + HEIGHT * (j + HEIGHT * (k+1))]));
	  tym = fabs((T[(i-1) + HEIGHT * (j + HEIGHT * (k+1))]));
	  txnyn = fabs((T[(i-1) + HEIGHT * (j + HEIGHT * (k-1))]));

	  ay = _azz; ax = _axx; az = _ayy;  	F=_Fc;
	  if (T[i + HEIGHT * ((j+1) + HEIGHT * k)] < 0)
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
		      tnew = fmin(tnew, st+dzz*dzz/sqrp);
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
				  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
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
	  if(tnew > fmin(st,ynt) || tnew > txnyn)
	    {
	      FoldSurf3Points(&tnew, st, ynt,txnyn,F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st,txnyn)) 
		{
		  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
				  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
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
				  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
		}
	    }
  
#endif

	  T[i + HEIGHT * ((j+1) + HEIGHT * k)]=(sign(T[i + HEIGHT * ((j+1) + HEIGHT * k)])*1.0)*tnew;
	}// IF  
    }
}


////////////////////////////////////////////////////////////////////////////////
// sweeping direction at z-right
////////////////////////////////////////////////////////////////////////////////
void function_k_up(double *T, int k,  
		   double dzz, double dxx, double dyy, 
		   double dxy, double dxz, double dyz, 
		   double F, double ay, double ax, double az)
  
{ 
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  int i,j;
  int grid=ceil(_nx*(_nx/128.)); 
#pragma acc parallel deviceptr(T) num_gangs(grid) vector_length(128)
#pragma acc loop collapse(2) private (i,j, tnew,st,az,ay,ax,txm,txy,txnyn,xt,tym,yt,ynt,xnt)      
  for(j = 1; j < _ny+1; ++j)
    {
      for(i = 1; i < _nx+1; ++i)
	{	     
	  tnew = fabs((T[i + HEIGHT * (j + HEIGHT * (k+1))]));
	  st = fabs((T[i + HEIGHT * (j + HEIGHT * k)]));
	  xt  = fabs((T[i + HEIGHT * ((j+1) + HEIGHT * k)]));
	  yt  = fabs((T[(i+1) + HEIGHT * (j + HEIGHT * k)]));
	  xnt = fabs((T[i + HEIGHT * ((j-1) + HEIGHT * k)]));
	  ynt = fabs((T[(i-1) + HEIGHT * (j + HEIGHT * k)]));
	  txm = fabs((T[(i-1) + HEIGHT * ((j+1) + HEIGHT * k)]));
	  txy = fabs((T[(i+1) + HEIGHT * ((j+1) + HEIGHT * k)]));
	  tym = fabs((T[(i+1) + HEIGHT * ((j-1) + HEIGHT * k)]));
	  txnyn = fabs((T[(i-1) + HEIGHT * ((j-1) + HEIGHT * k)]));

	  ay = _axx; ax = _ayy; az = _azz; F=_Fc;
	  if (T[i + HEIGHT * (j + HEIGHT * (k+1))] < 0)
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
		      tnew = fmin(tnew, st+dzz*dzz/sqrp);
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
	      if(tnew > fmin(st,xt))
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
				  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
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
	      FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, xnt))
		{
		  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
		}
	    }
	  if(tnew > fmin(st ,ynt) || tnew > txnyn)
	    {
	      FoldSurf3Points(&tnew, st, ynt, txnyn, F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, txnyn)) 
		{
		  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
				  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
		}
	    }
	  if(tnew > fmin(st, ynt) || tnew > txm)
	    {
	      FoldSurf3Points(&tnew, st, ynt, txm, F, -ay,  ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, ynt)) 
		{
		  FoldEdge2Points(&tnew,st, ynt, F, -ay, ax, az, dyy, dzz);
		}
	      if(tnew > fmin(st, txm)) 
		{
		  FoldEdge2Points(&tnew, st, txm, F,  (ax*dxx - ay*dyy)/dxy, 
				  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
		}
	    }
    
#endif
	  
	  T[i + HEIGHT * (j + HEIGHT * (k+1))] = (1.0*sign(T[i + HEIGHT * (j + HEIGHT * (k+1))]))*tnew;
	} // IF
    }
}


////////////////////////////////////////////////////////////////////////////////
// sweeping direction at z-left
////////////////////////////////////////////////////////////////////////////////
void function_k_down(double *T, int k,  
		     double dzz, double dxx, double dyy, 
		     double dxy, double dxz, double dyz, 
		     double F, double ay, double ax, double az)
  
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn;
  int i,j;
  int grid=ceil(_nx*(_nx/128.)); 
#pragma acc parallel deviceptr(T) num_gangs(grid) vector_length(128)
#pragma acc loop collapse(2) private (i,j, tnew,st,az,ay,ax,txm,txy,txnyn,xt,tym,yt,ynt,xnt)        
  for(int j = 1; j < _ny+1; ++j)
    {
      for(int i = 1; i < _nx+1; ++i)
	{
	  tnew = fabs((T[i + HEIGHT * (j + HEIGHT * (k-1))]));
	  st = fabs((T[i + HEIGHT * (j + HEIGHT * k)]));
	  xt  = fabs((T[i + HEIGHT * ((j-1) + HEIGHT * k)]));
	  yt  = fabs((T[(i-1) + HEIGHT * (j + HEIGHT * k)]));
	  xnt = fabs((T[i + HEIGHT * ((j+1) + HEIGHT * k)]));
	  ynt = fabs((T[(i+1) + HEIGHT * (j + HEIGHT * k)]));
	  txm = fabs((T[(i+1) + HEIGHT * ((j-1) + HEIGHT * k)]));
	  txy = fabs((T[(i-1) + HEIGHT * ((j-1) + HEIGHT * k)]));
	  tym = fabs((T[(i-1) + HEIGHT * ((j+1) + HEIGHT * k)]));
	  txnyn = fabs((T[(i+1) + HEIGHT * ((j+1) + HEIGHT * k)]));

	  ay = -_axx; ax = -_ayy; az = -_azz; F=_Fc;
	  if (T[i + HEIGHT * (j + HEIGHT * (k-1))] < 0)
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
		      tnew = fmin(tnew, st+dzz*dzz/sqrp);
		    }
		}
	    }
#ifdef UNROLLED
	  if(!(tnew < st && tnew < fmin(xt, yt) && tnew < fmin(txy,txm) && tnew < fmin(txnyn,tym)))
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
				  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
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
	      FoldSurf3Points(&tnew, st, xnt, txnyn, F, -ax, -ay, az, dxx, dyy, dzz, dxz);
	      if(tnew > fmin(st, xnt))
		{
		  FoldEdge2Points(&tnew, st, xnt, F, -ax, ay, az, dxx, dzz);
		}
	    }
	  if(tnew > fmin(st, ynt) || tnew > txnyn)
	    {
	      FoldSurf3Points(&tnew, st, ynt,txnyn,F, -ay, -ax, az, dyy, dxx, dzz, dyz);
	      if(tnew > fmin(st, txnyn)) 
		{
		  FoldEdge2Points(&tnew, st, txnyn, F, -(ax*dxx + ay*dyy)/dxy, 
				  (-ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
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
				  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
		}
	    }
  
#endif
  
	  T[i + HEIGHT * (j + HEIGHT * (k-1))] = (1.0*sign(T[i + HEIGHT * (j + HEIGHT * (k-1))]))*tnew;
	} // IF
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
	      //	      fprintf(out,"%f\n",T[i+(_nx+2)*(j+(_ny+2)*k)]);
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
  
  int i,j,k;
  _DOUBLE_ DX = ( _xmax - _xmin) / ( _nx - 1.0);
  _DOUBLE_ dy = ( _ymax - _ymin) / ( _ny - 1.0);
  _DOUBLE_ dz = ( _zmax - _zmin) / ( _nz - 1.0);
  
  _DOUBLE_ DXYP = sqrt(DX*DX + dy*dy);
  _DOUBLE_ DXZP = sqrt(DX*DX + dz*dz);
  _DOUBLE_ DYZP = sqrt(dy*dy + dz*dz);
  _DOUBLE_ DXYZP = sqrt(DX*DX + dy*dy + dz*dz);
  
  // length of the 3D array size in 1-direction + ghost points 
  int block = _nx + 2;
  
  // memory allocation for the calculation 1D- > 3D
  _DOUBLE_*T = (_DOUBLE_*)malloc(sizeof(_DOUBLE_)* block * block * block);
  
  // verify that allocations succeeded
  if (T == NULL )
    {
      fprintf(stderr, "Failed to allocate host vectors!\n");
      exit(EXIT_FAILURE);
    }
  const double F = _Fc;
  
  printf(" Initialization is started\n");
  double start_initT = wtime();
  
  ImplicitInitialiser(T, _nx, _ny, _nz, DX, dy, dz);	
  
  double end_initT = wtime();
  printf(" Initialization is finished\n");
  _DOUBLE_ *restrict d_T = NULL;
  d_T=acc_malloc((size_t)sizeof(_DOUBLE_)* block * block * block);
  acc_memcpy_to_device(d_T, T, sizeof(_DOUBLE_)* block * block * block);
  int sweep,tot = 8;
  //#pragma acc data copy(T[0:((_nx+2)*(_ny+2)*(_nz+2))])
  //  {
  
  double start_comp = wtime();
  for(sweep = 0; sweep < tot; sweep++)
    {
      printf(" sweep no is %d \n",sweep+1);
      for(i = 1; i < _nx ; i++)
	{
	  
	  function_x_down(d_T, i, DX, dz, dy, 
			  DYZP, DXZP,DXYP, F, _ayy, _azz, _axx);
	}
      
      for(i = _nx; i > 1; --i)    
	{
	  
	  function_x_up(d_T, i, DX, dz, dy, 
			DYZP, DXZP, DXYP, F, -_ayy, -_azz, -_axx);	  
	}
      
      for(i = 1; i < _nz; i++)
	{
	  
	  function_k_up(d_T, i, dz, dy, DX, 
			DXYP, DYZP, DXZP, F, _axx, _ayy, _azz);	  
	}
      
      for(i  = _nz; i > 1; i--)
	{
	  
	  function_k_down(d_T, i, dz, dy, DX, 
			  DXYP, DYZP, DXZP, F, -_axx, -_ayy, -_azz);	  
	}
      
      
      for(i = _ny; i > 1; i--)
	{
	  
	  function_y_up(d_T, i, dy, DX, dz, 
			DXZP, DXYP, DYZP, F, -_azz, -_axx, -_ayy); 
	}

      for(i = 1; i < _ny; i++)
	{
	  
	  function_y_down(d_T, i, dy, DX, dz, 
			  DXZP, DXYP, DYZP, F, _azz, _axx, _ayy);	  
	}
    }
    //  }
  double end_comp = wtime();
  
  acc_memcpy_from_device(T, d_T, sizeof(_DOUBLE_)* block * block * block);
  
  // print the solution to the txt file for the error checking   
#ifdef PRI
  FILE *init;
  init = fopen("an_iso_1_openacc.txt","w");
  
  for(i = 0; i < block; i++)
    {
      for(j = 0; j < block; j++)
	{
	  for(k = 0; k < block; k++)
	    {
	      fprintf(init,"%f\n",T[i + HEIGHT * (j + HEIGHT * k)]);
	    }
	}
    }
  
  fclose(init);
#endif 
 
  
  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  // printing the programme results and timings 
  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  printf("*******************************************************************************************\n");
  printf(" Initialised T, in CPU  -------------------------------- %3f seconds.\n", end_initT-start_initT);
  printf(" Created grid size: %d x %d x %d\n", _nx, _ny, _nz);
  printf(" Model is ---------------------------------------------- AN_Isotropic \n");
  printf(" Total sweep is ---------------------------------------- %d \n", tot);
  printf(" Time taken for the 1 GPU is --------------------------- %f seconds\n",end_comp-start_comp);
  
#ifdef VTK_PRI
  Write_VTK_Structured_Grid(T, _nx, _ny, _nz, 1, "testIMP.vtk" );
  printf(" Exported full 3D results to VTK format.\n");
#endif
  
  // free the CPU memory 
  free(T); 
  
#ifdef UNROLLED
  printf(" UNROLLED defined \n");
#endif
  
#ifdef OPTTEST
  printf(" OPTTEST defined \n");
#endif
  
#ifdef PRI
  printf(" Solution is written in the file------------------------ an_iso_1_openacc.txt \n");
#endif
  printf("*******************************************************************************************\n");
  
  return 0;
}






