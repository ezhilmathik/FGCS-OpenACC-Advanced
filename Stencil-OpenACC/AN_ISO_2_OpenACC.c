//-*-c++-*-
/*
  ============================================================================
  Name        : 3DparallelMarching.c
  Author      : Tor Gillberg           - serial c code
  Author      : Ezhilmathi Krishnasamy - parallel(CUDA) version for 2-GPU 
  model       : basic GPU model (no streams, OMP and p2p)
  Description : Fold3DPMM.c ( has initialization implementaiton functions)
  Description : Fold3DPMM.h ( has input parameter for the grid size and etc,.)
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

//#define VTK_PRI // print the VTK solution
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
void FoldEdge2Points(_DOUBLE_* tnew, const _DOUBLE_ st, const _DOUBLE_ xt, 
		     const _DOUBLE_ F, const _DOUBLE_ ax, const _DOUBLE_ ay, 
		     const _DOUBLE_ az, const _DOUBLE_ dxx, const _DOUBLE_ dzz)
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
  
  _DOUBLE_ ga = F*F - ay*ay; 
  _DOUBLE_ dtx = (xt - st) / dxx;
  _DOUBLE_ c = (1.0 - dtx*ax);
  _DOUBLE_ sqrp = ga * (c*c - dtx * dtx * (ga - az*az));

  if(0.0 < sqrp)
    {
      c = (-az*c + sqrt(sqrp)) / (ga - az*az);
      sqrp = st + dzz*c;
      if(sqrp < *tnew && sqrp > fmin(st, xt)) 
	{
	  ga = sqrt((dtx * dtx + c*c)* F*F / ga);
	  _DOUBLE_ xe = -dzz * (F * dtx+ax * ga) / (F*c + az*ga);
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
	  c = (c + sqrt(sqrp));
	  if (c > 0.0) 
	    {
	      *tnew = fmin(*tnew, xt + ga/c);
	    }
	}
    }
}



#pragma acc routine(FoldSurf3Points) seq
void FoldSurf3Points(_DOUBLE_* tnew, const _DOUBLE_ st, 
		     const _DOUBLE_ xt, const _DOUBLE_ yt, const _DOUBLE_ F, 
		     const _DOUBLE_ ax, const _DOUBLE_ ay, const _DOUBLE_ az, 
		     const _DOUBLE_ dxx, const _DOUBLE_ dyy, const _DOUBLE_ dzz, 
		     const _DOUBLE_ dxz)
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
  //#ifndef UNROLLED
  //	if((ax>0. && xt>st) || (ay>0. && yt>xt)) {return;}// -1;}//use one node from st//TEMP REMOVED
  //#endif
  _DOUBLE_ dtx = (xt - st)/ dxx;
  _DOUBLE_ dty = (yt - xt)/ dyy;
  _DOUBLE_ dtxdtysq = dtx*dtx + dty*dty;
  _DOUBLE_ ga = F*F - az*az;
  _DOUBLE_ al = 1.0 - ax*dtx - ay*dty;
  _DOUBLE_ sqrp = al*al - dtxdtysq*ga;
  
  if(sqrp > 0.0) 
    {
      al = (-az*al + F*sqrt(sqrp)) / ga;
      ga = st + dzz*al;

      _DOUBLE_ xe, ye;

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
      FoldEdge2Points(tnew, xt, yt, F, ay, (ax*dzz + az*dxx)/dxz,
		      (-ax*dxx + az*dzz) / dxz, dyy, dxz);
    }
}



void function_x_down(double *T, int i, 
		     _DOUBLE_ dzz, _DOUBLE_ dxx, _DOUBLE_ dyy, 
		     _DOUBLE_ dxy, _DOUBLE_ dxz, _DOUBLE_ dyz, 
		     double F, double ay, double ax, double az,int WIDTH, int ID)
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  int j,k;
  
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T) 
#pragma acc loop collapse(2) private (j,k, ay, ax, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
  for(j = 1; j < WIDTH-1; ++j)
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
	      _DOUBLE_ sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;
	      
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
	      if(tnew>fmin(st, xt)) 
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
	      if(tnew>fmin(st, tym)) 
		{
		  FoldEdge2Points(&tnew, st, tym, F, -(ax*dxx - ay*dyy)/dxy, 
				  (ax*dyy + ay*dxx)/dxy, az, dxy, dzz);
		}
	    }
	  if(tnew > fmin(st,xnt) || tnew > txnyn)
	    {
	      FoldSurf3Points(&tnew, st, xnt, txnyn,F, -ax, -ay, az, dxx, dyy, dzz, dxz);
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
	  
	  T[(i+1) + HEIGHT * (j + WIDTH * k)] = (sign(T[(i+1) + HEIGHT * (j + WIDTH * k)])*1.0)*tnew; 
	} // IF
    }
}


void function_x_up(double *T, int i, 
		   _DOUBLE_ dzz, _DOUBLE_ dxx, _DOUBLE_ dyy, 
		   _DOUBLE_ dxy, _DOUBLE_ dxz, _DOUBLE_ dyz, 
		   double F, double ay, double ax, double az, int WIDTH, int ID)
{
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn;
  int j,k;
  
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T) 
#pragma acc loop collapse(2) private (j,k, ay, ax, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
  for(j = 1; j < WIDTH-1; ++j)
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
	      _DOUBLE_ sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;

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

	  T[(i-1) + HEIGHT* (j + WIDTH * k)] = (sign(T[(i-1) + HEIGHT * (j + WIDTH * k)]) * 1.0) * tnew;
	} // IF
    }
}



void function_y_up(double *T, int j, 
		   _DOUBLE_ dzz, _DOUBLE_ dxx, _DOUBLE_ dyy, 
		   _DOUBLE_ dxy, _DOUBLE_ dxz, _DOUBLE_ dyz, 
		   double F, double ay, double ax, double az, int WIDTH, int ID)
  
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  int i,k;
  
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T)
#pragma acc loop collapse(2) private (i,k, ay, ax, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
  for(i = 1; i < WIDTH-1; ++i)
    {
      for(k = 1; k < HEIGHT-1; ++k)
  	{
	  tnew = fabs((T[i+ WIDTH *((j-1)+HEIGHT*k)]));
	  st =  fabs((T[i +WIDTH * (j + HEIGHT * k)]));
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
	      _DOUBLE_ sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;
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
	  if(tnew > fmin(st ,xnt) || tnew > tym)
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
	  T[i + WIDTH * ((j-1) + HEIGHT * k)] = (sign(T[i + WIDTH * ((j-1) + HEIGHT * k)]) * 1.0)*tnew;
	} // IF
    }
}




void function_y_down(double *T, int j, 
		     _DOUBLE_ dzz, _DOUBLE_ dxx, _DOUBLE_ dyy, 
		     _DOUBLE_ dxy, _DOUBLE_ dxz, _DOUBLE_ dyz, 
		     double F, double ay, double ax, double az, int WIDTH, int ID)//, int stream)  
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn;
  int i,k;
  
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T) //async(stream) 
#pragma acc loop collapse(2) private (i,k, ay, ax, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
  for(i = 1; i < WIDTH-1; ++i)
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
	      _DOUBLE_ sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;
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
  
	  T[i + WIDTH * ((j+1) + HEIGHT * k)] = (sign(T[i + WIDTH * ((j+1) + HEIGHT * k)])*1.0)*tnew;
	} // IF
    }
}


void function_z_up(double *T, int k, 
		   _DOUBLE_ dzz, _DOUBLE_ dxx, _DOUBLE_ dyy, 
		   _DOUBLE_ dxy, _DOUBLE_ dxz, _DOUBLE_ dyz, 
		   double F, double ay, double ax, double az, int WIDTH, int ID)
  
{

  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn;
  int i, j;
  
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T) 
#pragma acc loop collapse(2) private (i,j, ay, ax, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
  for(j = 1; j < WIDTH-1; ++j)
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
	      _DOUBLE_ sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;
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
	      FoldSurf3Points(&tnew, st, xnt, txnyn,F, -ax, -ay, az, dxx, dyy, dzz, dxz);
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
  
	  T[i + HEIGHT * (j + WIDTH * (k+1))] = (1.0*sign(T[i + HEIGHT * (j + WIDTH * (k+1))]))*tnew;
	}// IF
    }
}


void function_z_down(double *T, int k, 
		     _DOUBLE_ dzz, _DOUBLE_ dxx, _DOUBLE_ dyy, 
		     _DOUBLE_ dxy, _DOUBLE_ dxz, _DOUBLE_ dyz, 
		     double F, double ay, double ax, double az, int WIDTH, int ID)
  
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn;
  int i,j;
  
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T) 
#pragma acc loop collapse(2) private (j,i, ay, ax, az, tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn)
  for(j = 1; j < WIDTH-1; ++j)
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
	  txnyn =fabs((T[(i+1) + HEIGHT * ((j+1) + WIDTH * k)]));
  
	  ay = -_axx; ax = -_ayy; az = -_azz;
	  if (T[i + HEIGHT * (j + WIDTH * (k-1))] < 0)
	    {
	      ax *= -1; 
	      ay *= -1; 
	      az *= -1;
	    }
  
	  if(st < tnew)
	    {
	      _DOUBLE_ sqrp = (F*F - (ax*ax + ay*ay)) * dzz*dzz;
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
	  T[i + HEIGHT * (j + WIDTH * (k-1))] = (1.0*sign(T[i + HEIGHT * (j + WIDTH * (k-1))]))*tnew;
	} // IF
    }
}



//****************************//
// boudary layer exchange     //
// between GPU's              //
//****************************//

// Save top bounday layer from T1
void z_copy_from_1(int dk, double *a_1_2, double* c_T1, 
		   int k, int block, int WIDTH, int bot, int ID)
{
  int i;
  
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_1_2, c_T1) 
#pragma acc loop private (i)
  for(i = 0; i < block-2; ++i)
    {
      a_1_2[i] = c_T1[(i+1) + block* (bot + WIDTH * (k+dk+1))];
    }
}


// Save top bounday layer from T2
void z_copy_from_2(int dk, double *a_2_1, double* c_T2, 
		   int k, int block, int WIDTH, int top, int bot, int ID)
{  
  int i;
  
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_2_1, c_T2) 
#pragma acc loop private (i)
  for(i = 0; i < block-2; ++i)
    {
      a_2_1[i] = c_T2[(i+1) + block * (top + WIDTH * (k+dk+1))]; 
    }
}


// New bounday layer for top of T1
void z_copy_to_1(int dk, double* a_2_1, double *c_T1, 
		 int k, int block, int WIDTH, int bot, int ID)
{
  int i;

  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_2_1, c_T1) 
#pragma acc loop private (i)
  for(i = 0; i < block-2; ++i)
    {
      c_T1[(i+1) + block * (bot + WIDTH * (k+dk+1))]  =  a_2_1[i];
    }
}


// New bounday layer for top of T2
void z_copy_to_2(int dk, double* a_1_2, double *c_T2, 
		 int k, int block, int WIDTH, int bot, int ID)
{ 
  int i;
  
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_1_2, c_T2) 
#pragma acc loop private (i)
  for(i = 0; i < block-2; ++i)
    {
      c_T2[(i+1) + block * (0 + WIDTH * (k+dk+1))]  = a_1_2[i];
    } 
}



// Save top bounday layer from T1
void x_copy_from_1(int dk, double *a_1_2, double* c_T1, 
		   int i, int block, int WIDTH, int bot, int ID)
{
  int k;
  
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_1_2, c_T1) 
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)
    {
      a_1_2[k] = c_T1[(i+1+dk) + block * (bot + WIDTH * (k+1))] ;
    }
}


// Save top bounday layer from T2
void x_copy_from_2(int dk, double *a_2_1, double* c_T2, 
		   int i, int block, int WIDTH, int top, int bot, int ID)
{
int k;

 acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_2_1, c_T2) //async(stream) 
#pragma acc loop private (k)
 for(k = 0; k < block-2; ++k)
   {
     a_2_1[k] = c_T2[(i+1+dk) + block * (top + WIDTH * (k+1))] ;
   }
}


// New bounday layer for top of T1
void x_copy_to_1(int dk, double* a_2_1, double *c_T1, 
		 int i, int block, int WIDTH, int bot, int ID)
{  
  int k;
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_2_1, c_T1)
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)
    {
      c_T1[(i+1+dk) + block * (bot + WIDTH * (k+1))]  =  a_2_1[k];
    }  
}


// New bounday layer for top of T2
void x_copy_to_2(int dk, double* a_1_2, double *c_T2, 
		 int i, int block, int WIDTH, int bot, int ID)
{ 
  int k;
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_1_2, c_T2)
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)
    {
      c_T2[(i+1+dk) + block * (0 + WIDTH * (k+1))]  = a_1_2[k];
    }
}


// Save bottom bounday layer from T1
void y_copy_from_1(int dk, double *a_1_2, double* c_T1, 
		   int j, int block, int WIDTH, int bot, int ID)
{
  int k;
  
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_1_2, c_T1) 
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)
    {
      a_1_2[k] = c_T1[bot + WIDTH * ((j+1+dk) + block * (k+1))] ; 
    } 
}


// Save top bounday layer from T2
void y_copy_from_2(int dk, double *a_2_1, double* c_T2, 
		   int j, int block, int WIDTH, int top, int bot, int ID)
{ 
  int k;

  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_2_1, c_T2)
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)
    {
      a_2_1[k] = c_T2[top + WIDTH * ((j+1+dk) + block * (k+1))] ;
    } 
}


// New bounday layer for bottom of T1
void y_copy_to_1(int dk, double* a_2_1, double *c_T1, 
		 int j, int block, int WIDTH, int bot, int ID)
{ 
  int k;

  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_2_1, c_T1)
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)
    {
      c_T1[bot + WIDTH * ((j+1+dk) + block * (k+1))]  =  a_2_1[k];
    }
}


// New bounday layer for top of T2
void y_copy_to_2(int dk, double* a_1_2, double *c_T2, int j,
		 int block, int WIDTH, int bot, int ID)
{
  int k;
  
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(a_1_2, c_T2) //async(stream) 
#pragma acc loop private (k)
  for(k = 0; k < block-2; ++k)
    {
      c_T2[0 + WIDTH * ((j+1+dk) + block * (k+1))]  = a_1_2[k]; 
    }
}



//************************************//
// copy the partined data to          //
// Y- direction for the 3-direction   //
// sweep; from X or Z - direction     //    
// partions                           //
//************************************//  

// copy T1 from Z -direction  
void from_k_to_y_direction_T1(_DOUBLE_ *T1, _DOUBLE_ *T1_1, _DOUBLE_ *T1_2, 
			      int block, int WIDTH, int length, int ID)
{
  int i, j, k;
  
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T1, T1_1, T1_2)
#pragma acc loop collapse(3) private(i, j, k)    
  for(i = 0; i < WIDTH; ++i){
    for(j = 0; j < WIDTH; ++j){
      for(k = 0; k < block; ++k){  
	if(j < length)
	  {
	    T1_1[i + WIDTH * (j+WIDTH*k)] = T1[i + block * (j+WIDTH*k)];
	    T1_2[i + WIDTH * (j+length*k)] = T1[i + length + block * (j+WIDTH*k)];      
	  }
	else
	  {  
	    T1_1[i + WIDTH * (j+WIDTH*k)] = T1[i + block * (j+WIDTH*k)];
	  }
      } // if
    }
  }
}


// copy T2 from Z-direction
void from_k_to_y_direction_T2(const _DOUBLE_* T2,_DOUBLE_ *T2_1, _DOUBLE_ *T2_2, 
			      int block, int WIDTH, int length, int ID)
{
  
  int i, j, k;
  
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T2, T2_1, T2_2) 
#pragma acc loop collapse(3) private(i, j, k)  
  for(i = 0; i < WIDTH; ++i){
    for(j = 0; j < WIDTH; ++j){
      for(k = 0; k < block; ++k){  
	if(j < length)
	  {
	    T2_1[i + WIDTH * (j+length*k)] = T2[i + block * (j+2+WIDTH*k)];
	    T2_2[i + WIDTH * (j+WIDTH*k)] = T2[length+ i +block * (j+WIDTH*k)]; 
	  }
	else
	  {
	    T2_2[i + WIDTH * (j+WIDTH*k)] = T2[length + i + block * (j+WIDTH*k)];
	  }
      }
    }
  }
}



// T1 for Y-direction
void to_k_to_y_direction_T1(_DOUBLE_ *T1, _DOUBLE_ *T1_1, _DOUBLE_ *T2_1,
			    int block, int WIDTH, int length, int ID)
{
  int i, j, k;
  
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T1, T1_1, T2_1)
#pragma acc loop collapse(3) private(i, j, k)
  for(i = 0; i < WIDTH; ++i){
    for(j = 0; j < WIDTH; ++j){
      for(k = 0; k < block; ++k){
	if(j < length)
	  {
	    T1[i + WIDTH * (j+block*k)] =  T1_1[i + WIDTH * (j+WIDTH*k)];
	    T1[i + WIDTH * (j+((length+2)) + block*k)] =  T2_1[i + WIDTH*(j+length*k)];     
	  }
	else
	  {
	    T1[i + WIDTH * (j+block*k)] =  T1_1[i + WIDTH * (j+WIDTH*k)];
	  }
      } //
    }
  }
}


// T2 for Y-direction
void to_k_to_y_direction_T2(_DOUBLE_ *T2, _DOUBLE_ *T1_2, _DOUBLE_ *T2_2,
			    int block, int WIDTH, int length, int ID)
{
  int i, j, k;
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T2, T1_2, T2_2) 
#pragma acc loop collapse(3) private(i, j, k)
  for(i = 0; i < WIDTH; ++i){
    for(j = 0; j < WIDTH; ++j){
      for(k = 0; k < block; ++k){
	if(j < length)
	  {
	    T2[i + WIDTH * (j+block*k)] =  T1_2[i + WIDTH * (j+length*k)];
	    T2[i + WIDTH * (j+length+block*k)] =  T2_2[i + WIDTH * (j+WIDTH*k)];      
	  } 
	else
	  {     
	    T2[i + WIDTH * (j + length + block*k)] =  T2_2[i + WIDTH * (j+WIDTH*k)];
	  }
      }
    }
  }
}



//*************************************//
// copy the partined data to           //
// X or Z direction for the next fressh//
// new sweep from Y - direction        //    
// partions                            //
//*************************************//  

// copy T1 from Y-direction
void from_y_to_k_direction_T11(double* T1, double *T1_1, double *T1_2, 
			       int block, int WIDTH, int length, int ID)
{
  
  int i, j, k;
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T1, T1_1, T1_2)
#pragma acc loop collapse(3) private(i, j, k)
  for(i = 0; i < WIDTH; ++i){
    for(j = 0; j < WIDTH; ++j){
      for(k = 0; k < block; ++k){
	if(i < length)
	  {
	    T1_1[i + WIDTH * (j+WIDTH*k)] = T1[i + WIDTH * (j+block*k)];
	    T1_2[i + length * (j+WIDTH*k)] = T1[i + WIDTH * (length+j+block*k)];	     
	  }
	else
	  {
	    T1_1[i + WIDTH * (j + WIDTH*k)] = T1[i + WIDTH * (j+block*k)];	
	  }  
      }
    }
  }
}

// copy T2 from Y-direction
void from_y_to_k_direction_T21(double* T2, double *T2_1, double *T2_2, 
			       int block, int WIDTH, int length, int ID)
{
  int i, j, k;
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T2, T2_1, T2_2)
#pragma acc loop collapse(3)  private(i, j, k)
  for(i = 0; i < WIDTH; ++i){
    for(j = 0; j < WIDTH; ++j){
      for(k = 0; k < block; ++k){      
	if(i < length)
	  {
	    T2_1[i + length * (j+WIDTH*k)] = T2[i + 2 + WIDTH * (j+block*k)];
	    T2_2[i + WIDTH * (j+WIDTH*k)] = T2[i + WIDTH * (length+j+block*k)];	     
	  }
	else
	  {
	    T2_2[i + WIDTH * (j+WIDTH*k)] = T2[i + WIDTH * (length+j+block*k)];	
	  }
      }
    }
  }
}


// T1 for X and Z direction
void to_y_to_k_direction_T11(double *T1, double* T1_1, double* T2_1,
			     int block, int WIDTH, int length, int ID)
{
  int i, j, k;
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T1, T1_1, T2_1)
#pragma acc loop collapse(3)  private(i, j, k)
  for(i = 0; i < WIDTH; ++i){
    for(j = 0; j < WIDTH; ++j){
      for(k = 0; k < block; ++k){      
	
	if(i < length)
	  {
	    T1[i + block * (j+WIDTH*k)] =  T1_1[i + WIDTH * (j+WIDTH*k)]; 
	    T1[length + 2 + i + block * (j+WIDTH*k)] = T2_1[i + length * (j+WIDTH*k)]; 
	  }
	else
	  {
	    T1[i + block * (j+WIDTH*k)] = T1_1[i + WIDTH * (j+WIDTH*k)]; 	
	  }  
      }
    }
  }
}


// T2 for X and Z direction
void to_y_to_k_direction_T21(double *T2, double* T1_2, double* T2_2,
			     int block, int WIDTH, int length, int ID)
{
  int i, j, k;
  acc_set_device_num(ID, acc_device_nvidia);
#pragma acc parallel deviceptr(T2, T1_2, T2_2) 
#pragma acc loop collapse(3)  private(i, j, k)
  for(i = 0; i < WIDTH; ++i){
    for(j = 0; j < WIDTH; ++j){
      for(k = 0; k < block; ++k){      
	
	if(i < length)
	  {  
	    T2[i + block * (j+WIDTH*k)] =  T1_2[i + length * (j+WIDTH*k)]; 
	    T2[length + i + block * (j+WIDTH*k)] = T2_2[i + WIDTH * (j+WIDTH*k)];	
	  }
	else
	  {
	    T2[length + i + block * (j+WIDTH*k)] = T2_2[i + WIDTH * (j+WIDTH*k)]; 	
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

  int device_id = omp_get_default_device();
  int host_id = omp_get_initial_device();
  int num = omp_get_num_devices();
  
  printf("device numbers %d \n", device_id) ;
  
  // for loops and other utilities 
  int i, j, k;
 
  _DOUBLE_ DX = ( _xmax - _xmin) / (_nx - 1.0);
  _DOUBLE_ dy = ( _ymax - _ymin) / (_ny - 1.0);
  _DOUBLE_ dz = ( _zmax - _zmin) / (_nz - 1.0);

  _DOUBLE_ DXYP = sqrt(DX*DX + dy*dy);
  _DOUBLE_ DXZP = sqrt(DX*DX + dz*dz);
  _DOUBLE_ DYZP = sqrt(dy*dy + dz*dz);
  _DOUBLE_ DXYZP = sqrt(DX*DX + dy*dy + dz*dz);
   
  // length of the 3D array size in 1-direction + ghost points 
  int block = _nx + 2;

  // sub block height for the 3D array 
  int height = (_nx/2) + 2;
  
  // sub block height with out ghost points
  // for the data trasfer from the GPU-CPU
  int width = _nx/2;
  

  // memory allocation for the calculation 1D- > 3D
  _DOUBLE_*T = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * block * block * block);
  
  double start_initT = wtime();
  printf(" Initialization is started\n");

  ImplicitInitialiser(T, _nx, _ny, _nz, DX, dy, dz);	

  printf(" Initialization is finished\n");
  double end_initT = wtime();
  printf(" preparation work before go to GPU \n");
    
     
  double F = _Fc;

  // data size for sub blocks transfer from CPU-GPU vice-versa
  _DOUBLE_ size = height * block * block * sizeof(_DOUBLE_);
  // data size of boundary layer transfer between GPU-GPU
  _DOUBLE_ buff_size = sizeof(_DOUBLE_) * _nx;
  // data size for partial data transfer from GPU-CPU vice-versa
  _DOUBLE_ data_size = sizeof(_DOUBLE_) * width * height * block;
  

  // memmory allocation of blocks of data T1 and T2 for 2- GPU's
  _DOUBLE_ *restrict T1 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * size);
  _DOUBLE_ *restrict T2 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * size);

    
  // data partion from the initialized T
  for(k = 0 ;  k < block; k++)
    {
      for(j = 0; j < height; j++)
	{
	  for(i = 0; i < block; i++)
	    {
	      T1[i + block*(j + height * k)] = T[i + block * (j + block * k)];
	      T2[i + block*(j + height * k)] = T[i + block * (j + width + block * k)];
	    }
	}
    }

   
  _DOUBLE_ *restrict c_T1 = NULL;   // GPU block 1
  _DOUBLE_ *restrict c_T1_1 = NULL; // GPU partial data transfer
  _DOUBLE_ *restrict c_T1_2 = NULL; // GPU partial data transfer
  
  _DOUBLE_ *restrict c_T2 = NULL;   // GPU block 2
  _DOUBLE_ *restrict c_T2_1 = NULL; // GPU partial data transfer 
  _DOUBLE_ *restrict c_T2_2 = NULL; // GPU partial data transfer
  
  _DOUBLE_ *restrict a_1_2 = NULL;  // data for bounday layer data transfer
  _DOUBLE_ *restrict a_2_1 = NULL;  // data for bounday layer data transfer
 
  // CPU memmory for boundary layer trasfer
  _DOUBLE_ *restrict b_1_2 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * _nx);
  _DOUBLE_ *restrict b_2_1 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * _nx);

  // CPU memory for partial data-transfer from GPU-CPU
  _DOUBLE_ *restrict d_1_2 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * width * block * height); 
  _DOUBLE_ *restrict d_2_1 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * width * block * height);


  // set cuda device 1
  acc_set_device_num(0, acc_device_nvidia);
  c_T1 = acc_malloc((size_t)size);
  c_T1_1 = acc_malloc((size_t)sizeof(_DOUBLE_) * height * block * height);
  c_T1_2 = acc_malloc((size_t)sizeof(_DOUBLE_) * width * block * height);
  a_1_2 = acc_malloc((size_t)sizeof(_DOUBLE_) * _nx);
  acc_memcpy_to_device(c_T1, T1, (size_t)size);


  // set cuda device 2
  acc_set_device_num(1, acc_device_nvidia);
  c_T2 = acc_malloc((size_t)size);
  c_T2_1 = acc_malloc((size_t)sizeof(_DOUBLE_) * width * block * height);
  c_T2_2 = acc_malloc((size_t)sizeof(_DOUBLE_) * height * block * height);
  a_2_1 = acc_malloc((size_t)sizeof(_DOUBLE_) * _nx);
  acc_memcpy_to_device(c_T2, T2, (size_t)size);
  
  // for the top layer exchange 
  int top = 1;
  // for the bottom layer exchange
  int bot = _nx/2;
  printf(" preperation work finished before before go to GPU \n");
  printf(" GPU starts \n");

  // for sweep count
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
      printf(" sweep no is %d\n", sweep+1);
      direction_1 -=  omp_get_wtime();
      for(i = 1; i < _nx ; i++)
	{	
          // computation  
	  function_x_down(c_T1, i, DX, dz, dy, DYZP, DXZP, DXYP, F, _ayy, _azz, _axx, height, 0);
	  x_copy_from_1(1, a_1_2, c_T1, i-1, block, height, bot, 0);
	  
          function_x_down(c_T2, i, DX, dz, dy, DYZP, DXZP, DXYP, F, _ayy, _azz, _axx, height, 1);
          x_copy_from_2(1, a_2_1, c_T2, i-1, block, height, top,  bot, 1);
	  

	  acc_memcpy_from_device(b_1_2, a_1_2, buff_size);	  
	  acc_memcpy_from_device(b_2_1, a_2_1, buff_size);
	  
          
	  acc_memcpy_to_device(a_1_2, b_2_1, buff_size);
          x_copy_to_1(1, a_1_2, c_T1, i-1, block, height, bot+1, 0);
	  
	  
	  acc_memcpy_to_device(a_2_1, b_1_2, buff_size);
	  x_copy_to_2(1, a_2_1, c_T2, i-1, block, height, bot+1, 1);  
	}
      direction_1 +=  omp_get_wtime();      
      direction_2 -=  omp_get_wtime();            
      for(i = _nx; i > 1 ; i--)
	{
	  // computation 	  	  
	  function_x_up(c_T1, i, DX, dz, dy, DYZP, DXZP, DXYP, F, -_ayy, -_azz, -_axx, height, 0);
	  x_copy_from_1(-1, a_1_2, c_T1, i-1, block, height, bot, 0);
	  
	  function_x_up(c_T2, i, DX, dz, dy, DYZP, DXZP, DXYP, F, -_ayy, -_azz, -_axx, height, 1); 
	  x_copy_from_2(-1, a_2_1, c_T2, i-1, block, height, top,  bot, 1);
	  
	  acc_memcpy_from_device(b_1_2, a_1_2, buff_size);	  
	  acc_memcpy_from_device(b_2_1, a_2_1, buff_size);
	  
	  acc_memcpy_to_device(a_1_2, b_2_1, buff_size);
	  x_copy_to_1(-1, a_1_2, c_T1, i-1, block, height, bot+1, 0);
	  
	  acc_memcpy_to_device(a_2_1, b_1_2, buff_size);
	  x_copy_to_2(-1, a_2_1, c_T2, i-1, block, height, bot+1, 1);	    
	}
      direction_2 +=  omp_get_wtime();

      direction_3 -=  omp_get_wtime();      
      for(i = 1; i < _nx; i++)
	{
	  // computation 	  
	  function_z_up(c_T1, i, dz, dy, DX, DXYP, DYZP, DXZP, F, _axx, _ayy, _azz, height, 0);
	  z_copy_from_1(1, a_1_2, c_T1, i-1, block, height, bot, 0);	 
	  	  
	  function_z_up(c_T2, i, dz, dy, DX, DXYP, DYZP, DXZP, F, _axx, _ayy, _azz, height, 1);
	  z_copy_from_2(1, a_2_1, c_T2, i-1, block, height, top,  bot, 1);
	  
	  acc_memcpy_from_device(b_1_2, a_1_2, buff_size);	  
	  acc_memcpy_from_device(b_2_1, a_2_1, buff_size);	    
	  
	  acc_memcpy_to_device(a_1_2, b_2_1, buff_size);
	  z_copy_to_1(1, a_1_2, c_T1, i-1, block, height, bot+1, 0);
	  
	  acc_memcpy_to_device(a_2_1, b_1_2, buff_size);
	  z_copy_to_2(1, a_2_1, c_T2, i-1, block, height, bot+1, 1);	    
	}
      direction_3 +=  omp_get_wtime();

      direction_4 -=  omp_get_wtime();      
      for(i  = _ny; i > 1; i--)
	{
	  // computation           
	  function_z_down(c_T1, i, dz, dy, DX, DXYP, DYZP, DXZP, F, -_axx, -_ayy, -_azz, height, 0);
	  z_copy_from_1(-1, a_1_2, c_T1, i-1, block, height, bot, 0);
	  	  
          function_z_down(c_T2, i, dz, dy, DX, DXYP, DYZP, DXZP, F, -_axx, -_ayy, -_azz, height, 1);
	  z_copy_from_2(-1, a_2_1, c_T2, i-1, block, height, top,  bot, 1);
	  
	  acc_memcpy_from_device(b_1_2, a_1_2, buff_size);	  
	  acc_memcpy_from_device(b_2_1, a_2_1, buff_size);

	  acc_memcpy_to_device(a_1_2, b_2_1, buff_size);
	  z_copy_to_1(-1, a_1_2, c_T1, i-1, block, height, bot+1, 0);
	  
	  acc_memcpy_to_device(a_2_1, b_1_2, buff_size);
	  z_copy_to_2(-1, a_2_1, c_T2, i-1, block, height, bot+1, 1);    
	}
      direction_4 +=  omp_get_wtime();

      // time count starts for MULTIPLE GPU
      //      data_start =  omp_get_wtime();//wtime();//omp_get_wtime;
      data_timer_1 -=  omp_get_wtime();//wtime();//omp_get_wtime;
      from_k_to_y_direction_T1(c_T1, c_T1_1, c_T1_2, block, height, width, 0);
      from_k_to_y_direction_T2(c_T2, c_T2_1, c_T2_2, block, height, width, 1);
      
      acc_memcpy_from_device(d_1_2, c_T1_2, data_size);      
      acc_memcpy_from_device(d_2_1, c_T2_1, data_size);     
      acc_memcpy_to_device(c_T1_2, d_2_1, data_size);      
      acc_memcpy_to_device(c_T2_1, d_1_2, data_size);
      
      to_k_to_y_direction_T1(c_T1, c_T1_1, c_T1_2, block, height, width, 0);      
      to_k_to_y_direction_T2(c_T2, c_T2_1, c_T2_2, block, height, width, 1);
      //      double end_data = wtime();
      //      data_end = omp_get_wtime();// wtime();//omp_get_wtime();
      data_timer_1 += omp_get_wtime();// wtime();//omp_get_wtime();      

      direction_5 -=  omp_get_wtime();            
      for(i = _ny; i > 1; i--)
	{
	  // computation 	  
	  function_y_up(c_T1, i,  dy, DX, dz, DXZP, DXYP, DYZP, F, -_azz, -_axx, -_ayy, height, 0);
	  y_copy_from_1(-1, a_1_2, c_T1, i-1, block, height, bot, 0);	
          
          function_y_up(c_T2, i,  dy, DX, dz, DXZP, DXYP, DYZP, F, -_azz, -_axx, -_ayy, height, 1);
	  y_copy_from_2(-1, a_2_1, c_T2, i-1, block, height, top,  bot, 1);
	    
	  acc_memcpy_from_device(b_1_2, a_1_2, buff_size);	  
	  acc_memcpy_from_device(b_2_1, a_2_1, buff_size);
	  
	  acc_memcpy_to_device(a_1_2, b_2_1, buff_size);
	  y_copy_to_1(-1, a_1_2, c_T1, i-1, block, height, bot+1, 0);

	  acc_memcpy_to_device(a_2_1, b_1_2, buff_size);
	  y_copy_to_2(-1, a_2_1, c_T2, i-1, block, height, bot+1, 1);	    
	}
      direction_5 +=  omp_get_wtime();            

      direction_6 -=  omp_get_wtime();            
      for(i = 1; i < _ny; i++)
	{
	  // computation  	  
	  function_y_down(c_T1, i, dy, DX, dz, DXZP, DXYP, DYZP, F, _azz, _axx, _ayy, height, 0);
	  y_copy_from_1(1, a_1_2, c_T1, i-1, block, height, bot, 0);
	  
          function_y_down(c_T2, i, dy, DX, dz, DXZP, DXYP, DYZP, F, _azz, _axx, _ayy, height, 1);
	  y_copy_from_2(1, a_2_1, c_T2, i-1, block, height, top,  bot, 1);
	  
	  acc_memcpy_from_device(b_1_2, a_1_2, buff_size);	  
	  acc_memcpy_from_device(b_2_1, a_2_1, buff_size);
	  
	  acc_memcpy_to_device(a_1_2, b_2_1, buff_size);
	  y_copy_to_1(1, a_1_2, c_T1, i-1, block, height, bot+1, 0);
	  
	  acc_memcpy_to_device(a_2_1, b_1_2, buff_size);
	  y_copy_to_2(1, a_2_1, c_T2, i-1, block, height, bot+1, 1);
	  
	}
      direction_6 +=  omp_get_wtime();
      
      data_timer_2 -=  omp_get_wtime();//wtime();//omp_get_wtime;      
      from_y_to_k_direction_T11(c_T1, c_T1_1, c_T1_2, block, height, width, 0);    
      from_y_to_k_direction_T21(c_T2, c_T2_1, c_T2_2, block, height, width, 1);
      
      acc_memcpy_from_device(d_1_2, c_T1_2, data_size);
      acc_memcpy_from_device(d_2_1, c_T2_1, data_size);      
      acc_memcpy_to_device(c_T1_2, d_2_1, data_size);      
      acc_memcpy_to_device(c_T2_1, d_1_2, data_size);
      
      to_y_to_k_direction_T11(c_T1, c_T1_1, c_T1_2, block, height, width, 0);
      to_y_to_k_direction_T21(c_T2, c_T2_1, c_T2_2, block, height, width, 1);
      data_timer_2 +=  omp_get_wtime();//wtime();//omp_get_wtime;

      //      timer -= data_start; // Subtract start time from data_timer
      //      timer += data_end;   // Add end time to data_timer    
    }
 
 
  // end of the total time
  compute_timer += omp_get_wtime();
  //("total time is %lf sec\n",compute_timer);
  
  // copy back the computed solutioin in GPU to CPU
  
  //cudaMemcpy(T1, c_T1, size, cudaMemcpyDeviceToHost);
  acc_memcpy_from_device(T1, c_T1, size);
  
  //cudaMemcpy(T2, c_T2, size, cudaMemcpyDeviceToHost);
  acc_memcpy_from_device(T2, c_T2, size);
  
  // free the GPU memeory and boudary value buffer
  acc_free(c_T1);  acc_free(a_1_2);
  acc_free(c_T2);  acc_free(a_2_1); 
  
  // free the GPU partial data transfer Buffer
  acc_free(c_T2_1);   acc_free(c_T1_1);
  acc_free(c_T2_2);   acc_free(c_T1_2);

  // put the partitioned data into 1D arrray 
  for(i = 0 ; i < block; i++)
    {
      for(j = 0 ; j < height-1; j++)
        {
	  for(k = 0 ; k < block; k++)
	    {
	      T[i + block * (j + block * k)] = T1[i + block * (j+height * k)]; 
	      T[i + block * ((j + (width+1))+block * k)]  = T2[i + block * ((j+1)+height * k)];
	    }
        }
    }
  
  // free the CPU partition data memory and buffer of 
  // boundary value 
  free(T1); free(b_1_2);
  free(T2); free(b_2_1); 
  
  free(d_1_2); free(d_2_1);
  
  // printf the solution in a file
#ifdef PRI
  FILE *init;
  init = fopen("an_iso_2_openacc.txt","w");  
  for(i = 0; i < _nz+2; i++)
    {
      for(j = 0; j < _nz+2; j++)
	{
	  for(k = 0; k < _nz+2; k++)
	    {
	      fprintf(init,"%f\n",T[i + HEIGHT * (j + HEIGHT * k)]);
	    }
	}
    }
  
  fclose(init);
#endif
  
  free(T);

  printf("*******************************************************************************************\n");
  printf(" Initialised T, in CPU  -------------------------------- %3f seconds.\n", end_initT-start_initT);
  //  printf(" Data partition time   --------------------------------- %f seconds.\n", timer);
  printf(" Data partition time 1 --------------------------------- %f seconds.\n", data_timer_1);
  printf(" Data partition time 2 --------------------------------- %f seconds.\n", data_timer_2);
  printf(" Direction sweep 1     --------------------------------- %f seconds.\n", direction_1);
  printf(" Direction sweep 2     --------------------------------- %f seconds.\n", direction_2);
  printf(" Direction sweep 3     --------------------------------- %f seconds.\n", direction_3);
  printf(" Direction sweep 4     --------------------------------- %f seconds.\n", direction_4);
  printf(" Direction sweep 5     --------------------------------- %f seconds.\n", direction_5);
  printf(" Direction sweep 6     --------------------------------- %f seconds.\n", direction_6);	  
  printf(" Created grid size: %d x %d x %d\n", _nx, _ny, _nz);
  printf(" Model is ---------------------------------------------- AN_Isotropic \n");
  printf(" Total sweep is ---------------------------------------- %d \n", tot);
  printf(" Time taken for the 2 GPU is --------------------------- %f seconds\n",compute_timer);

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
  printf(" Solution is written in the file------------------------ an_iso_2_openacc.txt\n");
#endif
  printf("*******************************************************************************************\n");

  return 0;
}
