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

//#define VTK_PRI // print the VTK solution
//#define PRI
#include "Fold3dPMM.h"

#ifndef fmin
#define fmin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#ifndef sign
#define sign(a) (a > 0) ? 1 : -1
#endif

__device__
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


__device__
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


__global__
void function_x_down(double *T, int i, 
		     _DOUBLE_ dzz, _DOUBLE_ dxx, _DOUBLE_ dyy, 
		     _DOUBLE_ dxy, _DOUBLE_ dxz, _DOUBLE_ dyz, 
		     double F, double ay, double ax, double az,int WIDTH)
{	
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 

  int j = (blockDim.y * blockIdx.y + threadIdx.y)+1;
  int k = (blockDim.x * blockIdx.x + threadIdx.x)+1;

  if ((j >= 1 && j < WIDTH-1) && (k >= 1 && k < HEIGHT-1))
    { 
      tnew = fabs(__ldg(&T[(i+1) + HEIGHT * (j + WIDTH * k)]));
      st =  fabs(__ldg(&T[i + HEIGHT * (j + WIDTH * k)]));
      xt  = fabs(__ldg(&T[i + HEIGHT * (j + WIDTH * (k+1))]));
      yt  = fabs(__ldg(&T[i + HEIGHT * ((j+1) + WIDTH * k)]));
      xnt = fabs(__ldg(&T[i + HEIGHT * (j + WIDTH * (k-1))]));
      ynt = fabs(__ldg(&T[i + HEIGHT * ((j-1) + WIDTH * k)]));
      txm = fabs(__ldg(&T[i + HEIGHT * ((j-1) + WIDTH * (k+1))]));
      txy = fabs(__ldg(&T[i + HEIGHT * ((j+1) + WIDTH * (k+1))]));
      tym = fabs(__ldg(&T[i + HEIGHT * ((j+1) + WIDTH * (k-1))]));
      txnyn = fabs(__ldg(&T[i + HEIGHT * ((j-1) + WIDTH * (k-1))]));

  
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


__global__
void function_x_up(double *T, int i, 
		   _DOUBLE_ dzz, _DOUBLE_ dxx, _DOUBLE_ dyy, 
		   _DOUBLE_ dxy, _DOUBLE_ dxz, _DOUBLE_ dyz, 
		   double F, double ay, double ax, double az, int WIDTH)
{
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn;
  
  int j = (blockDim.y * blockIdx.y + threadIdx.y)+1;
  int k = (blockDim.x * blockIdx.x + threadIdx.x)+1;

  if ((j >= 1 && j < WIDTH-1) && (k >= 1 && k < HEIGHT-1))
    { 

  
      tnew = fabs(__ldg(&T[(i-1) + HEIGHT * (j + WIDTH * k)]));
      st =  fabs(__ldg(&T[i + HEIGHT * (j + WIDTH * k)]));
      xt  = fabs(__ldg(&T[i + HEIGHT * (j + WIDTH * (k-1))]));
      yt  = fabs(__ldg(&T[i + HEIGHT * ((j-1) + WIDTH * k)]));
      xnt = fabs(__ldg(&T[i + HEIGHT * (j + WIDTH * (k+1))]));
      ynt = fabs(__ldg(&T[i + HEIGHT * ((j+1) + WIDTH * k)]));
      txm = fabs(__ldg(&T[i + HEIGHT * ((j+1) + WIDTH * (k-1))]));
      txy = fabs(__ldg(&T[i + HEIGHT * ((j-1) + WIDTH * (k-1))]));
      tym = fabs(__ldg(&T[i + HEIGHT * ((j-1) + WIDTH* (k+1))]));
      txnyn = fabs(__ldg(&T[i + HEIGHT * ((j+1) + WIDTH * (k+1))]));
  
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



__global__
void function_y_up(double *T, int j, 
		   _DOUBLE_ dzz, _DOUBLE_ dxx, _DOUBLE_ dyy, 
		   _DOUBLE_ dxy, _DOUBLE_ dxz, _DOUBLE_ dyz, 
		   double F, double ay, double ax, double az, int WIDTH)
  
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  
  int i = (blockDim.y * blockIdx.y + threadIdx.y)+1;
  int k = (blockDim.x * blockIdx.x + threadIdx.x)+1;
  
  if ((i >= 1 && i < WIDTH-1) && (k >= 1 && k < HEIGHT-1))
    { 


      tnew = fabs(__ldg(&T[i+ WIDTH *((j-1)+HEIGHT*k)]));
      st =  fabs(__ldg(&T[i +WIDTH * (j + HEIGHT * k)]));
      xt  = fabs(__ldg(&T[(i-1) + WIDTH * (j + HEIGHT * k)]));
      yt  = fabs(__ldg(&T[i + WIDTH * (j + HEIGHT * (k-1))]));
      xnt = fabs(__ldg(&T[(i+1) + WIDTH * (j + HEIGHT * k)]));
      ynt = fabs(__ldg(&T[i + WIDTH * (j + HEIGHT * (k+1))]));
      txm = fabs(__ldg(&T[(i-1)+ WIDTH * (j + HEIGHT * (k+1))]));
      txy = fabs(__ldg(&T[(i-1) + WIDTH * (j + HEIGHT * (k-1))]));
      tym = fabs(__ldg(&T[(i+1) + WIDTH * (j + HEIGHT * (k-1))]));
      txnyn = fabs(__ldg(&T[(i+1) + WIDTH * (j + HEIGHT * (k+1))]));
  
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

__global__
void function_y_down(double *T, int j, 
		     _DOUBLE_ dzz, _DOUBLE_ dxx, _DOUBLE_ dyy, 
		     _DOUBLE_ dxy, _DOUBLE_ dxz, _DOUBLE_ dyz, 
		     double F, double ay, double ax, double az, int WIDTH)  
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn;
  
  int i = (blockDim.y * blockIdx.y + threadIdx.y)+1;
  int k = (blockDim.x * blockIdx.x + threadIdx.x)+1;

  if ((i >= 1 && i < WIDTH-1) && (k >= 1 && k < HEIGHT-1))
    { 

  
      tnew = fabs(__ldg(&T[i + WIDTH * ((j+1) + HEIGHT * k)]));
      st =  fabs(__ldg(&T[i + WIDTH * (j + HEIGHT * k)]));
      xt  = fabs(__ldg(&T[(i+1) + WIDTH * (j + HEIGHT * k)]));
      yt  = fabs(__ldg(&T[i + WIDTH * (j + HEIGHT * (k+1))]));
      xnt = fabs(__ldg(&T[(i-1) + WIDTH * (j + HEIGHT * k)]));
      ynt = fabs(__ldg(&T[i + WIDTH * (j + HEIGHT * (k-1))]));
      txm =  fabs(__ldg(&T[(i+1) + WIDTH * (j + HEIGHT * (k-1))]));
      txy =  fabs(__ldg(&T[(i+1) + WIDTH * (j + HEIGHT * (k+1))]));
      tym =  fabs(__ldg(&T[(i-1) + WIDTH * (j + HEIGHT * (k+1))]));
      txnyn =fabs(__ldg(&T[(i-1) + WIDTH * (j + HEIGHT * (k-1))]));
  

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


__global__
void function_z_up(double *T, int k, 
		   _DOUBLE_ dzz, _DOUBLE_ dxx, _DOUBLE_ dyy, 
		   _DOUBLE_ dxy, _DOUBLE_ dxz, _DOUBLE_ dyz, 
		   double F, double ay, double ax, double az, int WIDTH)
  
{

  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn;
  
  int i = (blockDim.x * blockIdx.x + threadIdx.x)+1;
  int j = (blockDim.y * blockIdx.y + threadIdx.y)+1;
  
  if ((j >= 1 && j < WIDTH-1) && (i >= 1 && i < HEIGHT-1))
    { 

  
      tnew = fabs(__ldg(&T[i + HEIGHT * (j + WIDTH * (k+1))]));
      st =  fabs(__ldg(&T[i + HEIGHT * (j + WIDTH * k)]));
      xt  = fabs(__ldg(&T[i + HEIGHT * ((j+1) + WIDTH * k)]));
      yt  = fabs(__ldg(&T[(i+1) + HEIGHT * (j + WIDTH * k)]));
      xnt = fabs(__ldg(&T[i + HEIGHT * ((j-1) + WIDTH * k)]));
      ynt = fabs(__ldg(&T[(i-1) + HEIGHT * (j + WIDTH * k)]));
      txm =  fabs(__ldg(&T[(i-1) + HEIGHT * ((j+1) + WIDTH * k)]));
      txy =  fabs(__ldg(&T[(i+1) + HEIGHT * ((j+1) + WIDTH * k)]));
      tym =  fabs(__ldg(&T[(i+1) + HEIGHT * ((j-1) +WIDTH * k)]));
      txnyn =fabs(__ldg(&T[(i-1) + HEIGHT * ((j-1) + WIDTH * k)]));
  
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


__global__
void function_z_down(double *T, int k, 
		     _DOUBLE_ dzz, _DOUBLE_ dxx, _DOUBLE_ dyy, 
		     _DOUBLE_ dxy, _DOUBLE_ dxz, _DOUBLE_ dyz, 
		     double F, double ay, double ax, double az, int WIDTH)
  
{
  
  double tnew, st, xt, yt, txy, xnt, ynt, txm, tym, txnyn; 
  
  int j = (blockDim.y * blockIdx.y + threadIdx.y)+1;
  int i = (blockDim.x * blockIdx.x + threadIdx.x)+1;

  if ((j >= 1 && j < WIDTH-1) && (i >= 1 && i < HEIGHT-1))
    { 

  
      tnew = fabs(__ldg(&T[i + HEIGHT * (j + WIDTH * (k-1))]));
      st =  fabs(__ldg(&T[i + HEIGHT * (j + WIDTH * k)]));
      xt  = fabs(__ldg(&T[i + HEIGHT * ((j-1) + WIDTH * k)]));
      yt  = fabs(__ldg(&T[(i-1) + HEIGHT * (j + WIDTH * k)]));
      xnt = fabs(__ldg(&T[i + HEIGHT * ((j+1) + WIDTH * k)]));
      ynt = fabs(__ldg(&T[(i+1) + HEIGHT * (j + WIDTH * k)]));
      txm =  fabs(__ldg(&T[(i+1) + HEIGHT * ((j-1) + WIDTH * k)]));
      txy =  fabs(__ldg(&T[(i-1) + HEIGHT * ((j-1) + WIDTH * k)]));
      tym =  fabs(__ldg(&T[(i-1) + HEIGHT * ((j+1) + WIDTH * k)]));
      txnyn =fabs(__ldg(&T[(i+1) + HEIGHT * ((j+1) + WIDTH * k)]));
  

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





//****************************//
// boudary layer exchange     //
// between GPU's              //
//****************************//

// Save top bounday layer from T1
__global__
void z_copy_from_1(int dk, double *a_1_2, double* c_T1, 
		   int k, int block, int WIDTH, int bot)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ( i >= 0 && i < block-2)
    {
      a_1_2[i] = c_T1[(i+1) + block* (bot + WIDTH * (k+dk+1))];
    }
}

// Save top bounday layer from T2
__global__
void z_copy_from_2(int dk, double *a_2_1, double* c_T2, 
		   int k, int block, int WIDTH, int top, int bot)
{  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ( i >= 0 && i < block-2)
    {
      a_2_1[i] = c_T2[(i+1) + block * (top + WIDTH * (k+dk+1))]; 
    }
}

// New bounday layer for top of T1
__global__
void z_copy_to_1(int dk, double* a_2_1, double *c_T1, 
		 int k, int block, int WIDTH, int bot)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ( i >= 0 && i < block-2)
    {
      c_T1[(i+1) + block * (bot + WIDTH * (k+dk+1))]  =  a_2_1[i];
    }
}

// New bounday layer for top of T2
__global__
void z_copy_to_2(int dk, double* a_1_2, double *c_T2, 
		 int k, int block, int WIDTH, int bot)
{ 
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ( i >= 0 && i < block-2)
    {
      c_T2[(i+1) + block * (0 + WIDTH * (k+dk+1))]  = a_1_2[i];
    } 
}

// Save top bounday layer from T1
__global__
void x_copy_from_1(int dk, double *a_1_2, double* c_T1, 
		   int i, int block, int WIDTH, int bot)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if ( k >= 0 && k < block-2)
    {
      a_1_2[k] = c_T1[(i+1+dk) + block * (bot + WIDTH * (k+1))] ;
    }
}

// Save top bounday layer from T2
__global__
void x_copy_from_2(int dk, double *a_2_1, double* c_T2, 
		   int i, int block, int WIDTH, int top, int bot)
{ 
  int k = blockIdx.x * blockDim.x + threadIdx.x; 
  if ( k >= 0 && k < block-2)
    {
      a_2_1[k] = c_T2[(i+1+dk) + block * (top + WIDTH * (k+1))] ;
    }
}

// New bounday layer for top of T1
__global__
void x_copy_to_1(int dk, double* a_2_1, double *c_T1, 
		 int i, int block, int WIDTH, int bot)
{  
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if ( k >= 0 && k < block-2)
    {
      c_T1[(i+1+dk) + block * (bot + WIDTH * (k+1))]  =  a_2_1[k];
    }  
}

// New bounday layer for top of T2
__global__
void x_copy_to_2(int dk, double* a_1_2, double *c_T2, 
		 int i, int block, int WIDTH, int bot)
{ 
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if ( k >= 0 && k < block-2)
    {
      c_T2[(i+1+dk) + block * (0 + WIDTH * (k+1))]  = a_1_2[k];
    }
}

// Save bottom bounday layer from T1
__global__
void y_copy_from_1(int dk, double *a_1_2, double* c_T1, 
		   int j, int block, int WIDTH, int bot)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if ( k >= 0 && k < block-2)
    {
      a_1_2[k] = c_T1[bot + WIDTH * ((j+1+dk) + block * (k+1))] ; 
    } 
}

// Save top bounday layer from T2
__global__
void y_copy_from_2(int dk, double *a_2_1, double* c_T2, 
		   int j, int block, int WIDTH, int top, int bot)
{ 
  int k = blockIdx.x * blockDim.x + threadIdx.x;  
  if ( k >= 0 && k < block-2)
    {
      a_2_1[k] = c_T2[top + WIDTH * ((j+1+dk) + block * (k+1))] ;
    } 
}

// New bounday layer for bottom of T1
__global__
void y_copy_to_1(int dk, double* a_2_1, double *c_T1, 
		 int j, int block, int WIDTH, int bot)
{ 
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if ( k >= 0 && k < block-2)
    {
      c_T1[bot + WIDTH * ((j+1+dk) + block * (k+1))]  =  a_2_1[k];
    }
}

// New bounday layer for top of T2
__global__
void y_copy_to_2(int dk, double* a_1_2, double *c_T2, int j,
		 int block, int WIDTH, int bot)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if ( k >= 0 && k < block-2)
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
__global__
void from_k_to_y_direction_T1(const _DOUBLE_* __restrict__ T1, _DOUBLE_ *T1_1, _DOUBLE_ *T1_2, 
			      int block, int WIDTH, int length)
{
  int i, j, k;
  
  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;
  k = blockIdx.z * blockDim.z + threadIdx.z;

  if( (i >= 0 && i < WIDTH) && ( j >= 0 && j < WIDTH ) && ( k >= 0 && k < block))
    {
      
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


// copy T2 from Z-direction
__global__
void from_k_to_y_direction_T2(const _DOUBLE_* __restrict__ T2,_DOUBLE_ *T2_1, _DOUBLE_ *T2_2, 
			      int block, int WIDTH, int length)
{
  
  int i, j, k;
  
  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;
  k = blockIdx.z * blockDim.z + threadIdx.z;
  if( (i >= 0 && i < WIDTH) && ( j >= 0 && j < WIDTH ) && ( k >= 0 && k < block))
    {
      
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

// T1 for Y-direction
__global__
void to_k_to_y_direction_T1(_DOUBLE_ *T1, const _DOUBLE_* __restrict__ T1_1, const _DOUBLE_* __restrict__ T2_1,
			    int block, int WIDTH, int length)
{
  int i, j, k;
  
  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;
  k = blockIdx.z * blockDim.z + threadIdx.z;
  if( (i >= 0 && i < WIDTH) && ( j >= 0 && j < WIDTH ) && ( k >= 0 && k < block))
    {
      
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

// T2 for Y-direction
__global__
void to_k_to_y_direction_T2(_DOUBLE_ *T2, const _DOUBLE_* __restrict__ T1_2, const _DOUBLE_* __restrict__ T2_2,
			    int block, int WIDTH, int length)
{
  int i, j, k;
  
  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;
  k = blockIdx.z * blockDim.z + threadIdx.z;
  if( (i >= 0 && i < WIDTH) && ( j >= 0 && j < WIDTH ) && ( k >= 0 && k < block))
    {
      
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



//*************************************//
// copy the partined data to           //
// X or Z direction for the next fressh//
// new sweep from Y - direction        //    
// partions                            //
//*************************************//  

// copy T1 from Y-direction
__global__
void from_y_to_k_direction_T11(double* T1, double *T1_1, double *T1_2, 
			       int block, int WIDTH, int length)
{
  
  int i, j, k;
  
  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;
  k = blockIdx.z * blockDim.z + threadIdx.z;
  
  if( (i >= 0 && i < WIDTH) && ( j >= 0 && j < WIDTH ) && ( k >= 0 && k < block))
    {
      
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


// copy T2 from Y-direction
__global__
void from_y_to_k_direction_T21(double* T2, double *T2_1, double *T2_2, 
			       int block, int WIDTH, int length)
{
  int i, j, k;
  
  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;
  k = blockIdx.z * blockDim.z + threadIdx.z;
  if( (i >= 0 && i < WIDTH) && ( j >= 0 && j < WIDTH ) && ( k >= 0 && k < block))
    {
      
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

// T1 for X and Z direction
__global__
void to_y_to_k_direction_T11(double *T1, double* T1_1, double* T2_1,
			     int block, int WIDTH, int length)
{
  int i, j, k;
  
  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;
  k = blockIdx.z * blockDim.z + threadIdx.z;
  if( (i >= 0 && i < WIDTH) && ( j >= 0 && j < WIDTH ) && ( k >= 0 && k < block))
    {
      
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

// T2 for X and Z direction
__global__
void to_y_to_k_direction_T21(double *T2, double* T1_2, double* T2_2,
			     int block, int WIDTH, int length)
{
  int i, j, k;
  
  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;
  k = blockIdx.z * blockDim.z + threadIdx.z;
  if( (i >= 0 && i < WIDTH) && ( j >= 0 && j < WIDTH ) && ( k >= 0 && k < block))
    {
      
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
  // for loops and other utilities 
  int i, j, k;
 
  _DOUBLE_ DX = ( _xmax - _xmin) / (_nx - 1.0);
  _DOUBLE_ dy = ( _ymax - _ymin) / (_ny - 1.0);
  _DOUBLE_ dz = ( _zmax - _zmin) / (_nz - 1.0);

  _DOUBLE_ DXYP = sqrt(DX*DX + dy*dy);
  _DOUBLE_ DXZP = sqrt(DX*DX + dz*dz);
  _DOUBLE_ DYZP = sqrt(dy*dy + dz*dz);
  _DOUBLE_ DXYZP = sqrt(DX*DX + dy*dy + dz*dz);



  int x_grid = ceil(_nx/8); // number of grids in 1 inner loop
  int y_grid = ceil((_ny/2)/8); // number of grids in 2 inner loop

  // halo_computaion
  dim3 halo_G(ceil(_nx/64)+1, 1, 1);
  dim3 halo_B(64, 1, 1);    
                          
  // main computation                                        
  dim3 comp_G((x_grid+1) ,(y_grid+1), 1);    
  dim3 comp_B(8, 8, 1); 

  // data transfer
  dim3 data_G((y_grid+1) ,(y_grid+1), (_nx+2));    
  dim3 data_B(8, 8, 1);   
   
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
  _DOUBLE_* T1 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * size);
  _DOUBLE_* T2 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * size);

    
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

   
  _DOUBLE_ *c_T1 = NULL;   // GPU block 1
  _DOUBLE_ *c_T1_1 = NULL; // GPU partial data transfer
  _DOUBLE_ *c_T1_2 = NULL; // GPU partial data transfer
  
  _DOUBLE_ *c_T2 = NULL;   // GPU block 2
  _DOUBLE_ *c_T2_1 = NULL; // GPU partial data transfer 
  _DOUBLE_ *c_T2_2 = NULL; // GPU partial data transfer
  
  _DOUBLE_ *a_1_2 = NULL;  // data for bounday layer data transfer
  _DOUBLE_ *a_2_1 = NULL;  // data for bounday layer data transfer
 
  // CPU memmory for boundary layer trasfer
  _DOUBLE_ *b_1_2 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * _nx);
  _DOUBLE_ *b_2_1 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * _nx);

  // CPU memory for partial data-transfer from GPU-CPU
  _DOUBLE_ *d_1_2 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * width * block * height); 
  _DOUBLE_ *d_2_1 = (_DOUBLE_*)malloc(sizeof(_DOUBLE_) * width * block * height);

 
  // set cuda device 1
  cudaSetDevice(0);
  cudaMalloc((void**)&c_T1, size );
  cudaMalloc((void**)&c_T1_1, sizeof(_DOUBLE_) * height * block * height);
  cudaMalloc((void**)&c_T1_2, sizeof(_DOUBLE_) * width * block * height);
  cudaMalloc((void**)&a_1_2, sizeof(_DOUBLE_) * _nx);
  cudaMemcpy(c_T1, T1, size, cudaMemcpyHostToDevice);                                           
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  // set cuda device 2
  cudaSetDevice(1);
  cudaMalloc((void**)&c_T2, size );
  cudaMalloc((void**)&c_T2_1, sizeof(_DOUBLE_) * width * block * height);
  cudaMalloc((void**)&c_T2_2, sizeof(_DOUBLE_) * height * block * height);
  cudaMalloc((void**)&a_2_1, sizeof(_DOUBLE_) * _nx);
  cudaMemcpy(c_T2, T2, size, cudaMemcpyHostToDevice);
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  
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
  compute_timer -= omp_get_wtime();
  
  for(sweep = 0; sweep < tot; sweep++)
    {
      printf(" sweep no is %d\n", sweep+1);
      for(i = 1; i < _nx ; i++)
	{	
          // computation  
	  cudaSetDevice(0);
	  function_x_down<<< comp_G, comp_B >>>(c_T1, i, DX, dz, dy, 
						DYZP, DXZP, DXYP, F, _ayy, _azz, _axx, height);
          x_copy_from_1<<< halo_G, halo_B >>>(1, a_1_2, c_T1, i-1, block, height, bot);
        
	  cudaSetDevice(1);
          function_x_down<<< comp_G, comp_B >>>(c_T2, i, DX, dz, dy, 
						DYZP, DXZP, DXYP, F, _ayy, _azz, _axx, height);
          x_copy_from_2<<< halo_G, halo_B >>>(1, a_2_1, c_T2, i-1, block, height, top,  bot);
	  
          cudaSetDevice(0);
          cudaMemcpy(b_1_2, a_1_2, buff_size, cudaMemcpyDeviceToHost);  
	  
          cudaSetDevice(1);  
          cudaMemcpy(b_2_1, a_2_1, buff_size, cudaMemcpyDeviceToHost);  
	  	    
          cudaSetDevice(0);
          cudaMemcpy(a_1_2, b_2_1, buff_size, cudaMemcpyHostToDevice);
          x_copy_to_1<<< halo_G, halo_B >>>(1, a_1_2, c_T1, i-1, block, height, bot+1);
	  
	  cudaSetDevice(1);
          cudaMemcpy(a_2_1, b_1_2, buff_size, cudaMemcpyHostToDevice);  
	  x_copy_to_2<<< halo_G, halo_B >>>(1, a_2_1, c_T2, i-1, block, height, bot+1);  
	}
      
      
      for(i = _nx; i > 1 ; i--)
	{
	  // computation 	  
	  cudaSetDevice(0);
	  function_x_up<<< comp_G, comp_B >>>(c_T1, i, DX, dz, dy, 
					      DYZP, DXZP, DXYP, F, -_ayy, -_azz, -_axx, height);
	  x_copy_from_1<<< halo_G, halo_B >>>(-1, a_1_2, c_T1, i-1, block, height, bot);

          
	  cudaSetDevice(1);
	  function_x_up<<< comp_G, comp_B >>>(c_T2, i, DX, dz, dy, 
					      DYZP, DXZP, DXYP, F, -_ayy, -_azz, -_axx, height);
	  x_copy_from_2<<< halo_G, halo_B >>>(-1, a_2_1, c_T2, i-1, block, height, top,  bot);
	
	  // boundary layer copy to CPU
	  cudaSetDevice(0);
	  cudaMemcpy(b_1_2, a_1_2, buff_size, cudaMemcpyDeviceToHost);  
	  
	  cudaSetDevice(1);  
	  cudaMemcpy(b_2_1, a_2_1, buff_size, cudaMemcpyDeviceToHost);  
	  	    
	  // copied boundary layer back GPU
	  cudaSetDevice(0);
          cudaMemcpy(a_1_2, b_2_1, buff_size, cudaMemcpyHostToDevice); 
	  x_copy_to_1<<< halo_G, halo_B >>>(-1, a_1_2, c_T1, i-1, block, height, bot+1);
	  
	  cudaSetDevice(1);
          cudaMemcpy(a_2_1, b_1_2, buff_size, cudaMemcpyHostToDevice); 
	  x_copy_to_2<<< halo_G, halo_B >>>(-1, a_2_1, c_T2, i-1, block, height, bot+1);	    
	}
      

      for(i = 1; i < _nx; i++)
	{
	  // computation 
	  cudaSetDevice(0);
	  function_z_up<<< comp_G, comp_B >>>(c_T1, i, dz, dy, DX, 
					      DXYP, DYZP, DXZP, F, _axx, _ayy, _azz, height);
	  z_copy_from_1<<< halo_G, halo_B >>>(1, a_1_2, c_T1, i-1, block, height, bot);	 
	  	  
	  cudaSetDevice(1);
	  function_z_up<<< comp_G, comp_B >>>(c_T2, i, dz, dy, DX, 
					      DXYP, DYZP, DXZP, F, _axx, _ayy, _azz, height);
	  z_copy_from_2<<< halo_G, halo_B >>>(1, a_2_1, c_T2, i-1, block, height, top,  bot);
	   
	  // boundary layer copy to CPU
	  cudaSetDevice(0);
	  cudaMemcpy(b_1_2, a_1_2, buff_size, cudaMemcpyDeviceToHost);  
	  
	  cudaSetDevice(1);
	  cudaMemcpy(b_2_1, a_2_1, buff_size, cudaMemcpyDeviceToHost); 
	  	    
	  // copied boundary layer back GPU
	  cudaSetDevice(0);
          cudaMemcpy(a_1_2, b_2_1, buff_size, cudaMemcpyHostToDevice);
	  z_copy_to_1<<< halo_G, halo_B >>>(1, a_1_2, c_T1, i-1, block, height, bot+1);
	  
	  cudaSetDevice(1);
          cudaMemcpy(a_2_1, b_1_2, buff_size, cudaMemcpyHostToDevice); 
	  z_copy_to_2<<< halo_G, halo_B >>>(1, a_2_1, c_T2, i-1, block, height, bot+1);	    
	}

      
      for(i  = _ny; i > 1; i--)
	{
	  // computation 
          cudaSetDevice(0);
	  function_z_down<<< comp_G, comp_B >>>(c_T1, i, dz, dy, DX, 
						DXYP, DYZP, DXZP, F, -_axx, -_ayy, -_azz, height);
	  z_copy_from_1<<< halo_G, halo_B >>>(-1, a_1_2, c_T1, i-1, block, height, bot);

          
	  cudaSetDevice(1);
          function_z_down<<< comp_G, comp_B >>>(c_T2, i, dz, dy, DX, 
						DXYP, DYZP, DXZP, F, -_axx, -_ayy, -_azz, height);
	  z_copy_from_2<<< halo_G, halo_B >>>(-1, a_2_1, c_T2, i-1, block, height, top,  bot);

	
	  // boundary layer copy to CPU
	  cudaSetDevice(0);
	  cudaMemcpy(b_1_2, a_1_2, buff_size, cudaMemcpyDeviceToHost);  
	  
	  cudaSetDevice(1);
	  cudaMemcpy(b_2_1, a_2_1, buff_size, cudaMemcpyDeviceToHost); 
	  
	    
	  // copied boundary layer back GPU
	  cudaSetDevice(0);
          cudaMemcpy(a_1_2, b_2_1, buff_size, cudaMemcpyHostToDevice);  
	  z_copy_to_1<<< halo_G, halo_B >>>(-1, a_1_2, c_T1, i-1, block, height, bot+1);
	  
	  
	  cudaSetDevice(1);
          cudaMemcpy(a_2_1, b_1_2, buff_size, cudaMemcpyHostToDevice);
	  z_copy_to_2<<< halo_G, halo_B >>>(-1, a_2_1, c_T2, i-1, block, height, bot+1);    
	}
      

      cudaSetDevice(0);
      from_k_to_y_direction_T1<<< data_G, data_B >>>(c_T1, c_T1_1, c_T1_2, 
						     block, height, width);
         
      cudaSetDevice(1);
      from_k_to_y_direction_T2<<< data_G, data_B >>>(c_T2, c_T2_1, c_T2_2, 
						     block, height, width);
       
      cudaSetDevice(0);
      cudaMemcpy(d_1_2, c_T1_2, data_size, cudaMemcpyDeviceToHost);
      
      cudaSetDevice(1);
      cudaMemcpy(d_2_1, c_T2_1, data_size, cudaMemcpyDeviceToHost);
      
      
      cudaSetDevice(0);
      cudaMemcpy(c_T1_2, d_2_1, data_size, cudaMemcpyHostToDevice);
      
      cudaSetDevice(1);
      cudaMemcpy(c_T2_1, d_1_2, data_size, cudaMemcpyHostToDevice);
      
       
      cudaSetDevice(0);
      to_k_to_y_direction_T1<<< data_G, data_B >>>(c_T1, c_T1_1, c_T1_2, 
						   block, height, width);
      
      cudaSetDevice(1);
      to_k_to_y_direction_T2<<< data_G, data_B >>>(c_T2, c_T2_1, c_T2_2, 
						   block, height, width);
      
      for(i = _ny; i > 1; i--)
	{
	  // computation 
	  cudaSetDevice(0);
	  function_y_up<<< comp_G, comp_B >>>(c_T1, i,  dy, DX, dz, 
					      DXZP, DXYP, DYZP, F, -_azz, -_axx, -_ayy, height);
	  y_copy_from_1<<< halo_G, halo_B >>>(-1, a_1_2, c_T1, i-1, 
					      block, height, bot);	
          
	  cudaSetDevice(1);
          function_y_up<<< comp_G, comp_B >>>(c_T2, i,  dy, DX, dz, 
					      DXZP, DXYP, DYZP, F, -_azz, -_axx, -_ayy, height);
	  y_copy_from_2<<< halo_G, halo_B >>>(-1, a_2_1, c_T2, i-1, 
					      block, height, top,  bot);
	    
	  // boundary layer copy to CPU
	  cudaSetDevice(0);  
	  cudaMemcpy(b_1_2, a_1_2, buff_size, cudaMemcpyDeviceToHost);  

	  cudaSetDevice(1);  
	  cudaMemcpy(b_2_1, a_2_1, buff_size, cudaMemcpyDeviceToHost);  
	  
	  // copied boundary layer back GPU	 
	  cudaSetDevice(0);
          cudaMemcpy(a_1_2, b_2_1, buff_size, cudaMemcpyHostToDevice);  
	  y_copy_to_1<<< halo_G, halo_B >>>(-1, a_1_2, c_T1, i-1, 
					    block, height, bot+1);
	 	  
	  cudaSetDevice(1);
          cudaMemcpy(a_2_1, b_1_2, buff_size, cudaMemcpyHostToDevice);  
	  y_copy_to_2<<< halo_G, halo_B >>>(-1, a_2_1, c_T2, i-1, 
					    block, height, bot+1);	    
	}
      

      for(i = 1; i < _ny; i++)
	{
	  // computation  
	  cudaSetDevice(0);
	  function_y_down<<< comp_G, comp_B  >>>(c_T1, i, dy, DX, dz, 
						 DXZP, DXYP, DYZP, F, _azz, _axx, _ayy, height);
	  y_copy_from_1<<< halo_G, halo_B >>>(1, a_1_2, c_T1, i-1, block, height, bot);

          
	  cudaSetDevice(1);
          function_y_down<<< comp_G, comp_B  >>>(c_T2, i, dy, DX, dz, 
						 DXZP, DXYP, DYZP, F, _azz, _axx, _ayy, height);
	  y_copy_from_2<<< halo_G, halo_B >>>(1, a_2_1, c_T2, i-1, block, height, top,  bot);

	  	    
          // boundary layer copy to CPU
	  cudaSetDevice(0);	  
	  cudaMemcpy(b_1_2, a_1_2, buff_size, cudaMemcpyDeviceToHost);  
	  
	  cudaSetDevice(1);  
	  cudaMemcpy(b_2_1, a_2_1, buff_size, cudaMemcpyDeviceToHost);  
	  
	  
	  // copied boundary layer back GPU 	 
	  cudaSetDevice(0);
          cudaMemcpy(a_1_2, b_2_1, buff_size, cudaMemcpyHostToDevice); 
	  y_copy_to_1<<< halo_G, halo_B >>>(1, a_1_2, c_T1, i-1, block, height, bot+1);
	  
       
	  cudaSetDevice(1);
          cudaMemcpy(a_2_1, b_1_2, buff_size, cudaMemcpyHostToDevice);  
	  y_copy_to_2<<< halo_G, halo_B >>>(1, a_2_1, c_T2, i-1, block, height, bot+1);
	    
	}

   
      cudaSetDevice(0);
      from_y_to_k_direction_T11<<< data_G, data_B >>>(c_T1, c_T1_1, c_T1_2, 
						      block, height, width);
      cudaSetDevice(1);
      from_y_to_k_direction_T21<<< data_G, data_B >>>(c_T2, c_T2_1, c_T2_2, 
						      block, height, width);
      
      cudaSetDevice(0);
      cudaMemcpy(d_1_2, c_T1_2, data_size, cudaMemcpyDeviceToHost);
      
      cudaSetDevice(1);
      cudaMemcpy(d_2_1, c_T2_1, data_size, cudaMemcpyDeviceToHost);
      
     
      cudaSetDevice(0);
      cudaMemcpy(c_T1_2, d_2_1, data_size, cudaMemcpyHostToDevice);
      
      cudaSetDevice(1);
      cudaMemcpy(c_T2_1, d_1_2, data_size, cudaMemcpyHostToDevice);
      
      
      cudaSetDevice(0);
      to_y_to_k_direction_T11<<< data_G, data_B >>>(c_T1, c_T1_1, c_T1_2, 
						    block, height, width);
          
      cudaSetDevice(1);
      to_y_to_k_direction_T21<<< data_G, data_B >>>(c_T2, c_T2_1, c_T2_2, 
						    block, height, width);
    }


  // end of the total time
  compute_timer += omp_get_wtime();
  //("total time is %lf sec\n",compute_timer);

  // copy back the computed solutioin in GPU to CPU
  cudaSetDevice(0);
  cudaMemcpy(T1, c_T1, size, cudaMemcpyDeviceToHost);
  
  cudaSetDevice(1);
  cudaMemcpy(T2, c_T2, size, cudaMemcpyDeviceToHost);

  // free the GPU memeory and boudary value buffer
  cudaFree(c_T1);  cudaFree(a_1_2);
  cudaFree(c_T2);  cudaFree(a_2_1); 

  // free the GPU partial data transfer Buffer
  cudaFree(c_T2_1);   cudaFree(c_T1_1);
  cudaFree(c_T2_2);   cudaFree(c_T1_2);

  // free the GPU device after the computation 
  for (int i = 0; i < 2; ++i)
    {
      cudaSetDevice (i);
      cudaDeviceReset ();
    }
  
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
  init = fopen("an_iso_2_gpu.txt","w");  
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
  printf(" Solution is written in the file------------------------ an_iso_2_gpu.txt\n");
#endif
  printf("*******************************************************************************************\n");

  return 0;
}






