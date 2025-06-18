#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include "Fold3dPMM.h"

_DOUBLE_ InitFoldFromPoint(_DOUBLE_ F, _DOUBLE_ ax,_DOUBLE_ ay,
			   _DOUBLE_ az, _DOUBLE_ x, _DOUBLE_ y,
			   _DOUBLE_ z)
{
  _DOUBLE_ xsq = x*x + y*y + z*z;
  _DOUBLE_ adotx = ax*x + ay*y + az*z;
  _DOUBLE_ sqrp = (F*F - (ax*ax + ay*ay + az*az)) 
    * xsq + adotx*adotx;
  if (sqrp > 0.0)
    {
      sqrp = adotx + sqrt(sqrp);
      if(sqrp > 0.0)
	{
	  return xsq/sqrp;
	}
    }
  return 1.0*MAXT;
}


_DOUBLE_ ImpSurfF(_DOUBLE_ x, _DOUBLE_ y, _DOUBLE_ z)
{
  return 5.0 + (sin(x) + cos(y+x) + sin(x+y+z)) * 0.4 - z + 
    (x-5.0) * (y-5.0) * (z-5) / 55.0;
}

void InitNode(_DOUBLE_ x, _DOUBLE_ y, _DOUBLE_ z, 
	      int i, int j, int k, _DOUBLE_* T)
{
  _DOUBLE_ F, ax, ay, az;
  ax = _axx;
  ay = _ayy;
  az = _azz;
  F = _Fc;
  if(T[i + HEIGHT * (j + HEIGHT * k)] < 0.0) 
    {
      ax *= -1;	
      ay *= -1;	
      az *= -1;
    }
  _DOUBLE_ tnew = InitFoldFromPoint(F, ax, ay, az, -x, -y, -z);
  if(tnew < fabs(T[ i + HEIGHT * (j + HEIGHT* k)])) 
    {
      T[i + HEIGHT * (j + HEIGHT * k)] =(sign(T[i + HEIGHT * (j + HEIGHT * k)]) * 1.0) * tnew;
    }
}

void updateVoxel(_DOUBLE_* T, int i, int j, int k, 
		 _DOUBLE_ xp, _DOUBLE_ yp, _DOUBLE_ zp, 
		 _DOUBLE_ dxx, _DOUBLE_ dyy, _DOUBLE_ dzz)
{
  
  //first all nodes in the voxel
  InitNode(xp, yp, zp, i, j, k, T);//, SIGN);
  InitNode(xp-dxx, yp, zp, i+1, j, k, T);//, SIGN);
  InitNode(xp, yp-dyy, zp, i, j+1, k, T);//, SIGN);
  InitNode(xp, yp, zp-dzz, i, j, k+1, T);//, SIGN);
  InitNode(xp-dxx, yp-dyy, zp, i+1, j+1, k, T);//, SIGN);
  InitNode(xp-dxx, yp, zp-dzz, i+1, j, k+1, T);//, SIGN);
  InitNode(xp, yp-dyy, zp-dzz, i, j+1, k+1, T);//, SIGN);
  InitNode(xp-dxx, yp-dyy, zp-dzz, i+1, j+1, k+1, T);//, SIGN);
  //then the closer behind+infront
  //return;
  InitNode(xp+dxx, yp, zp, i-1, j, k, T);//, SIGN);
  InitNode(xp+dxx, yp-dyy, zp, i-1, j+1, k, T);//, SIGN);
  InitNode(xp+dxx, yp,zp-dzz, i-1, j, k+1, T);//, SIGN);
  InitNode(xp+dxx, yp-dyy, zp-dzz, i-1, j+1, k+1, T);//, SIGN);
  InitNode(xp-2*dxx, yp, zp, i+2, j, k, T);//, SIGN);
  InitNode(xp-2*dxx, yp-dyy, zp, i+2, j+1,k, T);//, SIGN);
  InitNode(xp-2*dxx, yp, zp-dzz, i+2, j, k+1, T);//, SIGN);
  InitNode(xp-2*dxx, yp-dyy, zp-dzz, i+2, j+1, k+1, T);//, SIGN);
  //+above, below
  InitNode(xp, yp+dyy, zp, i, j-1, k, T);//, SIGN);
  InitNode(xp-dxx, yp+dyy, zp, i+1, j, k, T);//, SIGN);
  InitNode(xp, yp+dyy, zp-dzz, i, j-1, k+1, T);//, SIGN);
  InitNode(xp-dxx, yp+dyy, zp-dzz, i+1, j-1, k+1, T);//, SIGN);
  InitNode(xp, yp-2*dyy, zp, i, j+2, k, T);//, SIGN);
  InitNode(xp-dxx, yp-2*dyy, zp, i+2, j, k, T);//, SIGN);
  InitNode(xp, yp-2*dyy, zp-dzz, i, j+2, k+1, T);//, SIGN);
  InitNode(xp-dxx, yp-2*dyy, zp-dzz, i+1, j+2, k+1, T);//, SIGN);
  //+left, right
  InitNode(xp, yp, zp+dzz, i, j, k-1, T);//, SIGN);
  InitNode(xp-dxx, yp, zp+dzz, i+1, j, k-1, T);//, SIGN);
  InitNode(xp, yp-dyy, zp+dzz, i, j+1, k-1, T);//, SIGN);
  InitNode(xp-dxx, yp-dyy, zp+dzz, i+1, j+1, k-1, T);//, SIGN);
  InitNode(xp, yp, zp-2*dzz, i, j, k+2, T);//, SIGN);
  InitNode(xp-dxx, yp, zp-2*dzz, i+1, j, k+2, T);//, SIGN);
  InitNode(xp, yp-dyy, zp-2*dzz, i, j+1, k+2, T);//, SIGN);
  InitNode(xp-dxx, yp-dyy, zp-2*dzz, i+1, j+1, k+2, T);//, SIGN);
  // behind infront and above etc
  InitNode(xp, yp+dyy, zp+dzz, i, j-1, k-1, T);//, SIGN);
  InitNode(xp+dxx, yp, zp+dzz, i-1, j, k-1, T);//, SIGN);
  InitNode(xp+dxx, yp+dyy, zp, i-1, j-1, k, T);//, SIGN);
  InitNode(xp, yp-2*dyy, zp-2*dzz, i, j+2, k+2, T);//, SIGN);
  InitNode(xp-2*dxx, yp, zp-2*dzz, i+2, j, k+2, T);//, SIGN);
  InitNode(xp-2*dxx,yp-2*dyy, zp, i+2, j+2, k, T);//, SIGN);
  InitNode(xp-dxx, yp+dyy, zp+dzz, i+1, j-1, k-1, T);//, SIGN);
  InitNode(xp+dxx, yp-dyy, zp+dzz, i-1, j+1, k-1, T);//, SIGN);
  InitNode(xp+dxx, yp+dyy, zp-dzz, i-1, j-1, k+1, T);//, SIGN);
  InitNode(xp-dxx, yp-2*dyy, zp-2*dzz, i+1, j+2, k+2, T);//, SIGN);
  InitNode(xp-2*dxx, yp-dyy, zp-2*dzz, i+2, j+1, k+2, T);//, SIGN);
  InitNode(xp-2*dxx, yp-2*dyy, zp-dzz, i+2, j+2, k+1, T);//, SIGN);

}


_DOUBLE_ x(int i,_DOUBLE_ DX)
{
  return (i-1) * DX+_xmin;
}
_DOUBLE_ y(int j, _DOUBLE_ dy)
{
  return (j-1) * dy+_ymin;
}
_DOUBLE_ z(int k, _DOUBLE_ dz)
{
  return (k-1) * dz+_zmin;
}


void ImplicitInitialiser(_DOUBLE_* T, int nx, int ny, int nz, 
			 _DOUBLE_ dxx, _DOUBLE_ dyy, _DOUBLE_ dzz)
{
  _DOUBLE_ *IF = (_DOUBLE_*)malloc(sizeof(_DOUBLE_)*(nx+2)*(ny+2)*(nz+2));
  int i, j, k;
  _DOUBLE_ xp, yp, zp;	
  _DOUBLE_ f, fx, fy, fz, fzx, fzy, fxy, fxyz;
  
  for(i = 0; i < nx+2; i++)
    {
      for(j = 0; j < ny+2; j++)
	{
	  for(k = 0; k < nz+2; k++)
	    {
	      IF[i + HEIGHT * (j + HEIGHT * k)] = ImpSurfF(x(i,dxx), y(j,dyy), z(k,dzz));
	      T[i + HEIGHT * (j + HEIGHT * k)] = (sign(IF[ i + HEIGHT * (j + HEIGHT * k)]) * 1.0)*MAXT;
	    }
	}
    }
  
  //NEXT INITIALISE USING MARCHING CUBE LIKE APPROACH
  int np = 8; //number of source points in current voxel
  _DOUBLE_ xpos[8];
  _DOUBLE_ ypos[8];
  _DOUBLE_ zpos[8];
  
  //execute in parallel on GPU ? check if possible
  //#pragma omp for schedule(static,vin) collapse(3)
  for(i = 1; i < nx; i++) 
    {
      for(j = 1; j < ny;j++) 
	{
	  for(k = 1; k < nz; k++) 
	    {
	      np = 0; 
	      f 	= IF[i + HEIGHT * (j + HEIGHT * k)];
	      fx 	= IF[(i+1) + HEIGHT * (j + HEIGHT * k)];
	      fy 	= IF[i + HEIGHT * ((j+1) + HEIGHT * k)];
	      fz 	= IF[i + HEIGHT * (j + HEIGHT * (k+1))];
	      fxy = IF[(i+1) + HEIGHT * ((j+1) + HEIGHT* k)];
	      fzx = IF[(i+1) + HEIGHT * (j + HEIGHT* (k+1))];
	      fzy	= IF[i + HEIGHT*((j+1) + HEIGHT* (k+1))];
	      fxyz= IF[(i+1) + HEIGHT* ((j+1) + HEIGHT* (k+1))];
	      
	      xp = dxx*f / (f-fx);
	      if(0.0 <= xp && xp <= dxx)
		{
		  xpos[np] = xp;
		  ypos[np] = 0.0;
		  zpos[np] = 0.0;
		  np += 1;
		}
	      xp = dxx*fz / (fz-fzx);
	      if(0.0 <= xp && xp <= dxx)
		{
		  xpos[np] = xp;
		  ypos[np] = 0.0;
		  zpos[np] = dzz;
		  np += 1;
		}
	      xp = dxx*fy / (fy-fxy);
	      if(0.0 <= xp && xp <= dxx)
		{
		  xpos[np] = xp;
		  ypos[np] = dyy;
		  zpos[np] = 0.0;
		  np += 1;
		}
	      xp = dxx*fzy / (fzy-fxyz);
	      if(0.0 <= xp && xp <= dxx)
		{
		  xpos[np] = xp;
		  ypos[np] = dyy;
		  zpos[np] = dzz;
		  np += 1;
		}
	      yp = dyy*f / (f-fy);
	      if(0.0 <= yp && yp <= dyy)
		{
		  xpos[np] = 0.0;
		  ypos[np] = yp;
		  zpos[np] = 0.0;
		  np += 1;
		}
	      yp = dyy*fz / (fz-fzy);
	      if(0.0 <= yp && yp <= dyy)
		{
		  xpos[np] = 0.;
		  ypos[np] = yp;
		  zpos[np] = dzz;
		  np += 1;
		}
	      yp = dyy*fx / (fx-fxy);
	      if(0.0 <= yp && yp <= dyy)
		{
		  xpos[np] = dxx;
		  ypos[np] = yp;
		  zpos[np] = 0.0;
		  np += 1;
		}
	      yp = dyy*fzx / (fzx-fxyz);
	      if(0.0 <= yp && yp <= dyy)
		{
		  xpos[np] = dxx;
		  ypos[np] = yp;
		  zpos[np] = dzz;
		  np += 1;
		}
	      
	      zp = dzz*f / (f-fz);
	      if(0.0 <= zp && zp <= dzz)
		{
		  xpos[np] = 0.0;
		  ypos[np] = 0.0;
		  zpos[np] = zp;
		  np += 1;
		}
	      zp = dzz*fxy / (fxy-fxyz);
	      if(0.0 <= zp && zp <= dzz)
		{
		  xpos[np] = dxx;
		  ypos[np] = dyy;
		  zpos[np] = zp;
		  np += 1;
		}
	      zp = dzz*fx / (fx-fzx);
	      if(0.0 <= zp && zp <= dzz)
		{
		  xpos[np] = dxx;
		  ypos[np] = 0.0;
		  zpos[np] = zp;
		  np += 1;
		}
	      zp = dzz*fy / (fy-fzy);
	      if(0.0 <= zp && zp <= dzz)
		{
		  xpos[np] = 0.0;
		  ypos[np] = dyy;
		  zpos[np] = zp;
		  np += 1;
		}
	      int l;
	      for(l = 0; l < np; l += 1 )
		{
		  updateVoxel(T, i, j, k, xpos[l], ypos[l], zpos[l], dxx, dyy, dzz);
		  if(l > 0)
		    {
		      xp = (2*xpos[l] + xpos[l-1]) / 3;
		      yp = (2*ypos[l] + ypos[l-1]) / 3;
		      zp = (2*zpos[l] + zpos[l-1]) / 3;
		      updateVoxel(T, i, j, k, xp, yp, zp, dxx, dyy, dzz);
		      xp = (xpos[l] + 2*xpos[l-1]) / 3;
		      yp = (ypos[l] + 2*ypos[l-1]) / 3;
		      zp = (zpos[l] + 2*zpos[l-1]) / 3;
		      updateVoxel(T, i, j, k, xp, yp, zp, dxx, dyy, dzz);
		    }
		  if(l > 1)
		    {
		      xp = (2*xpos[l] + xpos[l-2]) / 3;
		      yp = (2*ypos[l] + ypos[l-2]) / 3;
		      zp = (2*zpos[l] + zpos[l-2]) / 3;
		      updateVoxel(T, i, j, k, xp, yp, zp, dxx, dyy, dzz);
		      xp = (xpos[l] + 2*xpos[l-2]) / 3;
		      yp = (ypos[l] + 2*ypos[l-2]) / 3;
		      zp = (zpos[l] + 2*zpos[l-2]) / 3;
		      updateVoxel(T, i, j, k, xp, yp, zp, dxx, dyy, dzz);
		    }
		  if(l > 2)
		    {
		      xp = (2*xpos[l] + xpos[l-3]) / 3;
		      yp = (2*ypos[l] + ypos[l-3]) / 3;
		      zp = (2*zpos[l] + zpos[l-3]) / 3;
		      updateVoxel(T, i, j, k, xp, yp, zp, dxx, dyy, dzz);
		      xp = (xpos[l] + 2*xpos[l-3]) / 3;
		      yp = (ypos[l] + 2*ypos[l-3]) / 3;
		      zp = (zpos[l] + 2*zpos[l-3]) / 3;
		      updateVoxel(T, i, j, k, xp, yp, zp, dxx, dyy, dzz);
		     
		      xp = (xpos[l] + xpos[l-2] + xpos[l-3]) / 3;
		      yp = (ypos[l] + ypos[l-2] + ypos[l-3]) / 3;
		      zp = (zpos[l] + ypos[l-2] + zpos[l-3]) / 3;
		      updateVoxel(T, i, j, k, xp, yp, zp, dxx, dyy, dzz);
		      xp = (2*xpos[l] + xpos[l-2] + xpos[l-3]) / 4;
		      yp = (2*ypos[l] + ypos[l-2] + ypos[l-3]) / 4;
		      zp = (2*zpos[l] + ypos[l-2] + zpos[l-3]) / 4;
		      updateVoxel(T, i, j, k, xp, yp, zp, dxx, dyy, dzz);
		      xp = (xpos[l] + 2*xpos[l-2] + xpos[l-3]) / 4;
		      yp = (ypos[l] + 2*ypos[l-2] + ypos[l-3]) / 4;
		      zp = (zpos[l] + 2*ypos[l-2] + zpos[l-3]) /4;
		      updateVoxel(T, i, j, k, xp, yp, zp, dxx, dyy, dzz);
		      xp = (xpos[l] + xpos[l-2] + 2*xpos[l-3]) / 4;
		      yp = (ypos[l] + ypos[l-2] + 2*ypos[l-3]) / 4;
		      zp = (zpos[l] + ypos[l-2] + 2*zpos[l-3]) / 4;
		      updateVoxel(T, i, j, k, xp, yp, zp, dxx, dyy, dzz);
		     
		    }
		  if(l > 3)
		    {
		      xp = (2*xpos[l] + xpos[l-4]) / 3;
		      yp = (2*ypos[l] + ypos[l-4]) / 3;
		      zp = (2*zpos[l] + zpos[l-4]) / 3;
		      updateVoxel(T, i, j, k, xp, yp, zp, dxx, dyy, dzz);
		      xp = (xpos[l] + 2*xpos[l-4]) / 3;
		      yp = (ypos[l] + 2*ypos[l-4]) / 3;
		      zp = (zpos[l] + 2*zpos[l-4]) / 3;
		      updateVoxel(T, i, j, k, xp, yp, zp, dxx, dyy, dzz);
		     
		      xp = (xpos[l] + xpos[l-2] + xpos[l-3]) / 3;
		      yp = (ypos[l] + ypos[l-2] + ypos[l-3]) / 3;
		      zp = (zpos[l] + ypos[l-2] + zpos[l-3]) / 3;
		      updateVoxel(T, i, j, k, xp, yp, zp, dxx, dyy, dzz);
		      xp = (xpos[l] + xpos[l-2] + xpos[l-4]) / 3;
		      yp = (ypos[l] + ypos[l-2] + ypos[l-4]) / 3;
		      zp = (zpos[l] + ypos[l-2] + zpos[l-4]) / 3;
		      updateVoxel(T, i, j, k, xp, yp, zp, dxx, dyy, dzz);
		      xp = (xpos[l] + xpos[l-3] + xpos[l-4]) / 3;
		      yp = (ypos[l] + ypos[l-3] + ypos[l-4]) / 3;
		      zp = (zpos[l] + ypos[l-3] + zpos[l-4]) / 3;
		      updateVoxel(T, i, j, k, xp, yp, zp, dxx, dyy, dzz);
		     
		    }
		  if(l > 4)
		    {
		      xp = (2*xpos[l] + xpos[l-5]) / 3;
		      yp = (2*ypos[l] + ypos[l-5]) / 3;
		      zp = (2*zpos[l] + zpos[l-5]) / 3;
		      updateVoxel(T, i, j, k, xp, yp, zp, dxx, dyy, dzz);
		      xp = (xpos[l] + 2*xpos[l-5]) / 3;
		      yp = (ypos[l] + 2*ypos[l-5]) / 3;
		      zp = (zpos[l] + 2*zpos[l-5]) / 3;
		      updateVoxel(T, i, j, k, xp, yp, zp, dxx, dyy, dzz);
		     
		      xp = (xpos[l] + xpos[l-2] + xpos[l-3]) / 3;
		      yp = (ypos[l] + ypos[l-2] + ypos[l-3]) / 3;
		      zp = (zpos[l] + ypos[l-2] + zpos[l-3]) / 3;
		      updateVoxel(T, i, j, k, xp, yp, zp, dxx, dyy, dzz);
		      xp = (xpos[l] + xpos[l-2] + xpos[l-4]) / 3;
		      yp = (ypos[l] + ypos[l-2] + ypos[l-4]) / 3;
		      zp = (zpos[l] + ypos[l-2] + zpos[l-4]) / 3;
		      updateVoxel(T, i, j, k, xp, yp, zp, dxx, dyy, dzz);
		      xp = (xpos[l] + xpos[l-2] + xpos[l-5]) / 3;
		      yp = (ypos[l] + ypos[l-2] + ypos[l-5]) / 3;
		      zp = (zpos[l] + ypos[l-2] + zpos[l-5]) / 3;
		      updateVoxel(T, i, j, k, xp, yp, zp, dxx, dyy, dzz);
		      xp = (xpos[l] + xpos[l-3] + xpos[l-5]) / 3;
		      yp = (ypos[l] + ypos[l-3] + ypos[l-5]) / 3;
		      zp = (zpos[l] + ypos[l-3] + zpos[l-5]) / 3;
		      updateVoxel(T, i, j, k, xp, yp, zp, dxx, dyy, dzz);
		      xp = (xpos[l] + xpos[l-3] + xpos[l-4]) / 3;
		      yp = (ypos[l] + ypos[l-3] + ypos[l-4]) / 3;
		      zp = (zpos[l] + ypos[l-3] + zpos[l-4]) / 3;
		      updateVoxel(T, i, j, k, xp, yp, zp, dxx, dyy, dzz);
		      
		    }
		}
	      
	    }
	}
    }
  free(IF);
}
