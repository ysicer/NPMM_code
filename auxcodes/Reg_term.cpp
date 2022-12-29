#include <mex.h>
#define cimg_display 0
#define cimg_plugin "/Volumes/HD/Users/liuj/Research/DevTools/CImg/CImg-1.3.4/plugins/cimgmatlab.h"
#include "/Volumes/HD/Users/liuj/Research/DevTools/CImg/CImg-1.3.4/CImg.h"
//#define cimg_plugin "/media/Study/C&C++/CImg/CImg-1.3.4/CImg-1.3.4/plugins/cimgmatlab.h"
//#include "/media/Study/C&C++/CImg/CImg-1.3.4/CImg-1.3.4/CImg.h"
//#define cimg_plugin "/media/Study/C&C++/CImg/CImg-1.3.4/CImg-1.3.4/plugins/cimgmatlab.h"
//#include"/media/Study/C&C++/CImg/CImg-1.3.4/CImg-1.3.4/CImg.h"
using namespace cimg_library;
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

       CImg<> U(prhs[0],true);
       CImg<> pfVecParameters(prhs[1],true);
	   int wsize= (int) pfVecParameters[0];
	  // int K= U.spectrum();//U.display();
       const int wsizeleft= -wsize/2,wsizeright=(wsize+1)/2;
       const int Nx=U.width(),Ny=U.height(),Nc=U.spectrum();
      // const int K=Nc;
	   CImg<> R(Nx,Ny,Nc);
	//Nz=img.depth(),Nc=img.spectrum()


		//Update Voronoi areas U.
		cimg_forXYC(U,x,y,c)
		{
			 float dist=0.0f;
			   for (int iy=wsizeleft;iy<wsizeright;iy++ )
			    for (int ix=wsizeleft;ix<wsizeright;ix++ )
				  if ((float)(ix*ix+iy*iy)<(float)(wsize*wsize)/4.0f)
				   {
                     int Ix=x+ix,Iy=y+iy;
					 if (Ix<0) Ix=-Ix; else if (Ix>Nx-1) Ix= 2*(Nx-1)-Ix;
					 if (Iy<0) Iy=-Iy; else if (Iy>Ny-1) Iy= 2*(Ny-1)-Iy;
					//if (U(Ix,Iy)!=m)
                     dist+=(1.0f-U(Ix,Iy,0,c));

				   }
               R(x,y,c)=dist;
        }

    plhs[0]=R.toMatlab();
}
