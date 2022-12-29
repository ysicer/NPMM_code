#include <mex.h>
#define cimg_display 0
#define PI 3.1415926
#define cimg_plugin "cimgmatlab.h"
#include "CImg.h"
//Original by Jun Liu, Modified by Shi Yan.
using namespace cimg_library;
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    CImg<> I(prhs[0],true);
    CImg<> X(prhs[1],true);
    CImg<> U(prhs[2],true);
    CImg<> pfVecParameters(prhs[3],true);
    
    float h= (float) pfVecParameters[0];
    const int Ix=I.width(),Iy=I.height();
    const int Xx=X.width(),Xy=X.height();
    
    
    CImg<> R(Xx,Xy);
    
    float temp1,temp2,temp3;
    
    
	//   int wsize= (int) pfVecParameters[0];
	  // int K= U.spectrum();//U.display();
    //   const int wsizeleft= -wsize/2,wsizeright=(wsize+1)/2;
    
      // const int K=Nc;
   //	   CImg<> R(Nx,Ny,Nc);
	//Nz=img.depth(),Nc=img.spectrum()
    temp1=0.0f;
    cimg_forXY(U,x,y)
    {
        temp1+=U(x,y);
    
    }
    
    
    cimg_forXY(X,x1,y1)
    {
        
        temp3=0.0f;
        cimg_forXY(I,x2,y2)
        {
            temp2=I(x2,y2)-X(x1,y1);
            temp3+=exp(-(temp2)*(temp2)/(h*h))*U(x2,y2);
        }
        R(x1,y1)=temp3/(sqrt(PI)*h*temp1);
    }
    plhs[0]=R.toMatlab();
    
}