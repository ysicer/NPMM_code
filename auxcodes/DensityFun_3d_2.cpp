#include <mex.h>
#define cimg_display 0
#define PI 3.1415926
#define cimg_plugin "cimgmatlab.h"
#include "CImg.h"
//Original by Jun Liu, Modified by Shi Yan.
using namespace cimg_library;
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //DensityFun(I1,I2,I3,U,h)
    //I1 is M*N, first victor of Img
    //I2 is M*N, second victor of Img
    //I3 is M*N, third victor of Img
    //U is current segmentation
    //h is parameter in NPMM.
    //X123 is for parameter size, normally 256*256*256.
    CImg<> I1(prhs[0],true);
    CImg<> I2(prhs[1],true);
    CImg<> I3(prhs[2],true);
    CImg<> U(prhs[3],true);
    CImg<> pfVecParameters(prhs[4],true);
    
    float h= (float) pfVecParameters[0];
    const int Ix=I1.width(),Iy=I1.height();
    //const int Xx=X1.width(),Xy=X1.height();
    //const int Rx=X123.width(),Ry=X123.height(),Rz = X123.spectrum();
    float temp1,temp2,temp3;
    
    CImg<> R(Ix,Iy);
    temp1=0.0f;
    cimg_forXY(U,x,y)
    {
        temp1+=U(x,y);
    }
    
   
    cimg_forXY(I1,x1,y1)
    {
        temp3=0.0f;
        cimg_forXY(I1,x2,y2)
        {
            temp2 = (I1(x2,y2)-I1(x1,y1))*(I1(x2,y2)-I1(x1,y1)) + (I2(x2,y2)-I2(x1,y1))*(I2(x2,y2)-I2(x1,y1)) + (I3(x2,y2)-I3(x1,y1))*(I3(x2,y2)-I3(x1,y1));
            temp3+=exp(-temp2/h/h)*U(x2,y2);
            
        }
        //z5mexPrintf("%d",z1);
        R(x1,y1)=temp3/(temp1);
                //back_up R(x1,y1)=temp3/(sqrt(PI)*h*temp1);
    }

    mexPrintf("Finish Running!\n");

    
    plhs[0]=R.toMatlab();
    
}
