#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;

enum NiblackVersion 
{
	NIBLACK=0,
    SAUVOLA,
    WOLFJOLION,
};

#define BINARIZEWOLF_VERSION	"2.4 (August 1st, 2014)"

#define uget(x,y)    at<unsigned char>(y,x)
#define uset(x,y,v)  at<unsigned char>(y,x)=v;
#define fget(x,y)    at<float>(y,x)
#define fset(x,y,v)  at<float>(y,x)=v;


// *************************************************************
// glide a window across the image and
// create two maps: mean and standard deviation.
//
// Version patched by Thibault Yohan (using opencv integral images)
// *************************************************************
double calcLocalStats (Mat &im, Mat &map_m, Mat &map_s, int winx, int winy);
/**********************************************************
 * The binarization routine
 **********************************************************/
void NiblackSauvolaWolfJolion (Mat im, Mat output, NiblackVersion version,
	int winx, int winy, double k, double dR);