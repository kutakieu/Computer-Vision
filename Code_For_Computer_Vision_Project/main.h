// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "binarizewolfjolion.h"
#include <string> 
#include <sstream>

using namespace std;
using namespace cv;

struct Comparator
{
    const vector<vector<Point> >  & value_vector;

    Comparator(const vector<vector<Point> >  & val_vec):
        value_vector(val_vec) {}

    bool operator()(int i1, int i2)
    {
    	Rect i1Rect = boundingRect(value_vector[i1]);
    	Rect i2Rect = boundingRect(value_vector[i2]);
        return i1Rect.x < i2Rect.x;
    }
};

int main(int argc, char *argv[]);