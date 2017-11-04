#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;

int main(){
    Mat image = imread("../../images/test_1.jpg", CV_LOAD_IMAGE_COLOR);
    if(! image.data )                             
    {
        std::cout <<  "Could not open or find the image" << std::endl;
        return -1;
    }
    imshow("Display window", image);
    waitKey(0);
	return 0;
}

