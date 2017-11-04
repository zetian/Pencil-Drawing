#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;

int main(){
    Mat image = imread("../../images/test_2.jpg", CV_LOAD_IMAGE_COLOR);
    if(! image.data )                             
    {
        std::cout <<  "Could not open or find the image" << std::endl;
        return -1;
    }
    

    cvtColor(image, image, COLOR_RGB2YCrCb);
    // image.convertTo(image, CV_64FC3);
    auto size = image.size();
    int num_channels = image.channels();

    vector<Mat> channels(3);
    split(image, channels);
    Mat lumIm = channels[0];

    std::cout << "Height: " << size.height << std::endl;
    std::cout << "Width: " << size.width << std::endl;
    std::cout << "Num of channels: " << num_channels << std::endl;

    auto size_2 = lumIm.size();
    int num_channels_2 = lumIm.channels();
    // GaussianBlur(lumIm, lumIm, Size(3, 3), 0, 0);
    // medianBlur(lumIm, lumIm, 3);

    std::cout << "Height: " << size_2.height << std::endl;
    std::cout << "Width: " << size_2.width << std::endl;
    std::cout << "Num of channels: " << num_channels_2 << std::endl;


    Mat imX,imY = lumIm;
    std::cout << "~~~~~~: " << lumIm.at<Vec3b>(3,3) << std::endl;
    // for(int j=0;j<lumIm.rows - 1;j++) 
    // {
    //   for (int i=0;i<lumIm.cols;i++)
    //   {   
    //     imX.at<uchar>(j,i) = std::abs(lumIm.at<uchar>(j,i) - lumIm.at<uchar>(j + 1,i)); //white
    //     imX.at<uchar>(lumIm.rows,i) = 0;
    //   }
      
    // }
    // Mat Dx,Dy;
    // Sobel(lumIm, Dx, CV_64F, 1, 0, 3);
    // Sobel(lumIm, Dy, CV_64F, 0, 1, 3);
    // // Laplacian(lumIm, lumIm, CV_64F);
    // // Canny( lumIm, lumIm, 20, 20*3, 3 );
    // Mat image_edge = Dx + Dy;


    imshow("Display window", imX);
    waitKey(0);
	return 0;
}

