#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>

#include <iostream>
#include <stdio.h>
#include <algorithm>

using namespace cv;

int main(){
    Mat image = imread("../../images/test_2.jpg", CV_LOAD_IMAGE_COLOR);
    if(! image.data )                             
    {
        std::cout <<  "Could not open or find the image" << std::endl;
        return -1;
    }
    
    // imshow("Display window", image);
    // waitKey(0);
    cvtColor(image, image, COLOR_RGB2YCrCb);
    // image.convertTo(image, CV_64FC3);
    auto size = image.size();
    int num_channels = image.channels();
    
    std::vector<Mat> channels(3);
    split(image, channels);
    Mat lumIm_1 = channels[0];
    Mat lumIm_2 = channels[1];
    Mat lumIm_3 = channels[2];


    // Vec3b Y_pix = image.at<Vec3b>(0, 0);
    // int pixelval = Y_pix[0];
    // std::cout << "@@@@@@@@@@@: " << Y_pix << std::endl;
    
    // imshow("Display window", lumIm_1);
    // waitKey(0);

    // std::cout << "Height: " << size.height << std::endl;
    // std::cout << "Width: " << size.width << std::endl;
    // std::cout << "Num of channels: " << num_channels << std::endl;

    auto size_2 = lumIm_1.size();
    int num_channels_2 = lumIm_1.channels();
    // GaussianBlur(lumIm, lumIm, Size(3, 3), 0, 0);
    // medianBlur(lumIm, lumIm, 3);

    // std::cout << "Height: " << size_2.height << std::endl;
    // std::cout << "Width: " << size_2.width << std::endl;
    // std::cout << "Num of channels: " << num_channels_2 << std::endl;

    // Mat imX = lumIm_1, imY = lumIm_1;
    Mat imX, imY, im_XY;
    // Vec3f intensity = image.at<Vec3f>(100, 100);

    // Vec3f Y_pix = lumIm.at<Vec3f>(200, 100);
    // int pixelval = Y_pix[0];
    
    // std::cout << "~~~~~@@@~: " << imX.at<Vec3f>(0, 0)[0] << std::endl;
    double scale = 1;
    double delta = 0.0;
    // int ddepth = CV_16S;
    int ddepth = -1;
    Sobel(lumIm_1, imX, ddepth, 1, 0, 1, scale, delta, BORDER_DEFAULT);
    Sobel(lumIm_1, imY, ddepth, 0, 1, 1, scale, delta, BORDER_DEFAULT);
    // convertScaleAbs(imX, imX);
    // convertScaleAbs(imY, imY);
    addWeighted(imX, 0.5, imY, 0.5, 0, im_XY);


    int len_conv_lines = 8;
    int num_dir = 8;
    int width = 1;

    // for(int j=0;j<lumIm_1.rows;j++) 
    // {
    //   for (int i=0;i<lumIm_1.cols - 1;i++)
    //   {   
    //     imX.at<Vec3b>(j, i)[0] = 0; //white
    //     imX.at<Vec3b>(j, i)[1] = 0; 
    //     imX.at<Vec3b>(j, i)[2] = 0; 



    //     imX.at<Vec3b>(j, i)[0] = std::abs(lumIm_1.at<Vec3b>(j, i)[0] - lumIm_1.at<Vec3b>(j, i + 1)[0]); //white
    //     imX.at<Vec3b>(j, i)[1] = std::abs(lumIm_1.at<Vec3b>(j, i)[1] - lumIm_1.at<Vec3b>(j, i + 1)[1]); 
    //     imX.at<Vec3b>(j, i)[2] = std::abs(lumIm_1.at<Vec3b>(j, i)[2] - lumIm_1.at<Vec3b>(j, i + 1)[2]); 
    //     // std::cout << "~~~~~~: " << imX.at<Vec3f>(j, i)[0] << std::endl;
    //     // imX.at<Vec3b>(lumIm_1.rows,i)[0] = 0;
    //     // imX.at<Vec3b>(lumIm_1.rows,i)[1] = 0;
    //     // imX.at<Vec3b>(lumIm_1.rows,i)[2] = 0;
    //     std::cout << "@@@@@@@@@@@: " << imX.at<Vec3b>(j, i) << std::endl;
    //   }
    // //   std::cout << "~~~~~~: "  << std::endl;
    // }

    // Vec3b Y_pix = imX.at<Vec3b>(0, 0);
    // // int pixelval = Y_pix[0];
    // std::cout << "@@@@@@@@@@@: " << Y_pix << std::endl;

    imshow("Display window", im_XY);

    waitKey(0);
    // Mat Dx,Dy;
    // Sobel(lumIm, Dx, CV_64F, 1, 0, 3);
    // Sobel(lumIm, Dy, CV_64F, 0, 1, 3);
    // // Laplacian(lumIm, lumIm, CV_64F);
    // // Canny( lumIm, lumIm, 20, 20*3, 3 );
    // Mat image_edge = Dx + Dy;

    // imshow("Display window", imX);
    // waitKey(0);
    // imshow("Display window", image);
    // waitKey(0);
	return 0;
}

