#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>

#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <climits>
#include <cmath>

#include "../pencil_drawing/pencil_drawing.h"

using namespace cv;

enum ConvolutionType 
{
   /* Return the full convolution, including border */
  CONVOLUTION_FULL, 
  /* Return only the part that corresponds to the original image */
  CONVOLUTION_SAME,
  /* Return only the submatrix containing elements that were not influenced by the border */
  CONVOLUTION_VALID
};
void conv2(const Mat &img, const Mat& kernel, ConvolutionType type, Mat& dest) 
{
  Mat source = img;
  if(CONVOLUTION_FULL == type) 
  {
    source = Mat();
    
    const int additionalRows = kernel.rows-1, additionalCols = kernel.cols-1;
    copyMakeBorder(img, source, (additionalRows+1)/2, additionalRows/2, (additionalCols+1)/2, additionalCols/2, BORDER_CONSTANT, Scalar(0));
  }
  
  Point anchor(kernel.cols - kernel.cols/2 - 1, kernel.rows - kernel.rows/2 - 1);
  
  int borderMode = BORDER_CONSTANT;
  
  filter2D(source, dest, img.depth(), kernel, anchor, 0, borderMode);
  std::cout <<  "C~~~~~~~" << std::endl;
  if(CONVOLUTION_VALID == type) 
  {
    dest = dest.colRange((kernel.cols-1)/2, dest.cols - kernel.cols/2)
               .rowRange((kernel.rows-1)/2, dest.rows - kernel.rows/2);
  }
}


cv::Mat imRotate(const cv::Mat source, double angle) {
    cv::Mat dst;
    // Special Cases
    if (std::fmod(angle, 360.0) == 0.0)
        dst = source;
    else{
        cv::Point2f center(source.cols / 2.0F, source.rows / 2.0F);
        cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
        // determine bounding rectangle
        cv::Rect bbox = cv::RotatedRect(center, source.size(), angle).boundingRect();
        // adjust transformation matrix
        rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
        rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;
        cv::warpAffine(source, dst, rot, bbox.size(), cv::INTER_LINEAR);

    }
    return dst;
}

Mat rotateImage(const Mat source, double angle,int border=20)
{
    Mat bordered_source;
    int top,bottom,left,right;
    top=bottom=left=right=border;
    copyMakeBorder( source, bordered_source, top, bottom, left, right, BORDER_CONSTANT,cv::Scalar() );
    Point2f src_center(bordered_source.cols/2.0F, bordered_source.rows/2.0F);
    Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
    Mat dst;
    warpAffine(bordered_source, dst, rot_mat, bordered_source.size());  
    return dst;
}

int main(){
    Mat origin_image = imread("../../images/test_2.jpg", CV_LOAD_IMAGE_COLOR);
    if(! origin_image.data )                             
    {
        std::cout <<  "Could not open or find the image" << std::endl;
        return -1;
    }
    int rows = origin_image.rows;
    int cols = origin_image.cols;
    cvtColor(origin_image, origin_image, COLOR_RGB2YCrCb);
    PencelDrawing pencil;

    std::vector<Mat> channels(3);
    split(origin_image, channels);
    Mat grey = channels[0];

    Mat grad = pencil.getGrad(grey);

    int ks = 8;
    int num_dir = 8;
    int width = 1;
    float kern_data[(ks*2 + 1)*(ks*2 + 1)];
    for (int i = 0; i < ks*2 + 1; i++){
        for (int j = 0; j < ks*2 + 1; j++){
            // std::cout << "i: " << i << "j: " << j << std::endl;
            kern_data[i*(ks*2 + 1) + j] = 0;
            if (i == ks + 1){
                kern_data[i*(ks*2 + 1) + j] = 0.7;
            }
        }
    }
    // for (int i = 0; i < ks*2 + 1; i++){
    //     for (int j = 0; j < ks*2 + 1; j++){
    //         std::cout << kern_data[i*(ks*2 + 1) + j]  << " ";
    //     }
    //     std::cout << std::endl;
    // }


    Mat kern((ks*2 + 1), (ks*2 + 1), CV_32F, kern_data);
    // Mat test;
    // filter2D(grad, test, -1, kern, Point(-1,-1), 0, BORDER_DEFAULT);

    // conv2(grad, test, CONVOLUTION_FULL, kern);
    
    imshow("Display window", test);
    waitKey(0);
    return 0;
    std::vector<Mat> response;
    for(int i = 0; i < num_dir; i++){
        Mat kernel = imRotate(kern, i*180/num_dir);
        Mat dst;
        filter2D(grad, dst, grad.depth(), kernel);
        response.push_back(dst);
    }

    std::vector<std::vector<int> > index(rows, std::vector<int>(cols));

    for(int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            int index_temp = 0;
            int max = INT_MIN;
            for (int k = 0; k < response.size(); k++){
                int color = response[k].at<uchar>(i, j);
                if (color > max){
                    index_temp = k;
                    max = color;
                }
            }
            index[i][j] = index_temp;
        }
    }

    std::vector<Mat> C_image;
    for (int i = 0; i < num_dir; i++){
        Mat temp = Mat::zeros(rows, cols, CV_8UC1);
        C_image.push_back(temp);
    }

    for (int k = 0; k < num_dir; k++){
        for(int i = 0; i < rows; i++){
            for (int j = 0; j < cols; j++){
                
                if (index[i][j] == k){
                    int color = grad.at<uchar>(i, j);
                    C_image[k].at<uchar>(i, j) = color;
                }
                else{
                    C_image[k].at<uchar>(i, j) = 0;
                }
            }
        }
    }

    // Mat Cp;
    Mat Cp = Mat::zeros(rows, cols, CV_8UC1);
    // im_XY.copyTo(Cp);
    // merge(C_image, Cp);
    // hconcat( C_image, Cp );
    for (int i = 0; i < num_dir; i++){
        Cp = Cp + C_image[i];
    }

    float kernRef_data[(ks*2 + 1)*(ks*2 + 1)];

    for (int i = 0; i < ks*2 + 1; i++){
        for (int j = 0; j < ks*2 + 1; j++){
            // std::cout << "i: " << i << "j: " << j << std::endl;
            kernRef_data[i*(ks*2 + 1) + j] = 0;
            if (i <= ks + 1 + width && i >= ks + 1 - width){
                kernRef_data[i*(ks*2 + 1) + j] = 1;
            }
        }
    }

    // for (int i = 0; i < ks*2 + 1; i++){
    //     for (int j = 0; j < ks*2 + 1; j++){
    //         std::cout << kernRef_data[i*(ks*2 + 1) + j]  << " ";
    //     }
    //     std::cout << std::endl;
    // }


    Mat kernel((ks*2 + 1), (ks*2 + 1), CV_32F, kernRef_data);

    std::vector<Mat> Spn;
    
    for(int i = 0; i < num_dir; i++){
        Mat temp = imRotate(kernel, i*180/num_dir);
        Mat dst;
        filter2D(C_image[i], dst, C_image[i].depth(), temp);
        Spn.push_back(dst);
    }

    Mat Sp = Mat::zeros(rows, cols, CV_8UC1);
    // Spn[0].copyTo(Sp);
    // merge(C_image, Cp);
    // hconcat( C_image, Cp );
    for (int i = 1; i < num_dir; i++){
        Sp = Sp + Spn[i];
    }

    // imshow("Display window", response[0]);
    // waitKey(0);
	return 0;
}

