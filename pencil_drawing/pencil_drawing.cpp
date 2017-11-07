#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

#include "pencil_drawing.h"
using namespace cv;

Mat PencelDrawing::getGrad(Mat source){
    int rows = source.rows;
    int cols = source.cols;
    Mat image_X = Mat::zeros(rows, cols, CV_8UC1);
    Mat image_Y = Mat::zeros(rows, cols, CV_8UC1);
    for(int i = 0; i < rows; i++){
        for (int j = 0; j < cols - 1; j++){   
            int color = source.at<uchar>(i, j);
            int color_next_y = source.at<uchar>(i, j + 1);
            int temp_y = std::abs(color - color_next_y);
            image_X.at<uchar>(i, j) = temp_y;
        }
    }
    for(int i = 0; i < rows - 1; i++){
        for (int j = 0; j < cols; j++){
            int color = source.at<uchar>(i, j);
            int color_next_x = source.at<uchar>(i + 1, j);
            int temp_x = std::abs(color - color_next_x);
            image_Y.at<uchar>(i, j) = temp_x;
        }
    }
    return image_X +image_Y;
}

Mat PencelDrawing::imRotate(const cv::Mat source, double angle) {
    Mat dst;
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
