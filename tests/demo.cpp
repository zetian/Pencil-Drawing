#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>

#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <climits>
#include <cmath>

using namespace cv;

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
    
    // imshow("Display window", image);
    // waitKey(0);
    cvtColor(origin_image, origin_image, COLOR_RGB2YCrCb);

    

    Mat image, image_X, image_Y;
    origin_image.copyTo(image);
    origin_image.copyTo(image_X);
    origin_image.copyTo(image_Y);
    
    // image.convertTo(image, CV_64FC3);
    auto size = image.size();
    int num_channels = image.channels();
    
    // Mat1f image_X(size.height, size.width);
    // image_X = image;
    // Mat image_Y;
    // Mat1f image_Y(size.height, size.width);
    // image_Y = image;

    Mat imX, imY, im_XY;

    for(int j=0;j<image.rows - 1;j++) 
    {
      for (int i=0;i<image.cols;i++)
      {   
        Vec3f color = origin_image.at<Vec3b>(Point(i, j));
        Vec3f color_next = origin_image.at<Vec3b>(Point(i, j + 1));
        color[0] = std::abs(color[0] - color_next[0]);

        image_X.at<Vec3b>(Point(i, j)) = color;
      }
    }
    
    

    std::vector<Mat> channels_1(3);
    split(image_X, channels_1);
    Mat lumIm_X = channels_1[0];
    // imshow("Display window", lumIm_X);
    // waitKey(0);
    // Mat lumIm_2 = channels[1];
    // Mat lumIm_3 = channels[2];


    for(int i = 0; i < image.cols - 1; i++) 
    {
      for (int j = 0; j < image.rows; j++)
      {   
        // Vec3f pre_color = origin_image.at<Vec3b>(Point(i, j));
        Vec3f color = origin_image.at<Vec3b>(Point(i, j));
        Vec3f color_next = origin_image.at<Vec3b>(Point(i + 1, j));
        color[0] = std::abs(color[0] - color_next[0]);

        image_Y.at<Vec3b>(Point(i, j)) = color;
      }
    }
    std::vector<Mat> channels_2(3);
    split(image_Y, channels_2);
    Mat lumIm_Y = channels_2[0];

    // imshow("Display window", lumIm_Y);
    // waitKey(0);
    
    // imshow("Display window", lumIm_Y);
    

    // std::cout << "Height: " << size.height << std::endl;
    // std::cout << "Width: " << size.width << std::endl;
    // std::cout << "Num of channels: " << num_channels << std::endl;

    // auto size_2 = lumIm_1.size();
    // int num_channels_2 = lumIm_1.channels();
    // GaussianBlur(lumIm, lumIm, Size(3, 3), 0, 0);
    // medianBlur(lumIm, lumIm, 3);

    // std::cout << "Height: " << size_2.height << std::endl;
    // std::cout << "Width: " << size_2.width << std::endl;
    // std::cout << "Num of channels: " << num_channels_2 << std::endl;

    // Mat imX = lumIm_1, imY = lumIm_1;
    
    // Vec3f intensity = image.at<Vec3f>(100, 100);

    // Vec3f Y_pix = lumIm.at<Vec3f>(200, 100);
    // int pixelval = Y_pix[0];
    
    // std::cout << "~~~~~@@@~: " << imX.at<Vec3f>(0, 0)[0] << std::endl;
    double scale = 1.0;
    double delta = 0.0;
    // int ddepth = CV_32F;
    int ddepth = -1;

    // Scharr(lumIm_1, imX, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    // Scharr(lumIm_1, imY, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    // Sobel(lumIm_1, imX, ddepth, 1, 0, 1, scale, delta, BORDER_DEFAULT);
    // Sobel(lumIm_1, imY, ddepth, 0, 1, 1, scale, delta, BORDER_DEFAULT);
    // // Sobel(lumIm_1, imX, ddepth, 1, 0, 1);
    // // Sobel(lumIm_1, imY, ddepth, 0, 1, 1);
    // // convertScaleAbs(imX, imX);
    // // convertScaleAbs(imY, imY);
    addWeighted(lumIm_X, 0.5, lumIm_Y, 0.5, 0, im_XY);
    // imshow("Display window", im_XY);
    // waitKey(0);
    int ks = 8;
    int num_dir = 8;
    int width = 1;

    // Mat kern = (Mat_<char>(ks*2 + 1, ks*2 + 1));

    // std::cout << "~!!@#!" << (ks*2 + 1)^2 << std::endl;

    float kern_data[(ks*2 + 1)*(ks*2 + 1)];
    // kern_data[0] = 0;


    for (int i = 0; i < ks*2 + 1; i++){
        for (int j = 0; j < ks*2 + 1; j++){
            // std::cout << "i: " << i << "j: " << j << std::endl;
            kern_data[i*(ks*2 + 1) + j] = 0;
            if (j == ks + 1){
                kern_data[i*(ks*2 + 1) + j] = 1;
            }
        }
    }
    Mat kern((ks*2 + 1), (ks*2 + 1), CV_32F, kern_data);

    std::vector<Mat> response;
    for(int i = 0; i < num_dir; i++){
        Mat kernel = imRotate(kern, i*180/num_dir);
        Mat dst;
        cv::filter2D(im_XY, dst, im_XY.depth(), kernel);
        response.push_back(dst);
    }
    // imshow("Display window", im_XY);
    // waitKey(0);

    std::vector<std::vector<int> > index(
        image.cols,
        std::vector<int>(image.rows));

    // std::vector<std::vector<int>> index;
    for(int i = 0; i < image.cols; i++){
        for (int j = 0; j < image.rows; j++){
            int index_temp = 0;
            int max = INT_MIN;
            for (int k = 0; k < response.size(); k++){
                Vec3f color = response[k].at<Vec3b>(Point(i, j));
                if (color[0] > max){
                    index_temp = k;
                    max = color[0];
                }
            }
            index[i][j] = index_temp;
        }
    }
    std::vector<Mat> C_image;
    for (int i = 0; i < num_dir; i++){
        Mat temp;
        im_XY.copyTo(temp);
        C_image.push_back(temp);
    }
    // std::cout << "~~QQQ" << std::endl;
    for (int k = 0; k < num_dir; k++){
        for(int i = 0; i < image.cols; i++){
            for (int j = 0; j < image.rows; j++){
                
                if (index[i][j] == k){
                    Vec3f color = im_XY.at<Vec3b>(Point(i, j));
                    
                    // Vec3f color_next = origin_image.at<Vec3b>(Point(i + 1, j));
                    // color[0] = color[0]
                    C_image[k].at<Vec3b>(Point(i, j)) = color;
                    // C_image[k]
                }
                else{
                    // std::cout << "~~QQQ" << std::endl;
                    Vec3f color = im_XY.at<Vec3b>(Point(i, j));
                    color[0] = 0;
                    color[1] = 0;
                    color[2] = 0;
                    C_image[k].at<Vec3b>(Point(i, j)) = color;
                }
            }
        }
    }
    // Mat Cp;
    // im_XY.copyTo(Cp);
    // // merge(C_image, Cp);
    // // hconcat( C_image, Cp );
    // for (int i = 0; i < num_dir; i++){
    //     Cp = Cp + C_image[i];
    // }


    float kernRef_data[(ks*2 + 1)*(ks*2 + 1)];
    // kern_data[0] = 0;


    for (int i = 0; i < ks*2 + 1; i++){
        for (int j = 0; j < ks*2 + 1; j++){
            // std::cout << "i: " << i << "j: " << j << std::endl;
            kernRef_data[i*(ks*2 + 1) + j] = 0;
            if (j <= ks + 1 + width && j >= ks + 1 - width){
                kernRef_data[i*(ks*2 + 1) + j] = 1;
            }
        }
    }
    Mat kernel((ks*2 + 1), (ks*2 + 1), CV_32F, kernRef_data);


    std::vector<Mat> Spn;
    // for (int i = 0; i < num_dir; i++){
    //     Mat temp;
    //     im_XY.copyTo(temp);
    //     Spn.push_back(temp);
    // }
    
    for(int i = 0; i < num_dir; i++){
        Mat temp = imRotate(kernel, i*180/num_dir);
        Mat dst;
        // std::cout << "~~QQQ" << std::endl;
        cv::filter2D(C_image[i], dst, C_image[i].depth(), temp);
        Spn.push_back(dst);
    }
    
    Mat Sp;
    Spn[0].copyTo(Sp);
    // merge(C_image, Cp);
    // hconcat( C_image, Cp );
    for (int i = 1; i < num_dir; i++){
        Sp = Sp + Spn[i];
    }
    // normalize(Sp, Sp, 205, 10, NORM_MINMAX, CV_8UC1);

    // cv::Mat img3;
    // Sp.convertTo(Sp, CV_32F, 1.0 / 255, 0);
    // normalize(Sp, Sp, 1, 0,NORM_MINMAX);
    // double min, max;
    // minMaxLoc(Sp, &min, &max);
    // std::cout << "~~QQQ" << min << "~~QQQ"  <<max << std::endl;
    // Sp = Sp - min(Sp);
    double min = INT_MAX;
    double max = INT_MIN;
    for(int i = 0; i < image.cols; i++){
        for (int j = 0; j < image.rows; j++){
            Vec3f color = Sp.at<Vec3b>(Point(i, j));
            if (color[0] > max){
                max = color[0];
            }
            if (color[0] < min){
                min = color[0];
            }
        }
    }
    std::cout << "~~QQQ" << min << "~~QQQ"  <<max << std::endl;
    for(int i = 0; i < image.cols; i++){
        for (int j = 0; j < image.rows; j++){
            Vec3f color = Sp.at<Vec3b>(Point(i, j));

            color[0] = color[0] - min;
            color[1] = 0;
            color[2] = 0;
            // color[0] = int((color[0] - min)*(min/max) + 0.5);
            Sp.at<Vec3b>(Point(i, j)) = color;
        }
    }

    imshow("Display window", Sp);
    waitKey(0);

    // for (int i = 0; i < ks*2 + 1; i++){
    //     for (int j = 0; j < ks*2 + 1; j++){
    //         // std::cout << "i: " << i << "j: " << j << std::endl;
    //         std::cout << kernRef_data[i*(ks*2 + 1) + j] ;
    //     }
    //     std::cout<< std::endl;
    // }
    // imshow("Display window", Cp);
    // waitKey(0);

    // for(int i = 0; i < image.cols; i++){
    //     for (int j = 0; j < image.rows; j++){
    //         std::cout << index[i][j] << " " ;
    //     }
    //     std::cout <<std::endl;
    // }


    // imshow("Display window", im_XY);


    // double angle = 45;
    // Point2f src_center(kern.cols/2.0F, kern.rows/2.0F);
    // Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);

    // cv::Mat rot_test = imRotate(kern, 45);

    // Mat rot_test_2 = rotateImage(kern, 45, ks*2 + 1);


    // Mat dst;
    // warpAffine(kern, dst, rot_mat, kern.size());
    // for (int i = 0; i < ks*2 + 1; i++){
    //     for (int j = 0; j < ks*2 + 1; j++){
    //         std::cout << dst.at<float>(j, i);
    //     }
    //     std::cout << std::endl;
    // }

    // for (int i = 0; i < ks*2 + 1; i++){
    //     for (int j = 0; j < ks*2 + 1; j++){
    //         std::cout << rot_test_2.at<float>(j, i);
    //     }
    //     std::cout << std::endl;
    // }

    // Mat rot_mat = getRotationMatrix2D( center, angle, scale );
    // std::cout << "~~~" << std::endl;
    // for (int i = 0; i < ks*2 + 1; i++){
    //     for (int j = 0; j < ks*2 + 1; j++){
    //         std::cout << kern_data[i*(ks*2 + 1) + j];
    //     }
    //     std::cout << std::endl;
    // }

    //  << -1, -1, -1, -1, -1,
    // -1, -1, -1, -1, -1,
    // -1, -1, 24, -1, -1,
    // -1, -1, -1, -1, -1,
    // -1, -1, -1, -1, -1);

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

    // imshow("Display window", im_XY);

    // waitKey(0);
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

