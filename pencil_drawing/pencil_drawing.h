#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>

using namespace cv;

class PencelDrawing{
public:
    PencelDrawing(){};
    ~PencelDrawing(){};
public:
    Mat getGrad(Mat source);
    Mat imRotate(const Mat source, double angle);
};

