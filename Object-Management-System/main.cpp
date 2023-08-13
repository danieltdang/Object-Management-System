#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "yolo.h"

using namespace cv;
using namespace std;
using namespace dnn;

int main()
{
    Yolo yolo;
    yolo.DisplayImages();

    return 0;
}