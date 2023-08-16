#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "yolo.h"

using namespace std;

int main(int argc, char* argv[])
{
    Yolo yolo;
    yolo.DisplayImages();

    return 0;
}