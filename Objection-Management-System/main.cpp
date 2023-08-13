#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace dnn;

// Constants
const float WIDTH = 640;
const float HEIGHT = 640;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

// Text parameters
const float FONT_SCALE = 0.8;
const int FONT_FACE = FONT_HERSHEY_TRIPLEX;
const int THICKNESS = 1;

// Colors
Scalar BLACK = Scalar(0, 0, 0);
Scalar WHITE = Scalar(255, 255, 255);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0, 0, 255);

int main()
{
    // Load categories
    vector<string> categories;
    ifstream ifs("coco.txt");
    string category;
    while (getline(ifs, category))
        categories.push_back(category);

    // Load image
    string path = "test_0.jpg";
    string filename = path.substr(path.find_last_of('/') + 1);
    Mat rawImg = imread(path, IMREAD_COLOR);

    // Resize image
    int targetWidth = 1280;
    int targetHeight = static_cast<int>((static_cast<float>(targetWidth) / rawImg.cols) * rawImg.rows);
    Mat resizedImg;
    resize(rawImg, resizedImg, Size(targetWidth, targetHeight), INTER_LINEAR);

    // Load model
    Net net = readNet("YOLOv5s.onnx");

    // Labels image with processing time
    double freq = getTickFrequency() / 1000;
    double ms = 1 / freq;
    string label = format("INFERENCED IN %.2f ms", ms);
    putText(resizedImg, label, Point(10, 30), FONT_FACE, FONT_SCALE, BLACK, THICKNESS + 1, LINE_AA);
    putText(resizedImg, label, Point(10, 30), FONT_FACE, FONT_SCALE, WHITE, THICKNESS, LINE_AA);

    // Display window
    imshow(filename, resizedImg);
    waitKey(0);
    return 0;
}