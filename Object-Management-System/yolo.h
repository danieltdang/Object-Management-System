#pragma once
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

class Yolo
{
public:
	Yolo();

	void LoadCategories();
	void LoadImages();
	void ReadModel();
	void DrawLabel(Mat& input_image, string label, int left, int top);
	void pre_process(Mat& input_image);
	Mat post_process(Mat& input_image);
	void DisplayImages();
private:
	// Categories with Colors
	vector<pair<string, Scalar>> colors;

	// Image Paths
	vector<string> imagePaths;

	// Model
	Net net;

	// Detections
	vector<Mat> detections;

	// Constants
	float INPUT_WIDTH;
	float INPUT_HEIGHT;
	float SCORE_THRESHOLD;
	float NMS_THRESHOLD;
	float CONFIDENCE_THRESHOLD;

	// Text parameters
	float FONT_SCALE;
	int FONT_FACE;
	int THICKNESS;
};