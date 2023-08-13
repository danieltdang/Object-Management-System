#include "yolo.h"

using namespace std;
using namespace cv;
using namespace dnn;
namespace fs = std::filesystem;

Yolo::Yolo()
{
	INPUT_WIDTH = 640;
	INPUT_HEIGHT = 640;
	SCORE_THRESHOLD = 0.5;
	NMS_THRESHOLD = 0.45;
	CONFIDENCE_THRESHOLD = 0.45;

	FONT_SCALE = 0.7;
	FONT_FACE = FONT_HERSHEY_TRIPLEX;
	THICKNESS = 1;

	LoadCategories();
	LoadImages();
	ReadModel();
}

void Yolo::LoadCategories()
{
	try
	{
		ifstream ifs("models/coco.txt");
		string category;

		while (getline(ifs, category))
		{
			srand(time(0));
			int b = rand() % 256;
			int g = rand() % 256;
			int r = rand() % 256;
			colors[category] = Scalar(b, g, r);
		}
	}
	catch (const exception& ex)
	{
		cout << "[LOAD_CATEGORY_ERROR] " << ex.what() << endl;
	}
}

void Yolo::LoadImages()
{
	try
	{
		string path = "images/";
		for (const auto& entry : fs::directory_iterator(path))
		{
			imagePaths.push_back(entry.path().string());
		}
	}
	catch (const exception& ex)
	{
		cout << "[LOAD_IMAGE_ERROR] " << ex.what() << endl;
	}
}

void Yolo::ReadModel()
{
	try
	{
		net = readNet("models/YOLOv5s.onnx");
	}
	catch (const exception& ex)
	{
		cout << "[READ_MODEL_ERROR] " << ex.what() << endl;
	}
}

void Yolo::DrawLabel(Mat& input_image, string label, int left, int top)
{
	try
	{
		// Display the label at the top of the bounding box.
		int baseLine;
		Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
		top = max(top, label_size.height);
		
		// Offset is used to move the label up
		int offset = 23;

		// Top left corner.
		Point tlc = Point(left, top - offset);

		// Bottom right corner.
		Point brc = Point(left + label_size.width, top + label_size.height + baseLine - offset);

		// Draw rectangle for label
		rectangle(input_image, tlc, brc, colors[label], FILLED);
		rectangle(input_image, tlc, brc, colors[label], 3 * THICKNESS);

		// Put the label on the rectangle
		putText(input_image, label, Point(left, top + label_size.height - offset), FONT_FACE, FONT_SCALE, Scalar(0, 0, 0), THICKNESS, LINE_AA);
	}
	catch (const exception& ex)
	{
		cout << "[DRAW_PREDICTION_ERROR] " << ex.what() << endl;
	}
}