#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "yolo.h"

using namespace cv;
using namespace std;
using namespace dnn;

// Constants
const float INPUT_WIDTH = 640;
const float INPUT_HEIGHT = 640;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

// Text parameters
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_TRIPLEX;
const int THICKNESS = 1;

// Colors
Scalar BLACK = Scalar(0, 0, 0);
Scalar WHITE = Scalar(255, 255, 255);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0, 0, 255);

void draw_label(Mat& input_image, string label, int left, int top)
{
    // Display the label at the top of the bounding box.
    int baseLine;
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, label_size.height);
    int offset = 23;

    // Top left corner.
    Point tlc = Point(left, top - offset);

    // Bottom right corner.
    Point brc = Point(left + label_size.width, top + label_size.height + baseLine - offset);

    // Draw rectangle for label
    rectangle(input_image, tlc, brc, BLUE, FILLED);
    rectangle(input_image, tlc, brc, BLUE, 3 * THICKNESS);

    // Put the label on the rectangle
    putText(input_image, label, Point(left, top + label_size.height - offset), FONT_FACE, FONT_SCALE, BLACK, THICKNESS, LINE_AA);
}

vector<Mat> pre_process(Mat& input_image, Net& net)
{
    // Convert to blob
    Mat blob;
    blobFromImage(input_image, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

    net.setInput(blob);

    // Forward propagate
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}

Mat post_process(Mat& input_image, vector<Mat>& outputs, const vector<string>& category)
{
    // Initialize vectors to hold respective outputs while unwrapping detections
    vector<int> category_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    // Resizing factor.
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    float* data = (float*)outputs[0].data;
    const int dimensions = 85;

    // 25200 for default size 640
    const int rows = 25200;

    // Iterate through 25200 detections
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];

        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float* categories = data + 5;

            // Create a 1x85 Mat and store class scores of 80 categories
            Mat scores(1, category.size(), CV_32FC1, categories);

            // Perform minMaxLoc and acquire the index of best class  score
            Point category_id;
            double max_category_score;
            minMaxLoc(scores, 0, &max_category_score, 0, &category_id);

            // Continue if the class score is above the threshold
            if (max_category_score > SCORE_THRESHOLD)
            {
                // Store class ID and confidence in the pre-defined respective vectors
                confidences.push_back(confidence);
                category_ids.push_back(category_id.x);

                // Center
                float cx = data[0];
                float cy = data[1];

                // Box dimension
                float w = data[2];
                float h = data[3];

                // Bounding box coordinates
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);

                // Store good detections in the boxes vector
                boxes.push_back(Rect(left, top, width, height));
            }
        }
        // Jump to the next row.
        data += 85;
    }

    // Perform Non-Maximum Suppression and draw predictions
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        
        // Draw bounding box
        rectangle(input_image, Point(left, top), Point(left + width, top + height), BLUE, 3 * THICKNESS);

        // Get the label for the class name and its confidence
        string label = format("%.2f", confidences[idx]);
        label = category[category_ids[idx]] + ": " + label;

        // Draw class labels
        draw_label(input_image, label, left, top);
    }

    return input_image;
}

int main()
{
    Yolo yolo;
    yolo.DisplayImages();

    return 0;
}