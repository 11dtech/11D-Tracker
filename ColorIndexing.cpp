#include<opencv/cv.h>
#include<opencv/highgui.h>
#include<opencv2\core\core.hpp>

#include<opencv2\highgui\highgui.hpp>
#include<opencv2\video\tracking.hpp>
#include<iostream>

using namespace cv;
using namespace std;

//function prototypes
Mat histMatrix(Mat image);
//void pixScan(Mat image, int i, int j);
void draw(Mat image, int i, int j);

//Global variables
Mat blue, green, red; //define B,G,R matrix as global variable for histMatrix function value passing
int histBin = 64; //define histogram resolution
Mat matchVal; //define template similarity matrix
Mat tempMatch; //declare template matching result matrix
Mat temp, crop, scaleCrop, crop_color, crop_edge; //create template and cropped instance in source image
Mat temp_b, temp_g, temp_r, temp_b_norm, temp_g_norm, temp_r_norm; //create template histo signature as global variable
//temporary debugging variable
double old_frc_b, old_frc_g, old_frc_r;
//color space dimentional weight
double whue = 1;
double wsat = 1;
double wval = 0.5;
double plotScale = 0;
double xplot, yplot;
//create match fractionaa
double frc = 0;
double old_frc = 0;
int x = 15; //arbitrary start
int y = 15; //arbitrary start
float tempscale = 0.5;	//template scaling factor
float imgscale = 0.5;		//video scaling factor
int sampling = 5;			//match sampling gap
double histSensN = 0.3;		//negative threshold
double histSensP = 0.4;		//positive threshold
bool detect = false;		//detection flag
int result_cols = 1;
int result_rows = 1;
int kalmanZoom = 120;	//search range around predicted match
double matchScale = 1;	//default match scale to 1
int crosshair = 5;
double accumEdgeFrac = 0; //accumulative edge density
int traceLength = 20;
cv::vector<cv::Point> traceList;
Point prevPnt;
Point currPnt;

int main(){
	//create image matrix
	Mat image, imageRecov;
	//create video
	VideoCapture cap;
	//link to internal camera
	cap.open(0);
	//load image template
	temp = imread("C:\\Users\\lakec\\OneDrive\\Documents\\OpenCV Image Database\\Tea_box.jpg", 1);
	//reduce image size
	resize(temp, temp, Size(), tempscale, tempscale, 1);
	//convert template into HSV color space
	cvtColor(temp, temp, CV_BGR2HSV);

	//get the template size
	int temprow = temp.rows;
	int tempcol = temp.cols;
	//show template edges
	Mat tempEdge;
	Canny(temp, tempEdge, 80, 100, 3);
	namedWindow("Template Edges", WINDOW_NORMAL);
	imshow("Template Edges", tempEdge);
	namedWindow("Template", WINDOW_NORMAL);
	imshow("Template", temp);
	//calculate the template sweeping range [result_cols and result_rows]
	cap >> image;	//camera captures the image
	resize(image, image, Size(), imgscale, imgscale, 1);
	result_cols = image.cols - matchScale*temp.cols + 1;
	result_rows = image.rows - matchScale*temp.rows + 1;

	//calculate histogram matrix
	Mat tempHist = histMatrix(temp);
	//store template histogram
	tempHist.col(0).copyTo(temp_b);
	tempHist.col(1).copyTo(temp_g);
	tempHist.col(2).copyTo(temp_r);
	//normalize for scalling purpose
	normalize(temp_b, temp_b_norm, 0, 1000, NORM_MINMAX, -1, Mat());
	normalize(temp_g, temp_g_norm, 0, 1000, NORM_MINMAX, -1, Mat());
	normalize(temp_r, temp_r_norm, 0, 1000, NORM_MINMAX, -1, Mat());

	//Initialize Kalman Filter
	KalmanFilter Kalman(9, 3, 0);

	Kalman.transitionMatrix = *(Mat_<float>(9, 9) << 1, 0, 0, 1, 0, 0, 0.5, 0, 0,
		0, 1, 0, 0, 1, 0, 0, 0.5, 0,
		0, 0, 1, 0, 0, 1, 0, 0, 0.5,
		0, 0, 0, 1, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 1, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 1, 0, 0, 1,
		0, 0, 0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 1);
	Mat_<float> measurement(3, 1);
	measurement.setTo(Scalar(0));
	Kalman.statePre.at<float>(0) = 0;
	Kalman.statePre.at<float>(1) = 0;
	Kalman.statePre.at<float>(2) = 0;
	Kalman.statePre.at<float>(3) = 0;
	Kalman.statePre.at<float>(4) = 0;
	Kalman.statePre.at<float>(5) = 0;
	Kalman.statePre.at<float>(6) = 0;

	setIdentity(Kalman.measurementMatrix);
	setIdentity(Kalman.processNoiseCov, Scalar::all(1e-5));
	setIdentity(Kalman.measurementNoiseCov, Scalar::all(2e-2));
	setIdentity(Kalman.errorCovPost, Scalar::all(.1));


	namedWindow("Video", WINDOW_NORMAL);

	waitKey(1000);
	//-------------------------------------DETECTION-------------------------------------------------------------
	while (1){
		while (detect == true){
			cap >> image;	//camera captures the image
			resize(image, image, Size(), imgscale, imgscale, 1);	//reduce the size of the image
			cvtColor(image, image, CV_BGR2HSV);		//convert template into HSV color space
			//Kalman prediction
			Mat prediction = Kalman.predict();

			float predict_x;
			float predict_y;
			float predict_s;
			if (prediction.at<float>(0) < 0){
				predict_x = 0;
			}
			else{
				predict_x = prediction.at<float>(0);
			}
			if (prediction.at<float>(1) < 0){
				predict_y = 0;
			}
			else{
				predict_y = prediction.at<float>(1);
			}
			if (prediction.at<float>(2) < 0){
				predict_s = 0;
			}
			else{
				predict_s = prediction.at<float>(2);
			}
			//computer match scale
			int tempScaleCols = temp.cols*matchScale;
			int tempScaleRows = temp.rows*matchScale;
			// Mark Kalman filter predition
			line(image, Point(predict_x + tempScaleCols / 2, predict_y - crosshair + tempScaleRows / 2), Point(predict_x + tempScaleCols / 2, predict_y + crosshair + tempScaleRows / 2), Scalar(0, 0, 255), 1, 8);
			line(image, Point(predict_x - crosshair + tempScaleCols / 2, predict_y + tempScaleRows / 2), Point(predict_x + crosshair + tempScaleCols / 2, predict_y + tempScaleRows / 2), Scalar(0, 0, 255), 1, 8);

			int i, j;
			int ilow = predict_x - kalmanZoom / 2;
			int jlow = predict_y - kalmanZoom / 2;
			int ihigh = predict_x + kalmanZoom / 2;
			int jhigh = predict_y + kalmanZoom / 2;

			for (i = max(ilow, 1); i <= min(ihigh, result_cols); i += sampling){
				for (j = max(jlow, 1); j <= min(jhigh, result_rows); j += sampling){
					//crop from the source image
					//make sure tempScale won't for the crop to go out of the bound
					tempScaleCols = min(tempScaleCols, image.cols - i);
					tempScaleRows = min(tempScaleRows, image.rows - j);

					crop = image(Rect(i, j, tempScaleCols, tempScaleRows));
					//declare crop b,g,r histogram matrices
					Mat crop_b, crop_g, crop_r;
					//compute histogram for the cropped region
					Mat cropHist = histMatrix(crop);
					cropHist.col(0).copyTo(crop_b);
					cropHist.col(1).copyTo(crop_g);
					cropHist.col(2).copyTo(crop_r);
					//computer histogram intersect region matrices
					Mat b_hist_intersec = min(temp_b, crop_b);
					Mat g_hist_intersec = min(temp_g, crop_g);
					Mat r_hist_intersec = min(temp_r, crop_r);
					//compute fraction match value (0-1) for each channel
					float sum_intersec_b = 0;
					float sum_intersec_g = 0;
					float sum_intersec_r = 0;
					float sum_model_b = 0;
					float sum_model_g = 0;
					float sum_model_r = 0;

					for (int s = 0; s<histBin - 1; s++){
						sum_intersec_b = sum_intersec_b + b_hist_intersec.at<float>(s);
						sum_intersec_g = sum_intersec_g + g_hist_intersec.at<float>(s);
						sum_intersec_r = sum_intersec_r + r_hist_intersec.at<float>(s);
						sum_model_b = sum_model_b + temp_b.at<float>(s);
						sum_model_g = sum_model_g + temp_g.at<float>(s);
						sum_model_r = sum_model_r + temp_r.at<float>(s);
					}
					double frc_b = double(sum_intersec_b) / double(sum_model_b);
					double frc_g = double(sum_intersec_g) / double(sum_model_g);
					double frc_r = double(sum_intersec_r) / double(sum_model_r);
					frc = pow(frc_b, whue)*pow(frc_g, wsat)*pow(frc_r, wval);	//weighted match value from all three channel
					if (frc > old_frc){
						x = i;
						y = j;
						old_frc = frc;
					}
				}
			}
			double scalefrc = 0;
			double scaleDev = 0.03; //scale step size
			for (int i = -2; i <= 2; i++){
				//Scalling Test the match region
				double scale = (1 + scaleDev*i);
				int scaleWidth = scale*tempScaleCols;
				int scaleHeight = scale*tempScaleRows;
				int cropWidth = x + scaleWidth;
				int cropHeight = y + scaleHeight;
				if (cropWidth > image.cols){
					scaleWidth = image.cols - x;
				}
				if (cropHeight > image.rows){
					scaleHeight = image.rows - y;
				}
				int shiftx = x + (tempScaleCols - scaleWidth) / 2;
				int shifty = y + (tempScaleRows - scaleHeight) / 2;
				//make sure tempScale won't for the crop to go out of the bound
				scaleWidth = min(scaleWidth, image.cols - shiftx);
				scaleHeight = min(scaleHeight, image.rows - shifty);
				shiftx = max(shiftx, 1);
				shifty = max(shifty, 1);

				scaleCrop = image(Rect(shiftx, shifty, scaleWidth, scaleHeight));

				//declare crop b,g,r histogram matrices
				Mat scale_b, scale_g, scale_r;
				//compute histogram for the cropped region
				Mat scaleCropHist = histMatrix(scaleCrop);
				scaleCropHist.col(0).copyTo(scale_b);
				scaleCropHist.col(1).copyTo(scale_g);
				scaleCropHist.col(2).copyTo(scale_r);
				//normalize histogram [0-1000] for different scale
				normalize(scale_b, scale_b, 0, 1000, NORM_MINMAX, -1, Mat());
				normalize(scale_g, scale_g, 0, 1000, NORM_MINMAX, -1, Mat());
				normalize(scale_r, scale_r, 0, 1000, NORM_MINMAX, -1, Mat());
				//computer histogram intersect region matrices
				Mat b_hist_scale_intersec = min(temp_b_norm, scale_b);
				Mat g_hist_scale_intersec = min(temp_g_norm, scale_g);
				Mat r_hist_scale_intersec = min(temp_r_norm, scale_r);
				//compute fraction match value (0-1) for each channel
				float sum_intersec_b = 0;
				float sum_intersec_g = 0;
				float sum_intersec_r = 0;
				float sum_model_b = 0;
				float sum_model_g = 0;
				float sum_model_r = 0;
				for (int s = 0; s<histBin - 1; s++){
					sum_intersec_b = sum_intersec_b + b_hist_scale_intersec.at<float>(s);
					sum_intersec_g = sum_intersec_g + g_hist_scale_intersec.at<float>(s);
					sum_intersec_r = sum_intersec_r + r_hist_scale_intersec.at<float>(s);
					sum_model_b = sum_model_b + temp_b_norm.at<float>(s);
					sum_model_g = sum_model_g + temp_g_norm.at<float>(s);
					sum_model_r = sum_model_r + temp_r_norm.at<float>(s);
				}
				double frc_b = double(sum_intersec_b) / double(sum_model_b);
				double frc_g = double(sum_intersec_g) / double(sum_model_g);
				double frc_r = double(sum_intersec_r) / double(sum_model_r);
				frc = pow(frc_b, whue)*pow(frc_g, wsat)*pow(frc_r, wval);
				if (frc > scalefrc){
					tempscale = scale;
					scalefrc = frc;
				}
			}
			//Update template scale
			double newScale = tempscale*matchScale;
			if (newScale < 1.2 && newScale > 0.8){
				matchScale = newScale;
			}
			//Get Kalman measurement coordinate
			measurement(0) = x;
			measurement(1) = y;
			measurement(2) = 500 * matchScale;

			Mat estimated = Kalman.correct(measurement);

			//Clear temporary scale
			tempscale = 0;
			//draw rectangles
			plotScale = prediction.at<float>(2) / 500;
			Point p1(estimated.at<float>(0), estimated.at<float>(1));
			Point p2(estimated.at<float>(0) + plotScale*temp.cols, estimated.at<float>(1) + plotScale*temp.rows);
			Point c1(estimated.at<float>(0) + plotScale*temp.cols / 2, estimated.at<float>(1) + plotScale*temp.rows / 2);

			if (old_frc < histSensN){
				detect = false; //change iteration loop when max is more than threshold
			}
			old_frc = 0;

			cvtColor(image, imageRecov, CV_HSV2BGR);	 //recover to BGR
			//crop the detection area for canny edge detection
			if (estimated.at<float>(0) > image.cols - plotScale*temp.cols){
				xplot = image.cols - plotScale*temp.cols;
			}
			else if (estimated.at<float>(0) < 0){
				xplot = 1;
			}
			else{
				xplot = estimated.at<float>(0);
			}

			if (estimated.at<float>(1) > image.rows - plotScale*temp.rows){
				yplot = image.rows - plotScale*temp.rows;
			}
			else if (estimated.at<float>(1) < 0){
				yplot = 1;
			}
			else{
				yplot = estimated.at<float>(1);
			}
			crop_color = imageRecov(Rect(xplot, yplot, plotScale*temp.cols, plotScale*temp.rows)); //CHANGE WINDOW SIZE
			Canny(crop_color, crop_edge, 80, 100, 3);

			//---------------------Edge Density--------------------------
			int edgeCount = 0;
			double nowFrc = 0;
			for (int i = 1; i <= crop_edge.cols; i++){
				for (int j = 1; j <= crop_edge.rows; j++){
					if (crop_edge.at<float>(i, j) != 0){
						edgeCount++;
					}
				}
			}
			nowFrc = double(edgeCount) / (crop_edge.cols*crop_edge.rows);
			accumEdgeFrac = 0.9*accumEdgeFrac + 0.1*nowFrc;
			if (accumEdgeFrac > 0.45 && accumEdgeFrac < 0.6){
				rectangle(imageRecov, p1, p2, Scalar(0, 255, 0), 1, 8);
			}
			else{
				rectangle(imageRecov, p1, p2, Scalar(0, 0, 255), 1, 8);
			}

			namedWindow("Edge", WINDOW_NORMAL);
			imshow("Edge", crop_edge);
			namedWindow("Color Edge", WINDOW_NORMAL);
			imshow("Color Edge", crop_color);

			//display histogram
			imshow("Video", imageRecov);
			//imshow("Template",temp);
			waitKey(15);
		}

		//---------------------------------------NO DETECTION-----------------------------------------------------------------------
		while (detect == false){
			matchScale = 1; //reset scalling
			cap >> image;	//camera captures the image
			resize(image, image, Size(), imgscale, imgscale, 1);	//reduce the size of the image
			cvtColor(image, image, CV_BGR2HSV);		//convert template into HSV color space
			//Kalman prediction
			Mat prediction = Kalman.predict();
			//sweep through every other columns/rows of the entire image
			int i, j;
			for (i = 0; i<result_cols; i += sampling){
				for (j = 0; j<result_rows; j += sampling){
					//crop from the source image
					crop = image(Rect(i, j, temp.cols, temp.rows));
					//declare crop b,g,r histogram matrices
					Mat crop_b, crop_g, crop_r;
					//compute histogram for the cropped region
					Mat cropHist = histMatrix(crop);
					cropHist.col(0).copyTo(crop_b);
					cropHist.col(1).copyTo(crop_g);
					cropHist.col(2).copyTo(crop_r);
					//computer histogram intersect region matrices
					Mat b_hist_intersec = min(temp_b, crop_b);
					Mat g_hist_intersec = min(temp_g, crop_g);
					Mat r_hist_intersec = min(temp_r, crop_r);
					//compute fraction match value (0-1) for each channel
					float sum_intersec_b = 0;
					float sum_intersec_g = 0;
					float sum_intersec_r = 0;
					float sum_model_b = 0;
					float sum_model_g = 0;
					float sum_model_r = 0;
					for (int s = 0; s<histBin - 1; s++){
						sum_intersec_b = sum_intersec_b + b_hist_intersec.at<float>(s);
						sum_intersec_g = sum_intersec_g + g_hist_intersec.at<float>(s);
						sum_intersec_r = sum_intersec_r + r_hist_intersec.at<float>(s);
						sum_model_b = sum_model_b + temp_b.at<float>(s);
						sum_model_g = sum_model_g + temp_g.at<float>(s);
						sum_model_r = sum_model_r + temp_r.at<float>(s);
					}
					double frc_b = double(sum_intersec_b) / double(sum_model_b);
					double frc_g = double(sum_intersec_g) / double(sum_model_g);
					double frc_r = double(sum_intersec_r) / double(sum_model_r);
					frc = pow(frc_b, whue)*pow(frc_g, wsat)*pow(frc_r, wval);	//weighted match value from all three channel 
					if (frc > old_frc){
						x = i;
						y = j;
						old_frc = frc;
					}
				}
			}
			//Get Kalman measurement coordinate
			measurement(0) = x;
			measurement(1) = y;
			measurement(2) = 500;

			//Correct the Kalman prediction
			Mat estimated = Kalman.correct(measurement);

			if (old_frc > histSensP){
				detect = true; //change iteration loop when max is more than threshold
			}
			old_frc = 0;
			//----------------------------------------------------------------------------------------------------------
			//display histogram
			cvtColor(image, imageRecov, CV_HSV2BGR);	 //recover to BGR
			imshow("Video", imageRecov);
			//imshow("Template",temp);
			waitKey(15);
		}
	}
	return 0;
}

Mat histMatrix(Mat image){
	vector<Mat> bgr_planes;
	split(image, bgr_planes);
	//define B,G,R ranges
	float range[] = { 0, 256 };
	const float*histRange = { range };
	bool uniform = true; bool accumulate = false;
	//compute B,G,R histograms
	calcHist(&bgr_planes[0], 1, 0, Mat(), blue, 1, &histBin, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), green, 1, &histBin, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), red, 1, &histBin, &histRange, uniform, accumulate);
	//recombine three histogram into one matrix for output
	Mat R(histBin, 3, CV_32F);
	blue.copyTo(R.col(0));
	green.copyTo(R.col(1));
	red.copyTo(R.col(2));
	return R;
}