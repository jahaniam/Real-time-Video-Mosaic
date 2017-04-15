/*
Ali Jahani Amiri
Advanced Robotics and Intelligent Systems
University of Tehran, Iran

Demos at: https://www.youtube.com/watch?v=on_sG_X79oQ
Link to the paper: http://ieeexplore.ieee.org/document/7886813/
*/

#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/gpu/gpumat.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;


static void printUsage()
{
	cout <<
		"Real-time Video Mosaic.\n\n"
		"A.Jahaniamiri@gmail.com\n\n"
		"Flags:\n"
		"  --source <file address/device number> \n"
		"		 Feed the code from a device number or video file\n"
		"		 Example: --source c:\\\movie.mp4 .\n"
		"		 Example: --source 0\n"

		"	--x <double>\n"
		"		 This will manually set the x coordinate starting point of first image. Default is in the middle of field of view.\n"
		"		 Example --x 1.5\n"
		"	--y <double>\n"
		"		 This will manually set the y coordinate of starting point of first image. Default is in the middle of field of view.\n"
		"		 Example --y 2.\n"
		"  --xscale <double>\n"
		"		 This will manually set the field of view in X axis. Default is 4.\n"
		"		 Example --xsacle 5.\n"
		"  --yscale <double>\n"
		"		 This will manually set the field of view in y axis. Default is 4.\n"
		"		 Example --ysacle 5.\n"
		"\nMotion Estimation Flags:\n"
		"  --detector (fast|surf|orb|fast_grid|orb_grid|surf_grid|sift_grid) <number of features>"
		"		 You can manually select your detector type. \"fast\",\"surf\",\"orb\",\"fast_grid\",\"orb_grid\",\"surf_grid\" and \"sift_grid\" are available.\n"
		"		 Default detector is \"fast\" and default feature numbers is 300"
		"		 Example : --detector fast 400"
		"  --descriptor (brisk|orb|brief|freak|surf)\n"
		"       Type of features descriptor used for images matching. The default is BRISK.\n"
		"  --match_filter (0|1)\n"
		"       Type of filtering good matches. The default is 0. Number 2 means no filter.\n"
		"  --save <file_name>\n"
		"      Save file to <file_name> file. The default is \"panorama.jpg\"\n"
		"      Example: --save panorama_Jahani.jpg"
		"\nCompositing Flags:\n"
		"  --warp (affine|perspective)\n"
		"      Warp surface type. The default is 'affine'.\n"
		"  --log\n"
		"      Saving the last FOV image automatically when an error occurs. Default is off.\n"
		"\n *****YOU CAN PRESS \"R\" TO RESET THE FOV> THE FOV WILL BE AUTOMATICALLY SAVED BEFORE RESET.\n"
		"\n *****YOU CAN PRESS \"E\" TO EXIT.\n"
		;
}
//Default argumans
string filename = "c:\\data\\test8_2.mp4";

double XScale = 4;
double YScale = 4;
bool log_flag = false;
int device_num = -1;
int features_num = 300;
string detector_type = "fast";
string descriptor_type = "brisk";
string save_pano_to = "panorama.jpg";
string warp_type = "affine";
double x_start_scale = -1;
double y_start_scale = -1;
int detector_set_input = 10;
int match_filter = 0;
Mat H_kol = Mat::eye(3, 3, CV_32F);
Mat H = cv::Mat::eye(3, 3, CV_64F);
Mat H2 = Mat::eye(2, 3, CV_32F);
Mat H_old = Mat::eye(3, 3, CV_32F);

static int parseCmdArgs(int argc, char** argv)
{
	if (argc == 0)
	{
		printUsage();
		return -1;
	}
	for (int i = 1; i < argc; ++i)
	{
		if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
		{
			printUsage();
			return -1;
		}

		else if (string(argv[i]) == "--source")
		{
			filename = argv[i + 1];
			try{
				device_num = stoi(filename);
				}
			catch (...)
			{
				device_num = -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--xscale")
		{
			XScale = atof(argv[i + 1]);

			i++;
		}
		else if (string(argv[i]) == "--yscale")
		{
			YScale = atof(argv[i + 1]);
			i++;
		}


		else if (string(argv[i]) == "--matchfilter")
		{
			if (string(argv[i + 1]) == "0" ||
				string(argv[i + 1]) == "1"||
				string(argv[i + 1]) == "1")
				match_filter = atof(argv[i + 1]);

			i++;
		}

		else if (string(argv[i]) == "--x")
		{
			x_start_scale = atof(argv[i + 1]);
			i++;
		}

		else if (string(argv[i]) == "--y")
		{
			y_start_scale = atof(argv[i + 1]);
			i++;
		}
		else if (string(argv[i]) == "--detector")
		{
			if (string(argv[i + 1]) == "orb_grid" ||
				string(argv[i + 1]) == "fast_grid" ||
				string(argv[i + 1]) == "surf_grid" ||
				string(argv[i + 1]) == "sift_grid" ||
				string(argv[i + 1]) == "fast" ||
				string(argv[i + 1]) == "surf" ||
				string(argv[i + 1]) == "orb")
				detector_type = argv[i + 1];
			if (argc > i + 2){


				if (atoi(argv[i + 2]) <= 0) i++;
				else
				{
					features_num = atoi(argv[i + 2]);
					i += 2;
				}
			}

		}
		else if (string(argv[i]) == "--save")
		{
			save_pano_to = argv[i + 1];
			i++;
		}
		else if (string(argv[i]) == "--warp")
		{
			if (string(argv[i + 1]) == "perspective" ||
				string(argv[i + 1]) == "affine")
				warp_type = string(argv[i + 1]);
			else std::cout << "Bad Warp method. You can use \"affine\" or \"perspective\". Processing with default method which is \"affine\"." << endl;
			i++;
		}

		else if (string(argv[i]) == "--log")
		{
			log_flag = true;
		}

		else if (string(argv[i]) == "--descriptor")
		{
			if (string(argv[i + 1]) == "brisk" ||
				string(argv[i + 1]) == "freak" ||
				string(argv[i + 1]) == "brief" ||
				string(argv[i + 1]) == "orb"   ||
				string(argv[i + 1]) == "surf")
				descriptor_type = argv[i + 1];
			else
			{
				cout << "Bad extractor type method. Using default : BRISK ";
			}
			i++;

		}

	}
	return 0;
}

bool cmpfun(DMatch a, DMatch b) { return a.distance < b.distance; }
//this fuction automatically finds the argumans for feature detectors in order to have defined number of features.
int automatic_feature_detection_set(Mat image)
{
	cv::FeatureDetector* detectors = NULL;
	int last_progress = 0;
	if (detector_type == "surf")
		detectors = new SurfFeatureDetector(detector_set_input, true);
	else if (detector_type == "fast")
		detectors = new FastFeatureDetector(detector_set_input, true);

	if ((detector_type == "surf") || (detector_type == "fast"))
	{
		vector< KeyPoint > keypoints_last;
		detectors->detect(image, keypoints_last);
		int max = keypoints_last.size();
		while (keypoints_last.size() > features_num + 50)
		{
			detector_set_input += 3;
			if (detector_type == "fast") detectors->set("threshold", detector_set_input);
			if (detector_type == "surf") detectors->set("hessianThreshold", detector_set_input);
			detectors->detect(image, keypoints_last);
			int progress = 100 - ((keypoints_last.size() - 348) * 100 / max);
			if (progress >= 100 || progress<0) progress = 100;
			if (last_progress != progress) std::cout << "Please wait:  It might take some minutes for configuration. " << progress << "%" << endl;
			last_progress = progress;
		}
		std::cout << "Configured successfully. The parameter for feature detector is " << detector_set_input << endl;
	}
}

int main(int argc, char* argv[])
{
	double time_algorithm=0;
	double time_complete=0;
	parseCmdArgs(argc, argv);
	double norms = 0;
	cv::Mat img, current_frame, gray_lastimage, gray_curimage;
	Mat img_last, img_last_key;
	Mat img_cur, img_cur_key;
	Mat mask;
	Mat img_last_scaled, img_cur_scaled;
	int fps = 0;
	//this part will open the file or camera device
	VideoCapture cap;
	if (device_num >= 0)
	{
		std::cout << "Opening camera device number " << device_num << ". Please wait..." << endl;
		if (!cap.open(device_num))
		{
			std::cout << "Camera device number " << device_num << " does not exst or installed. Please select another device or read from video file." << endl;
			return -1;
		}
	}
	else
	{
		if (!cap.open(filename))
		{
			std::cout << "Bad file name. Can't read this file." << endl;
			return -1;
		}
	}
	if (device_num >= 0)
		for (int k = 0; k < 10; k++)
			cap >> img;
	else cap >> img;

	cv::Mat offset = cv::Mat::eye(3, 3, CV_64F);
	int counter = 0;
	/////printing which method is going to be used :
	
	cout << "______________________________________________________________________________ " << endl;
	cout << "X-Sacle: " << XScale << " & Y-Scale: " << YScale << endl;
	cout <<"Detector type: "<< detector_type << "	Features: "<<features_num<<endl;
	cout << "Descriptor type: " << descriptor_type << endl;
	cout << "Warping Mode: " << warp_type << endl;
	cout << "Filter Mode: " << match_filter << endl;
	cout << "______________________________________________________________________________ " << endl;


	

	//////////////////////////starting point
	double start_width;
	double start_height;

	//if the x and y are not set from arguments they will be in the middle of FOV
	if (x_start_scale < 0) start_width = img.cols*(XScale / 2 - 0.5);
	else start_width = img.cols*(x_start_scale);
	if (y_start_scale < 0) start_height = img.rows*(YScale / 2 - 0.5);
	else start_height = img.rows*(y_start_scale);

	// making the final image and copying the first frame in the middle of it
	Mat final_img(Size(img.cols * XScale, img.rows * YScale), CV_8UC3);
	Mat f_roi(final_img, Rect(start_width, start_height, img.cols, img.rows));
	img.copyTo(f_roi);
	img.copyTo(img_last);

	//this will reduce the resuloution of the image to medium size inorder to be more robust for different resolutions
	Size size_wrap(final_img.cols, final_img.rows);
	char key;
	Mat rImg;
	double work_megapix = 0.7;
	double work_scale = min(1.0, sqrt(work_megapix * 1e6 / img.size().area()));
	resize(img_last, img_last_scaled, Size(), work_scale, work_scale);

	cv::cvtColor(img_last_scaled, gray_lastimage, CV_RGB2GRAY);
	automatic_feature_detection_set(gray_lastimage);

	// making feature detector object
	cv::FeatureDetector* detector = NULL;
	if (detector_type == "fast_grid")
		detector = new GridAdaptedFeatureDetector(new FastFeatureDetector(10, true), features_num, 3, 3);
	else if (detector_type == "surf_grid")
		detector = new GridAdaptedFeatureDetector(new SurfFeatureDetector(700, true), features_num, 3, 3);
	else if (detector_type == "sift_grid")
		detector = new GridAdaptedFeatureDetector(new SiftFeatureDetector(), features_num, 3, 3);
	else if (detector_type == "orb_grid")
		detector = new GridAdaptedFeatureDetector(new OrbFeatureDetector(), features_num, 3, 3);
	else if (detector_type == "surf")
		detector = new SurfFeatureDetector(detector_set_input, true);
	else if (detector_type == "orb")
		detector = new OrbFeatureDetector(features_num);
	else
		detector = new FastFeatureDetector(detector_set_input, true);



	// making descriptor object. It is needed for matching.
	vector< KeyPoint > keypoints_last, keypoints_cur;
	detector->detect(gray_lastimage, keypoints_last);

	DescriptorExtractor* extractor = NULL;
	if (descriptor_type == "brisk") extractor = new BRISK;

	else if (descriptor_type == "orb") extractor = new OrbDescriptorExtractor;

	else if (descriptor_type == "freak") extractor = new FREAK;

	else if (descriptor_type == "brief") extractor = new BriefDescriptorExtractor;

	else if (descriptor_type == "surf") extractor = new SurfDescriptorExtractor;

	Mat descriptors_last, descriptors_cur;
	extractor->compute(gray_lastimage, keypoints_last, descriptors_last);
	double gui_time;
	int64 t, start_app,start_algorithm;
	Mat panorama_temp;
	int64 start_app_main = getTickCount();
	bool last_err = false;

	bool first_time_4pointfound = false;
	bool second_time_4pointfound = false;


	//starting loop
	while (true)
	{	//take the new frame

		start_app = getTickCount();
		counter++;
		cout << "______________________________________________________________________________ " << endl;
		cout << counter << endl;
		cap >> img_cur;

		start_algorithm = getTickCount();
		if (img_cur.empty()) break;
		cvNamedWindow("current video", CV_WINDOW_NORMAL);
		imshow("current video", img_cur);
		waitKey(1);
		resize(img_cur, img_cur_scaled, Size(), work_scale, work_scale);
		if (img_cur.empty()) break;
		//convert to grayscale
		cvtColor(img_cur_scaled, gray_curimage, CV_RGB2GRAY);

		//First step: feature extraction
		t = getTickCount();
		detector->detect(gray_curimage, keypoints_cur);
		cout << "features= " << keypoints_cur.size() << endl;
		cout << "detecting time: " << ((getTickCount() - t) / getTickFrequency()) << endl;

		//Second step: descriptor extraction
		t = getTickCount();

		extractor->compute(gray_curimage, keypoints_cur, descriptors_cur);
		cout << "descriptor time: " << ((getTickCount() - t) / getTickFrequency()) << endl;
		t = getTickCount();

		//Third step: match with BFMatcher
		if (descriptors_last.type() != CV_32F) {
			descriptors_last.convertTo(descriptors_last, CV_32F);
		}
		if (descriptors_cur.type() != CV_32F) {
			descriptors_cur.convertTo(descriptors_cur, CV_32F);
		}
		vector< DMatch > matches;
		BFMatcher matcher(NORM_L2, true);

		matcher.match(descriptors_last, descriptors_cur, matches);
		if (matches.empty()){
			last_err = false;
			continue;
		}

		vector< DMatch > good_matches;
		vector< DMatch > good_matches2;

		vector<Point2f> match1, match2;
		sort(matches.begin(), matches.end(), cmpfun);
		//matching filter number 0. It will calculate the distance of matches and the first 50 ones which has less 4 times distance than max distance will be considered.
		if (match_filter == 0)
		{

			double max_dist = 0; double min_dist = 100;

			for (int i = 0; i < matches.size(); i++)
			{
				double dist = matches[i].distance;
				if (dist < min_dist) min_dist = dist;
				if (dist > max_dist) max_dist = dist;
			}
			for (int i = 0; i < matches.size() && i < 50; i++)
			{
				if (matches[i].distance <= 4 * min_dist)
				{
					good_matches2.push_back(matches[i]);
				}
			}
		}



		//matching filter number 1. It will calculate the norms od distances and the it has the same matching filter number 0 at the end to reduce and find the best the matching features.
		else if (match_filter == 1) {
			int counterx;
			float res;
			for (int i = 0; i < (int)matches.size(); i++){
				counterx = 0;
				for (int j = 0; j < (int)matches.size(); j++){
					if (i != j){
						res = cv::norm(keypoints_last[matches[i].queryIdx].pt - keypoints_last[matches[j].queryIdx].pt) - cv::norm(keypoints_cur[matches[i].trainIdx].pt - keypoints_cur[matches[j].trainIdx].pt);
						if (abs(res) < (img.rows*0.03 + 3)){ //this value(0.03) has to be adjusted
							counterx++;
						}
					}
				}
				if (counterx > (matches.size() / 10)){
					good_matches.push_back(matches[i]);
				}
			}

			double max_dist = 0; double min_dist = 100;
			for (int i = 0; i < good_matches.size(); i++)
			{
				double dist = good_matches[i].distance;
				if (dist < min_dist) min_dist = dist;
				if (dist > max_dist) max_dist = dist;
			}

			cout << "max_dist:" << max_dist << endl;
			cout << "min_dist:" << min_dist << endl;
			//take just the good points
			if ((max_dist == 0) && (min_dist == 0))
			{
				last_err = false;
				continue;
			}
			sort(good_matches.begin(), good_matches.end(), cmpfun);
			for (int i = 0; i < good_matches.size() && i < 50; i++)
			{
				if (good_matches[i].distance <= 4 * min_dist)
				{
					good_matches2.push_back(good_matches[i]);
				}
			}
		}
		//no filter
		else if (match_filter == 2)
			good_matches2 = matches;


		
		cout << "goodmatches features=" << good_matches2.size() << endl;

		vector< Point2f > obj_last;
		vector< Point2f > scene_cur;

		//take the keypoints

		for (int i = 0; i < good_matches2.size(); i++)
		{
			obj_last.push_back(keypoints_last[good_matches2[i].queryIdx].pt);
			scene_cur.push_back(keypoints_cur[good_matches2[i].trainIdx].pt);
		}
		cout << "match time: " << ((getTickCount() - t) / getTickFrequency()) << endl;
		t = getTickCount();
		Mat mat_match;

		if (scene_cur.size() >= 4)
		{
			first_time_4pointfound = true;
			//drawing some features and matches
			drawMatches(img_last, keypoints_last, img_cur, keypoints_cur, good_matches2, mat_match);
			cvNamedWindow("match", WINDOW_NORMAL);
			imshow("match", mat_match);
			if (counter == 1) waitKey(0);
			
			// finding homography matrix 
			
			if (warp_type == "affine")
			{
				H2 = estimateRigidTransform(scene_cur, obj_last, 0);
				if (H2.data == NULL) {
					last_err = false;
					good_matches.clear();
					good_matches2.clear();
					scene_cur.clear();
					obj_last.clear();
					continue;
				}

				H.at<double>(0, 0) = H2.at<double>(0, 0);
				H.at<double>(0, 1) = H2.at<double>(0, 1);
				H.at<double>(0, 2) = H2.at<double>(0, 2);
				H.at<double>(1, 0) = H2.at<double>(1, 0);
				H.at<double>(1, 1) = H2.at<double>(1, 1);
				H.at<double>(1, 2) = H2.at<double>(1, 2);
				cout << "H=" << H2 << endl;
			}

			else if (warp_type == "perspective")
			{
				H = findHomography(scene_cur, obj_last, CV_RANSAC, 3);
				if (H.empty()){
					good_matches.clear();
					good_matches2.clear();
					scene_cur.clear();
					obj_last.clear();
					continue;
				}
				cout << "H=" << H << endl;

			}

			// using corelations and norms to find the errors and skip that frame. Bad matching can lead to bad homography matrix
			H.convertTo(H, CV_32F);
			H_old.convertTo(H_old, CV_32F);
			Mat correlation;
			matchTemplate(H_old, H, correlation, CV_TM_CCOEFF_NORMED);
			cout << "correlation:" << correlation << endl;
			double nownorms = norm(H - H_old, 2);
			H.convertTo(H, CV_64F);
			H_old.convertTo(H_old, CV_64F);
			cout << "now norm:" << nownorms << endl;
			cout << "miangin norm:" << norms << endl;

			if (norms == 0)	norms = nownorms;


			if ((nownorms > 2 * norms) && (abs(correlation.at<float>(0)) < 0.8) || (nownorms > 10 * norms))
			{
				if (!last_err && (counter != 1))
				{
					cout << "Errorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr" << endl;
					if (log_flag){
						string name = "Error_frame_" + to_string(counter) + ".jpg";
						imwrite(name, panorama_temp);
								}
					last_err = true;
					good_matches.clear();
					good_matches2.clear();
					scene_cur.clear();
					obj_last.clear();
					continue;
				}
				else last_err = false;
			}
			else{
				norms = (norms* (counter - 1) + nownorms) / counter;
				last_err = false;

			}

			//take the x_offset and y_offset
			/*the offset matrix is of the type

			|1 0 x_offset |
			|0 1 y_offset |
			|0 0 1        |

			*/
			offset.at<double>(0, 2) = start_width;
			offset.at<double>(1, 2) = start_height;
			if (first_time_4pointfound == true && second_time_4pointfound == false)
			{
				H_kol = offset*H;
				second_time_4pointfound = true;

			}
			else
			{
				H_kol = H_kol*H;
			}
			cout << "Homography time: " << ((getTickCount() - t) / getTickFrequency()) << endl;
			t = getTickCount();
			//using gpu for applying homography matrix to images
			gpu::GpuMat rImg_gpu, img_cur_gpu;
			img_cur_gpu.upload(img_cur);
			gpu::warpPerspective(img_cur_gpu, rImg_gpu, H_kol, size_wrap, INTER_NEAREST);
			rImg_gpu.download(rImg);


			cout << "warpPerspective time: " << ((getTickCount() - t) / getTickFrequency()) << endl;
			t = getTickCount();
		
			//ROI for img1
			t = getTickCount();
			mask = cv::Mat::ones(final_img.size(), CV_8U) * 0;

			vector<Point2f> corners(4), corner_trans(4);

			corners[0] = Point2f(0, 0);
			corners[1] = Point2f(0, img.rows);
			corners[2] = Point2f(img.cols, 0);
			corners[3] = Point2f(img.cols, img.rows);
			perspectiveTransform(corners, corner_trans, H_kol);

			//making mask
			vector<Point> line1;
			int Most_top_x_corrner = 0;
		
			line1.push_back(corner_trans[0]);
			line1.push_back(corner_trans[2]);
			line1.push_back(corner_trans[3]);
			line1.push_back(corner_trans[1]);
			fillConvexPoly(mask, line1, Scalar::all(255), 4); 
			line(mask, corner_trans[0], corner_trans[2], Scalar::all(0), 5, 4);
			line(mask, corner_trans[2], corner_trans[3], Scalar::all(0), 5, 4);
			line(mask, corner_trans[3], corner_trans[1], Scalar::all(0), 5, 4);
			line(mask, corner_trans[1], corner_trans[0], Scalar::all(0), 5, 4);
			cout << "mask time: " << ((getTickCount() - t) / getTickFrequency()) << endl;
			
			//applying img into final pano
			rImg.copyTo(final_img, mask);
			t = getTickCount();
			//making gui red lines andd text for FPS

			
			final_img.copyTo(panorama_temp);

			line(final_img, corner_trans[0], corner_trans[2], CV_RGB(255, 0, 0), 5, 4);
			line(final_img, corner_trans[2], corner_trans[3], CV_RGB(255, 0, 0), 5, 4);
			line(final_img, corner_trans[3], corner_trans[1], CV_RGB(255, 0, 0), 5, 4);
			line(final_img, corner_trans[1], corner_trans[0], CV_RGB(255, 0, 0), 5, 4);

			fps = 1 / ((getTickCount() - start_app) / getTickFrequency());
			putText(final_img, "fps=" + std::to_string(fps), Point2f(100, 100), CV_FONT_NORMAL, 1.5, CV_RGB(255, 0, 0), 4, 8);
			putText(final_img, "frame=" + std::to_string(counter), Point2f(100, 150), CV_FONT_NORMAL, 1.5, CV_RGB(255, 0, 0), 4, 8);
			namedWindow("Img", WINDOW_NORMAL);
			imshow("Img", final_img);
			gui_time = ((getTickCount() - t) / getTickFrequency());
			cout << "GUI time: " << gui_time << endl;

			////looop preparation
			t = getTickCount();
			gray_curimage.copyTo(gray_lastimage);
			img_cur.copyTo(img_last);
			keypoints_last = keypoints_cur;
			descriptors_last = descriptors_cur;
			H.copyTo(H_old);
			last_err = false;
			
			time_algorithm+=(getTickCount() - start_algorithm) / getTickFrequency()   - gui_time;
			time_complete+=(getTickCount() - start_app) / getTickFrequency() - gui_time;


			cout << "loop prepration time: " << ((getTickCount() - t) / getTickFrequency()) << endl;

		}
		else printf("match point are less than 3 or 4 for homography ");
		cout << "_________________________ " << endl;


		good_matches.clear();
		good_matches2.clear();
		key = waitKey(2);

		panorama_temp.copyTo(final_img);
		//exit
		if (key == 'e')
			break;

		//reseting fov
		if (key == 'r')
		{
			string name = "reset_frame_" + to_string(counter) + ".jpg";
			imwrite(name, panorama_temp);
			panorama_temp = panorama_temp * 0;
			final_img = final_img * 0;

			H.at<double>(2, 0) = 0;
			H.at<double>(2, 1) = 0;
			H.at<double>(2, 2) = 1;

			H_kol.at<double>(0, 0) = 1;
			H_kol.at<double>(0, 1) = 0;
			H_kol.at<double>(0, 2) = 0;
			H_kol.at<double>(1, 0) = 0;
			H_kol.at<double>(1, 1) = 1;
			H_kol.at<double>(1, 2) = 0;

			H_kol.at<double>(2, 0) = 0;
			H_kol.at<double>(2, 1) = 0;
			H_kol.at<double>(2, 2) = 1;

			first_time_4pointfound = true;
			second_time_4pointfound = false;

			//copy current img in the middle of fov
			Mat f_roi(panorama_temp, Rect(start_width, start_height, img_last.cols, img_last.rows));
			cap>>img_last;
			img_last.copyTo(f_roi);
			
		}

	}

	cout << "TOTal time from start: " << ((getTickCount() - start_app_main) / getTickFrequency()) << endl;
	cout << "Total time with hard/webcam consideration:" << time_complete<<endl;
	cout << "Total time algorithm:" << time_algorithm << endl;
	imwrite(save_pano_to, panorama_temp);

}
