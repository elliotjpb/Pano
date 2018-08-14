#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/opencv.hpp>

using namespace cv::xfeatures2d;
using namespace std;
using namespace cv;

Mat Stitching(Mat image1,Mat image2){

    Mat I_1 = image1;
    Mat I_2 = image2;

    cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();


    	// Step 1: Detect the keypoints:
    	std::vector<KeyPoint> keypoints_1, keypoints_2;
    	f2d->detect( I_1, keypoints_1 );
    	f2d->detect( I_2, keypoints_2 );

    	// Step 2: Calculate descriptors (feature vectors)
    	Mat descriptors_1, descriptors_2;
    	f2d->compute( I_1, keypoints_1, descriptors_1 );
    	f2d->compute( I_2, keypoints_2, descriptors_2 );

    	// Step 3: Matching descriptor vectors using BFMatcher :
    	BFMatcher matcher;
    	std::vector< DMatch > matches;
    	matcher.match( descriptors_1, descriptors_2, matches );

    	// Keep best matches only to have a nice drawing.
    	// We sort distance between descriptor matches
    	Mat index;
    	int nbMatch = int(matches.size());
    	Mat tab(nbMatch, 1, CV_32F);
    	for (int i = 0; i < nbMatch; i++)
    		tab.at<float>(i, 0) = matches[i].distance;
    	sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
    	vector<DMatch> bestMatches;

    	for (int i = 0; i < 200; i++)
    		bestMatches.push_back(matches[index.at < int > (i, 0)]);


    	// 1st image is the destination image and the 2nd image is the src image
    	std::vector<Point2f> dst_pts;                   //1st
    	std::vector<Point2f> source_pts;                //2nd

    	for (vector<DMatch>::iterator it = bestMatches.begin(); it != bestMatches.end(); ++it) {
    		cout << it->queryIdx << "\t" <<  it->trainIdx << "\t"  <<  it->distance << "\n";
    		//-- Get the keypoints from the good matches
    		dst_pts.push_back( keypoints_1[ it->queryIdx ].pt );
    		source_pts.push_back( keypoints_2[ it->trainIdx ].pt );
    	}

    	Mat H_12 = findHomography( source_pts, dst_pts, CV_RANSAC );
    	cout << H_12 << endl;
      Mat warpImage2;
      warpPerspective(I_2, warpImage2, H_12, Size(I_1.cols*2, I_1.rows*2), INTER_CUBIC);

      //Point a cv::Mat header at it (no allocation is done)
      Mat final(Size(I_1.cols*2 + I_1.cols, I_1.rows*2),CV_8UC3);

      Mat roi1(final, Rect(0, 0,  I_1.cols, I_1.rows));
      Mat roi2(final, Rect(0, 0, warpImage2.cols, warpImage2.rows));
      warpImage2.copyTo(roi2);
      I_1.copyTo(roi1);

      // cv::Mat result;
      // warpPerspective(I_2,result,H_12,cv::Size(800,600));
      // cv::Mat half(result,cv::Rect(0,0,I_2.cols,I_2.rows));
      // I_1.copyTo(half);
      // return result;

    return final;

}

void readme(){
    std::cout << " Usage: ./SURF_descriptor <img1> <img2>" << std::endl;
}

/** @function main */
int main(int argc, char** argv){

  // Create a VideoCapture object and open the input file
  // If the input is the web camera, pass 0 instead of the video file name
  VideoCapture cap1("left.mov");
  VideoCapture cap2("right.mov");

  // Check if camera opened successfully
  if(!cap1.isOpened() || !cap2.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }

    //int loop=0;
    while(1){

    Mat cap1frame;
    Mat cap2frame;
    // Capture frame-by-frame
    cap1 >> cap1frame;
    cap2 >> cap2frame;

    // If the frame is empty, break immediately
    if (cap1frame.empty() || cap2frame.empty())
      break;

    imshow( "Result", Stitching(cap1frame,cap2frame));

    destroyWindow("Stitching");

    waitKey(0);
    return 0;
  }
}
