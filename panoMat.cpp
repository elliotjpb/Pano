#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
//#include "cuda.hpp"
//#include <cuda.h>
//#include <cuda_runtime_api.h>

//using namespace cv::gpu;

using namespace cv::xfeatures2d;
using namespace std;
using namespace cv;
//To get homography from images passed in. Currently just spesifying homography
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
    		//cout << it->queryIdx << "\t" <<  it->trainIdx << "\t"  <<  it->distance << "\n";
    		//-- Get the keypoints from the good matches
    		dst_pts.push_back( keypoints_1[ it->queryIdx ].pt );
    		source_pts.push_back( keypoints_2[ it->trainIdx ].pt );
    	}

    	Mat H_12 = findHomography( source_pts, dst_pts, CV_RANSAC );
    	//cout << H_12 << endl;

      return H_12;
}

/** @function main */
int main(int argc, char** argv){

  Mat I1, h_I1;
  Mat I2, h_I2;

  // Create a VideoCapture object and open the input file
  VideoCapture cap1("left.mov");
  VideoCapture cap2("right.mov");
  cap1.set(CV_CAP_PROP_BUFFERSIZE, 10);
  cap2.set(CV_CAP_PROP_BUFFERSIZE, 10);
  //Check if camera opened successfully
  if(!cap1.isOpened() || !cap2.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }

  if (cap1.read(I1)){
     h_I1 = I1;
   }

   if (cap2.read(I2)){
     h_I2 = I2;
   }
   Mat homography;

   homography = Stitching(h_I1,h_I2);
  std::cout << homography << '\n';


VideoWriter video("video/output.avi",CV_FOURCC('M','J','P','G'),30, Size(1280,720));


//Trying to loop frames
    for (;;){
    Mat cap1frame;
    Mat cap2frame;

    cap1 >> cap1frame;
    cap2 >> cap2frame;

    // If the frame is empty, break immediately
    if (cap1frame.empty() || cap2frame.empty())
      break;
      //Need to precalcualte the warp between the images so it doesn't process each frame.
      Mat warpImage2;
      //warpPerspective(cap2frame, warpImage2, homography, Size(cap1frame.cols*2, cap1frame.rows*2), INTER_CUBIC);
      warpPerspective(cap2frame, warpImage2, homography, Size(1280,720), INTER_CUBIC);
      //remap()
      //cout << homography << endl;
      //Point a cv::Mat header at it (no allocation is done)
      Mat final (Size(1280,720), CV_8UC3);
      //Mat final(Size(cap1frame.cols*2 + cap1frame.cols, cap1frame.rows*2),CV_8UC3);
      Mat roi1(final, Rect(0, 0,  cap1frame.cols, cap1frame.rows));
      Mat roi2(final, Rect(0, 0, warpImage2.cols, warpImage2.rows));
      warpImage2.copyTo(roi2);
      cap1frame.copyTo(roi1);

      video.write(final);

      //imshow ("Result", final);
    if(waitKey(30) >= 0) break;
     //destroyWindow("Stitching");
    // waitKey(0);
  }
  video.release();
  return 0;
}
