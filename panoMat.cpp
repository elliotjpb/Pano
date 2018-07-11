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

    if (argc != 3){
        readme(); return -1;
    }

    Mat image1 = imread(argv[1], IMREAD_COLOR);
    Mat image2 = imread(argv[2], IMREAD_COLOR);

    // Mat image1 = imread(argv[1], IMREAD_GRAYSCALE);
    // Mat image2 = imread(argv[2], IMREAD_GRAYSCALE);

    if (!image1.data || !image2.data){
        std::cout << " --(!) Error reading images " << std::endl; return -1;
    }

    imshow( "Result", Stitching(image1,image2));

    destroyWindow("Stitching");

    waitKey(0);
    return 0;
}

//imshow( "Result", trans_mat);


//    while(1) {
//        frame = cvQueryFrame(capture);
//        if(loop>0){
//            if(!frame) break;
//
//            image2=Mat(frame, false);
//            result=Stitching(image1,image2);
//            before_frame=result;
//            frame=&before_frame;
//            image1=result;
//            image2.release();
//            //imshow("Stitching",frame);
//            cvShowImage("Stitching",frame);
//            //break;
//
//        }else if(loop==0){
//            //Mat aimage1(frame);
//            image1=Mat(frame, false);
//        }
//        loop++;
//        char c = cvWaitKey(33);
//        if(c==27) break;
//    }
//

//perspectiveTransform(obj_corners, scene_corners, H_12);
//-- Show detected matches


//Have 2 images. I_1, I_2. Need to transofm to the other with Homography H_12.
//Now creating blnak matrix twice size of images.
//alternative with homogenious coordinates
//Mat canvas(720, 540, 1, Scalar(0));
//This is for combining images I_1, I_2.


//Mat V_12 = Mat::zeros(720,540,CV_8UC1);

//cv::warpPerspective(I_2, V_12, H_12, V_12.size( ));

//Now you have four canvases, all of which are the width of the 4 combined images,
//and with one of the images transformed into the relevant place on each.

//NOW -- merge the transformed images onto eachother. With ROI
/*
selecting the rectangle region of interest inside the image and cut or
display part of the image from the bigger picture.
*/

//set the left side to white size - 360x270.
//Mat M_1 = Mat::zeros(270,360,CV_8UC1);
//M_1.setTo(cv::Scalar(255,255,255));

//Mat M_2 = Mat::zeros(270,360,CV_8UC1);


//cv::warpPerspective(M_1, M_2, H_12, M_1.size( ));


//cv::Mat pano = Mat::zeros(M_1.size( ), CV_8UC3);

// I_1.copyTo(pano, M_1);
// V_12.copyTo(pano, M_2);
//

// imshow("Pano Homography", pano);

//Need to define M_1...
//--To later bring it all into the Pano
// cv::Mat pano = zeros(M_1.size( ), CV_8UC3);
// I_1.copyTo(pano, M_1);
// V_12.copyTo(pano, M_2);

//imshow("Pano Homography", img_matches);

//--Compile
/*

g++ panoMat.cpp -I"/usr/local/Cellar/opencv/3.4.1_5/include" -L"/usr/local/Cellar/opencv/3.4.1_5/lib/" -I/u
sr/local/include/opencv -I/usr/local/include -L/usr/local/lib -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -
lopencv_bioinspired -lopencv_ccalib -lopencv_dnn_objdetect -lopencv_dpm -lopencv_face -lopencv_photo -lopencv_fuzzy -lopencv_hfs -lopencv_img_hash -lo
pencv_line_descriptor -lopencv_optflow -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrappin
g -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_dnn -lopencv_plot -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv
_ml -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_flann -lopencv_xobjdetect -lopencv_imgcodecs -lo
pencv_objdetect -lopencv_xphoto -lopencv_imgproc -lopencv_core -o panoMat

*/

// --Run
// ./panoMat IMG_0856-2.png IMG_0857-2.png
