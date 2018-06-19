#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace cv::xfeatures2d;

void readme();

/** @function main */
int main(int argc, char** argv)
{
    if (argc != 3)
    {
        readme(); return -1;
    }

    Mat I_1 = imread(argv[1], IMREAD_GRAYSCALE);
    Mat I_2 = imread(argv[2], IMREAD_GRAYSCALE);

    if (!I_1.data || !I_2.data)
    {
        std::cout << " --(!) Error reading images " << std::endl; return -1;
    }

    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;

    Ptr<SURF> detector = SURF::create(minHessian);

    std::vector<KeyPoint> keypoints_object, keypoints_scene;

    detector->detect(I_1, keypoints_object);
    detector->detect(I_2, keypoints_scene);

    //-- Step 2: Calculate descriptors (feature vectors)
    Ptr<SURF> extractor = SURF::create();
    //SurfDescriptorExtractor extractor;

    Mat descriptors_object, descriptors_scene;

    extractor->compute(I_1, keypoints_object, descriptors_object);
    extractor->compute(I_2, keypoints_scene, descriptors_scene);

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match(descriptors_object, descriptors_scene, matches);

    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for (int i = 0; i < descriptors_object.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< DMatch > good_matches;

    for (int i = 0; i < descriptors_object.rows; i++)
    {
        if (matches[i].distance < 3 * min_dist)
        {
            good_matches.push_back(matches[i]);
        }
    }

    Mat img_matches;
    drawMatches(I_1, keypoints_object, I_2, keypoints_scene,
                good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for (int i = 0; i < good_matches.size(); i++)
    {
        //-- Get the keypoints from the good matches
        obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
        scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
    }

    Mat H_12 = findHomography(obj, scene, RANSAC);

    std::cout << H_12 << '\n';

    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(I_1.cols, 0);
    obj_corners[2] = cvPoint(I_1.cols, I_1.rows); obj_corners[3] = cvPoint(0, I_1.rows);
    std::vector<Point2f> scene_corners(4);

    //perspectiveTransform(obj_corners, scene_corners, H_12);
    //-- Show detected matches


    //Have 2 images. I_1, I_2. Need to transofm to the other with Homography H_12.
    //Now creating blnak matrix twice size of images.
    //alternative with homogenious coordinates
    //Mat canvas(720, 540, 1, Scalar(0));
    //This is for combining images I_1, I_2.
    Mat v_12 = Mat::zeros(720,540,CV_8UC1);


    //--To later bring it all into the Pano
    // cv::Mat pano = zeros(M_1.size( ), CV_8UC3);
    // I_1.copyTo(pano, M_1);
    // V_12.copyTo(pano, M_2);

    imshow("Pano Homography", img_matches);

    waitKey(0);
    return 0;
}

/** @function readme */
void readme()
{
    std::cout << " Usage: ./SURF_descriptor <img1> <img2>" << std::endl;
}

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
