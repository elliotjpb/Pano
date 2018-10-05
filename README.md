# Video Panorama

Using openCV with homography to stitch video frames. 

Here I get the first frame of each video. Then pass that to a function which works out the homography between the two. This matches the similar points in the video so they can be stitched together.

I then use this pre-calculated homography with all the rest of the video frames.

I loop through all the frames and keep the first video on the top left corner and warp the second video to combine/align with the first video based on the homography.

I also create a video writer object to export the resultant video to an avi file. 

## Little openCV tips I have found

Here defining a matrix from the array of values in data. Then in the creation of the Mat defining it as a 3x3 matrix and passing by reference the matrix values.

 `float data [9] ={1.000822995828627, 1.28264576544064e-06, 599.9977067973233,     0.0001423991871877334, 1.000058080711918, 0.001653846848108326,    1.214023035530543e-06, 2.990814694940521e-08, 1};`
    
`Mat homography = Mat(3, 3, CV_32FC1, &data);`

## Compile

`g++ panoMat.cpp -I"/usr/local/Cellar/opencv/3.4.1_5/include" -L"/usr/local/Cellar/opencv/3.4.1_5/lib/" -I/u
sr/local/include/opencv -I/usr/local/include -L/usr/local/lib -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -
lopencv_bioinspired -lopencv_ccalib -lopencv_dnn_objdetect -lopencv_dpm -lopencv_face -lopencv_photo -lopencv_fuzzy -lopencv_hfs -lopencv_img_hash -lo
pencv_line_descriptor -lopencv_optflow -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrappin
g -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_dnn -lopencv_plot -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv
_ml -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_flann -lopencv_xobjdetect -lopencv_imgcodecs -lo
pencv_objdetect -lopencv_xphoto -lopencv_imgproc -lopencv_core -o panoMat`

./panoMat
