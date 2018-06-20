# Testing Panoramas


Currently showing this:

![Image](https://github.com/elliotjpb/Pano/blob/master/example.jpg?raw=true | width=200)

Compile

`g++ panoMat.cpp -I"/usr/local/Cellar/opencv/3.4.1_5/include" -L"/usr/local/Cellar/opencv/3.4.1_5/lib/" -I/u
sr/local/include/opencv -I/usr/local/include -L/usr/local/lib -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -
lopencv_bioinspired -lopencv_ccalib -lopencv_dnn_objdetect -lopencv_dpm -lopencv_face -lopencv_photo -lopencv_fuzzy -lopencv_hfs -lopencv_img_hash -lo
pencv_line_descriptor -lopencv_optflow -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrappin
g -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_dnn -lopencv_plot -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv
_ml -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_flann -lopencv_xobjdetect -lopencv_imgcodecs -lo
pencv_objdetect -lopencv_xphoto -lopencv_imgproc -lopencv_core -o panoMat`


--Run
./panoMat IMG_0856-2.png IMG_0857-2.png
