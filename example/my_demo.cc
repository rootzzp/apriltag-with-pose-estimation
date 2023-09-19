/* Copyright (C) 2013-2016, The Regents of The University of Michigan.
All rights reserved.
This software was developed in the APRIL Robotics Lab under the
direction of Edwin Olson, ebolson@umich.edu. This software may be
available under alternative licensing terms; contact the address above.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the Regents of The University of Michigan.
*/

#include <iostream>
#include "opencv2/opencv.hpp"
#include <eigen3/Eigen/Dense>  //necessary before including the opencv eigen lib
#include <opencv2/core/eigen.hpp> //include linear calculation lib

extern "C" {
#include "apriltag.h"
#include "apriltag_pose.h" //pose estimation lib
#include "tag36h11.h"
#include "tag25h9.h"
#include "tag16h5.h"
#include "tagCircle21h7.h"
#include "tagCircle49h12.h"
#include "tagCustom48h12.h"
#include "tagStandard41h12.h"
#include "tagStandard52h13.h"
#include "common/getopt.h"
}

using namespace std;
using namespace cv;


int main(int argc, char *argv[])
{
    getopt_t *getopt = getopt_create();

    getopt_add_bool(getopt, 'h', "help", 0, "Show this help");
    getopt_add_bool(getopt, 'd', "debug", 0, "Enable debugging output (slow)");
    getopt_add_bool(getopt, 'q', "quiet", 0, "Reduce output");
    getopt_add_string(getopt, 'f', "family", "tag36h11", "Tag family to use");
    getopt_add_int(getopt, 't', "threads", "1", "Use this many CPU threads");
    getopt_add_double(getopt, 'x', "decimate", "2.0", "Decimate input image by this factor");
    getopt_add_double(getopt, 'b', "blur", "0.0", "Apply low-pass blur to input");
    getopt_add_bool(getopt, '0', "refine-edges", 1, "Spend more time trying to align edges of tags");
    getopt_add_string(getopt, 'a', "axis", "world", "Choose axis to use (world/camera)");


    if (!getopt_parse(getopt, argc, argv, 1) ||
            getopt_get_bool(getopt, "help")) {
        printf("Usage: %s [options]\n", argv[0]);
        getopt_do_usage(getopt);
        exit(0);
    }


    // Initialize tag detector with options
    apriltag_family_t *tf = NULL;
    const char *famname = getopt_get_string(getopt, "family");
    if (!strcmp(famname, "tag36h11")) {
        tf = tag36h11_create();
    } else if (!strcmp(famname, "tag25h9")) {
        tf = tag25h9_create();
    } else if (!strcmp(famname, "tag16h5")) {
        tf = tag16h5_create();
    } else if (!strcmp(famname, "tagCircle21h7")) {
        tf = tagCircle21h7_create();
    } else if (!strcmp(famname, "tagCircle49h12")) {
        tf = tagCircle49h12_create();
    } else if (!strcmp(famname, "tagStandard41h12")) {
        tf = tagStandard41h12_create();
    } else if (!strcmp(famname, "tagStandard52h13")) {
        tf = tagStandard52h13_create();
    } else if (!strcmp(famname, "tagCustom48h12")) {
        tf = tagCustom48h12_create();
    } else {
        printf("Unrecognized tag family name. Use e.g. \"tag36h11\".\n");
        exit(-1);
    }

    const char *axisname = getopt_get_string(getopt, "axis");
    unsigned int coordinate;
    if(!strcmp(axisname,"world"))
        coordinate = 1;
    else
        coordinate = 0;
    coordinate = 1;

//  input camera intrinsic matrix
    apriltag_detection_info_t info;
    info.tagsize = 0.0625;   //The size of Tag
    info.det = NULL;
    info.fx = 591.797;   //focal length of x
    info.fy = 591.829;   //focal length of y
    info.cx = 373.161;
    info.cy = 203.246;

    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family(td, tf);
    td->quad_decimate = getopt_get_double(getopt, "decimate");
    td->quad_sigma = getopt_get_double(getopt, "blur");
    td->nthreads = getopt_get_int(getopt, "threads");
    td->debug = getopt_get_bool(getopt, "debug");
    td->refine_edges = getopt_get_bool(getopt, "refine-edges");

    Mat frame, gray;
    auto file = "../TY0913/-5.png";
    frame = imread(file);
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    // Make an image_u8_t header for the Mat data
    image_u8_t im = { .width = gray.cols,
        .height = gray.rows,
        .stride = gray.cols,
        .buf = gray.data
    };

    // Draw detection outlines
    for (int i = 0; i < 1; i++) {
        apriltag_detection_t *det = new apriltag_detection_t;
        det->p[0][0] = 314;
        det->p[0][1] = 93;
        det->p[1][0] = 422;
        det->p[1][1] = 79;
        det->p[2][0] = 449;
        det->p[2][1] = 186;
        det->p[3][0] = 338;
        det->p[3][1] = 207;
        vector<Point2f> pts_src;
        pts_src.push_back(Point2f(314, 93));
        pts_src.push_back(Point2f(422, 79));
        pts_src.push_back(Point2f(449, 186));
        pts_src.push_back(Point2f(338, 207));
        vector<Point2f> pts_dst;
        pts_dst.push_back(Point2f(-1, -1));
        pts_dst.push_back(Point2f(1, -1));
        pts_dst.push_back(Point2f(1, 1));
        pts_dst.push_back(Point2f(-1, 1));
        Mat h = Mat( 3, 3,CV_32FC1,Scalar( 0));
        h = findHomography(pts_dst,pts_src);
        //cout<<h<<endl;
        Vec3f pt(314, 93, 1.0);
        //cout<<h.inv()<<endl;
        Mat inv = h.inv();
        inv.convertTo(inv, CV_32F);
        auto res = inv * pt;
        //cout<<res<<endl;

        det->H = matd_create(3, 3);
        MATD_EL(det->H, 0, 0) = h.at<float>(0, 0);
        MATD_EL(det->H, 0, 1) = h.at<float>(0, 1);
        MATD_EL(det->H, 0, 2) = h.at<float>(0, 2);
        MATD_EL(det->H, 1, 0) = h.at<float>(1, 0);
        MATD_EL(det->H, 1, 1) = h.at<float>(1, 1);
        MATD_EL(det->H, 1, 2) = h.at<float>(1, 2);
        MATD_EL(det->H, 2, 0) = h.at<float>(2, 0);
        MATD_EL(det->H, 2, 1) = h.at<float>(2, 1);
        MATD_EL(det->H, 2, 2) = h.at<float>(2, 2);
 

        info.det = det;
        apriltag_pose_t pose;
        estimate_tag_pose(&info, &pose);
        //calculate cam position in world coordinate
        if (coordinate)
        {
            Mat rvec(3, 3, CV_64FC1, pose.R->data); //rotation matrix
            Mat tvec(3, 1, CV_64FC1, pose.t->data); //translation matrix
            cout<<rvec<<endl;
            cout<<tvec<<endl;
            Mat Pos(3, 3, CV_64FC1);
            Pos = rvec.inv() * tvec;
            cout << "x: " << Pos.ptr<double>(0)[0] << endl;
            cout << "y: " << Pos.ptr<double>(1)[0] << endl;
            cout << "z: " << Pos.ptr<double>(2)[0] << endl;
            cout << "-----------world--------------" << endl;
        }
        else
        {
            cout << "x: " << pose.t ->data[0] << endl;
            cout << "y: " << pose.t ->data[1] << endl;
            cout << "z: " << pose.t ->data[2] << endl;
            cout << "-----------camera-------------" << endl;
        }

        //draw the line and show tag ID
        line(frame, Point(det->p[0][0], det->p[0][1]),
                    Point(det->p[1][0], det->p[1][1]),
                    Scalar(0, 0xff, 0), 2);
        line(frame, Point(det->p[0][0], det->p[0][1]),
                    Point(det->p[3][0], det->p[3][1]),
                    Scalar(0, 0, 0xff), 2);
        line(frame, Point(det->p[1][0], det->p[1][1]),
                    Point(det->p[2][0], det->p[2][1]),
                    Scalar(0xff, 0, 0), 2);
        line(frame, Point(det->p[2][0], det->p[2][1]),
                    Point(det->p[3][0], det->p[3][1]),
                    Scalar(0xff, 0, 0), 2);
    }

    imshow("Tag Detections", frame);
    waitKey(0);

    apriltag_detector_destroy(td);

    if (!strcmp(famname, "tag36h11")) {
        tag36h11_destroy(tf);
    } else if (!strcmp(famname, "tag25h9")) {
        tag25h9_destroy(tf);
    } else if (!strcmp(famname, "tag16h5")) {
        tag16h5_destroy(tf);
    } else if (!strcmp(famname, "tagCircle21h7")) {
        tagCircle21h7_destroy(tf);
    } else if (!strcmp(famname, "tagCircle49h12")) {
        tagCircle49h12_destroy(tf);
    } else if (!strcmp(famname, "tagStandard41h12")) {
        tagStandard41h12_destroy(tf);
    } else if (!strcmp(famname, "tagStandard52h13")) {
        tagStandard52h13_destroy(tf);
    } else if (!strcmp(famname, "tagCustom48h12")) {
        tagCustom48h12_destroy(tf);
    }

    getopt_destroy(getopt);
    return 0;
}
