// Copyright 2016 Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdlib>
#include <functional>
#include <geometry_msgs/msg/detail/pose__struct.hpp>
#include <memory>
#include <opencv2/core.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/publisher.hpp>
#include <string>

#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/string.hpp"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

using namespace std::chrono_literals;

class MinimalNode : public rclcpp::Node {
public:
  MinimalNode() : Node("opencv_aruco_ros") {
    cv::namedWindow("window");
    rmw_qos_profile_t custom_qos = rmw_qos_profile_default;
    sub_ = image_transport::create_subscription(
        this, "/camera/image",
        std::bind(&MinimalNode::img_callback, this, std::placeholders::_1),
        "raw", custom_qos);
    publisher_ = this->create_publisher<geometry_msgs::msg::Pose>(
        "/opencv_aruco_ros/pose", 10);

    // load camera calibration params
    if (!readCameraParameters("camera_calibration.yaml", camMatrix,
                              distCoeffs)) {
      RCLCPP_ERROR(this->get_logger(),
                   "could not load camera calibration parameters.");
      exit(1);
    }

    // set coordinate system
    objPoints.ptr<cv::Vec3f>(0)[0] =
        cv::Vec3f(-markerLength / 2.f, markerLength / 2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[1] =
        cv::Vec3f(markerLength / 2.f, markerLength / 2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[2] =
        cv::Vec3f(markerLength / 2.f, -markerLength / 2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[3] =
        cv::Vec3f(-markerLength / 2.f, -markerLength / 2.f, 0);

    RCLCPP_INFO(this->get_logger(), "running...");
  }

  ~MinimalNode() { cv::destroyWindow("window"); }

private:
  // ros camera subscription
  image_transport::Subscriber sub_;

  // pose publisher
  rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr publisher_;

  // aruco utilities
  std::vector<int> markerIds;
  std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
  cv::aruco::DetectorParameters detectorParams =
      cv::aruco::DetectorParameters();
  cv::aruco::Dictionary dictionary =
      cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
  cv::aruco::ArucoDetector detector =
      cv::aruco::ArucoDetector(dictionary, detectorParams);

  // set coordinate system
  double markerLength = 200.0f;
  cv::Mat objPoints = cv::Mat(4, 1, CV_32FC3);

  // camera calibration params
  cv::Mat camMatrix, distCoeffs;

  void img_callback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
    } catch (cv_bridge::Exception &e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    // aruco marker detection
    detector.detectMarkers(cv_ptr->image, markerCorners, markerIds,
                           rejectedCandidates);

    // copy image to draw debug info
    cv::Mat outputImage = cv_ptr->image.clone();
    cv::aruco::drawDetectedMarkers(outputImage, markerCorners, markerIds);

    // pose estimation
    std::vector<cv::Vec3d> rvecs(markerCorners.size()),
        tvecs(markerCorners.size());
    if (markerIds.size() > 0) {
      cv::solvePnP(objPoints, markerCorners.at(0), camMatrix, distCoeffs,
                   rvecs.at(0), tvecs.at(0));
      cv::drawFrameAxes(outputImage, camMatrix, distCoeffs, rvecs.at(0),
                        tvecs.at(0), markerLength * 1.5f, 2);

      auto pose = geometry_msgs::msg::Pose();
      // position
      pose.position.set__x(tvecs.at(0).val[0]);
      pose.position.set__y(tvecs.at(0).val[1]);
      pose.position.set__z(tvecs.at(0).val[2]);
      // orientation
      cv::Matx33d rodrigues_in;
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          rodrigues_in(i, j) = rvecs.at(0)[j];
        }
      }
      cv::Matx31d rodrigues_out;
      cv::Rodrigues(rodrigues_in, rodrigues_out);
      cv::Matx41d out = to_quaternion(rodrigues_out);
      pose.orientation.set__x(out.val[0]);
      pose.orientation.set__y(out.val[1]);
      pose.orientation.set__z(out.val[2]);
      pose.orientation.set__w(out.val[3]);
      // publish the Pose message
      publisher_->publish(pose);
    }

    // for aiding in debugging, draw stuff on the output frame
    cv::imshow("window", outputImage);
    cv::waitKey(1);
  }

  // readCameraParameters will read a YAML file `filename` and fill the matrices
  // `camMatrix` & `distCoeffs` with the values read from file
  bool readCameraParameters(std::string filename, cv::Mat &camMatrix,
                            cv::Mat &distCoeffs) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
      return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
  }

  // https://answers.opencv.org/question/209974/aruco-markers-rvec-to-quaternion-and-180-degrees-flip/
  cv::Matx41d to_quaternion(const cv::Matx31d &in) {
    double angle = cv::norm(in);
    cv::Matx31d axis(in(0) / angle, in(1) / angle, in(2) / angle);
    double angle_2 = angle / 2;
    cv::Matx41d q(axis(0) * sin(angle_2), axis(1) * sin(angle_2),
                  axis(2) * sin(angle_2), cos(angle_2));
    return q;
  }
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalNode>());
  rclcpp::shutdown();
  return 0;
}
