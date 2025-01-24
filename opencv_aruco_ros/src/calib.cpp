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

#include <chrono>
#include <functional>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/string.hpp"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>

/* This example creates a subclass of Node and uses std::bind() to register a
 * member function as a callback from the timer. */

class MinimalNode : public rclcpp::Node {
public:
  MinimalNode() : Node("opencv_aruco_ros"), count_(0) {
    rmw_qos_profile_t custom_qos = rmw_qos_profile_default;
    sub_ = image_transport::create_subscription(
        this, "/camera/image",
        std::bind(&MinimalNode::img_callback, this, std::placeholders::_1),
        "raw", custom_qos);
    RCLCPP_INFO(this->get_logger(), "calibration started");
  }

  ~MinimalNode() {}

private:
  image_transport::Subscriber sub_;
  size_t count_;
  std::string window_name_ = "frame";

  // charuco params and calibration data that lives for the duration of the
  // entire calibration
  int squaresX = 5;
  int squaresY = 7;
  int squareLength = 200;
  int markerLength = 120;
  cv::aruco::Dictionary dictionary =
      cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
  cv::aruco::CharucoParameters charucoParams = cv::aruco::CharucoParameters();
  cv::aruco::DetectorParameters detectorParams =
      cv::aruco::DetectorParameters();
  cv::aruco::CharucoBoard board = cv::aruco::CharucoBoard(
      cv::Size(squaresX, squaresY), squareLength, markerLength, dictionary);
  cv::aruco::CharucoDetector detector =
      cv::aruco::CharucoDetector(board, charucoParams, detectorParams);
  std::vector<cv::Mat> allCharucoCorners, allCharucoIds;
  std::vector<std::vector<cv::Point2f>> allImagePoints;
  std::vector<std::vector<cv::Point3f>> allObjectPoints;
  std::vector<cv::Mat> allImages;
  cv::Size imageSize;

  // time
  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();

  template <class result_t = std::chrono::milliseconds,
            class clock_t = std::chrono::steady_clock,
            class duration_t = std::chrono::milliseconds>
  auto since(std::chrono::time_point<clock_t, duration_t> const &start) {
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
  }

  // img_callback is called once per ROS update with a new image frame every
  // call
  void img_callback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    // first we get a frame from ROS and convert it to OpenCV Mat
    cv_bridge::CvImagePtr cv_ptr;
    cv::Mat frame;
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
      frame = cv_ptr->image.clone();
    } catch (cv_bridge::Exception &e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    // if not enough time has passed, wait
    auto capture = since(begin).count() > 10000;
    if (capture) {
      begin = std::chrono::steady_clock::now();
    }

    // charuco calibration
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
    cv::Mat currentCharucoCorners, currentCharucoIds;
    std::vector<cv::Point3f> currentObjectPoints;
    std::vector<cv::Point2f> currentImagePoints;
    detector.detectBoard(frame, currentCharucoCorners, currentCharucoIds);

    if (capture && currentCharucoCorners.total() > 3) {
      board.matchImagePoints(currentCharucoCorners, currentCharucoIds,
                             currentObjectPoints, currentImagePoints);
      if (currentImagePoints.empty() || currentObjectPoints.empty()) {
        RCLCPP_INFO(this->get_logger(), "Point matching failed, try again.");
        return;
      }

      allCharucoCorners.push_back(currentCharucoCorners);
      allCharucoIds.push_back(currentCharucoIds);
      allImagePoints.push_back(currentImagePoints);
      allObjectPoints.push_back(currentObjectPoints);
      allImages.push_back(frame);
      imageSize = frame.size();

      RCLCPP_INFO(this->get_logger(),
                  "frame captured, have %ld good frames so far",
                  allImages.size());
    }

    if (allImages.size() >= 5) {
      cv::Mat cameraMatrix, distCoeffs;
      double repError = cv::calibrateCamera(
          allObjectPoints, allImagePoints, imageSize, cameraMatrix, distCoeffs,
          cv::noArray(), cv::noArray(), cv::noArray(), cv::noArray(),
          cv::noArray(), 0);
      saveCameraParams("camera_calibration.yaml", imageSize,
                       frame.size().aspectRatio(), 0, cameraMatrix, distCoeffs,
                       repError);
      RCLCPP_INFO(this->get_logger(), "calibration done, error: %f", repError);
    }
  }

  static bool saveCameraParams(const std::string &filename, cv::Size imageSize,
                               float aspectRatio, int flags,
                               const cv::Mat &cameraMatrix,
                               const cv::Mat &distCoeffs, double totalAvgErr) {
    cv::FileStorage fs = cv::FileStorage(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened())
      return false;

    time_t tt;
    time(&tt);
    struct tm *t2 = localtime(&tt);
    char buf[1024];
    strftime(buf, sizeof(buf) - 1, "%c", t2);

    fs << "calibration_time" << buf;

    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;

    if (flags & cv::CALIB_FIX_ASPECT_RATIO)
      fs << "aspectRatio" << aspectRatio;

    if (flags != 0) {
      sprintf(
          buf, "flags: %s%s%s%s",
          flags & cv::CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
          flags & cv::CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
          flags & cv::CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
          flags & cv::CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
    }

    fs << "flags" << flags;
    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs << "avg_reprojection_error" << totalAvgErr;

    return true;
  }
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalNode>());
  rclcpp::shutdown();
  return 0;
}
