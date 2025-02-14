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
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <rclcpp/logging.hpp>
#include <string>

#include "geometry_msgs/msg/point.hpp"
#include "opencv_aruco_ros_msgs/msg/marker_array.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/string.hpp"
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/detail/pose__struct.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

using namespace std::chrono_literals;

class MinimalNode : public rclcpp::Node {
public:
  MinimalNode() : Node("opencv_aruco_ros") {
    rmw_qos_profile_t custom_qos = rmw_qos_profile_default;
    sub_ = image_transport::create_subscription(
        this, "/camera/image",
        std::bind(&MinimalNode::img_callback, this, std::placeholders::_1),
        "raw", custom_qos);

    // set coordinate system
    objPoints.ptr<cv::Vec3f>(0)[0] =
        cv::Vec3f(-markerLength / 2.f, markerLength / 2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[1] =
        cv::Vec3f(markerLength / 2.f, markerLength / 2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[2] =
        cv::Vec3f(markerLength / 2.f, -markerLength / 2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[3] =
        cv::Vec3f(-markerLength / 2.f, -markerLength / 2.f, 0);

    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    cv::Size size = cv::Size(640, 480);
    vidWriter.open("vid.avi", codec, 25.0, size, true);

    RCLCPP_INFO(this->get_logger(), "running...");
  }

  ~MinimalNode() {
    if (vidWriter.isOpened()) {
      vidWriter.release();
    }
  }

private:
  // ros camera subscription
  image_transport::Subscriber sub_;

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

  // record video
  cv::VideoWriter vidWriter;

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

    // encode each marker into the ROS message type
    auto marker_array = opencv_aruco_ros_msgs::msg::MarkerArray();
    for (size_t i = 0; i < markerIds.size(); i++) {
      auto marker = opencv_aruco_ros_msgs::msg::Marker();
      marker.id = markerIds.at(i);

      for (int j = 0; j < 4; j++) {
        auto point = geometry_msgs::msg::Point();
        point.x = markerCorners.at(i).at(j).x;
        point.y = markerCorners.at(i).at(j).y;
        marker.corners[j] = point;
      }

      marker_array.markers.push_back(marker);
    }

    // copy image to draw debug info
    cv::Mat outputImage = cv_ptr->image.clone();
    cv::aruco::drawDetectedMarkers(outputImage, markerCorners, markerIds);
    vidWriter.write(outputImage);
  }
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalNode>());
  rclcpp::shutdown();
  return 0;
}
