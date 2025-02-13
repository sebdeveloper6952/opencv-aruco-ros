cmake_minimum_required(VERSION 3.8)
project(opencv_aruco_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(opencv_aruco_ros_msgs REQUIRED)

include_directories(/usr/local/include/opencv4)
link_libraries(
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_imgcodecs
    opencv_objdetect
)

add_executable(main src/main.cpp)
ament_target_dependencies(
    main
    rclcpp
    std_msgs
    sensor_msgs 
    geometry_msgs
    cv_bridge
    image_transport
    opencv_aruco_ros_msgs
)

add_executable(calib src/calib.cpp)
ament_target_dependencies(
    calib
    rclcpp
    std_msgs
    sensor_msgs 
    cv_bridge
    image_transport
)

add_executable(record src/calib.cpp)
ament_target_dependencies(
    record
    rclcpp
    std_msgs
    sensor_msgs 
    cv_bridge
    image_transport
)

install(
    TARGETS main calib record
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()
