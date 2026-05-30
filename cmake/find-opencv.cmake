find_package(OpenCV REQUIRED core imgproc imgcodecs)

if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
    set(OPENCV_LIBS ${OpenCV_LIBS} CACHE INTERNAL "OpenCV core libraries")
    message(STATUS "OpenCV libs: ${OPENCV_LIBS}")
endif()