# Include path
#include_directories(${CMAKE_SOURCE_DIR}/inc)
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Dependency libraries
find_package(OpenCV REQUIRED)

add_executable(test test.cpp)
target_link_libraries(test pencil ${OpenCV_LIBS})
add_executable(demo demo.cpp)
target_link_libraries(demo pencil ${OpenCV_LIBS})

