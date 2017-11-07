# Include path
#include_directories(${CMAKE_SOURCE_DIR}/inc)
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Dependency libraries
find_package(OpenCV REQUIRED)

# Add libraries
set(PENCIL_DRAWING
	pencil_drawing.cpp
)
add_library(pencil STATIC ${PENCIL_DRAWING})
target_link_libraries(pencil ${OpenCV_LIBS})

# Add executables
#add_executable(test_graph test_graph.cpp)
#target_link_libraries(test_graph h2c graph_vis ${OpenCV_LIBS})
