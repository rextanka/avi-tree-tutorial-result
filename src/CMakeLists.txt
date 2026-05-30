add_library(avl_tree_lib STATIC avl_tree.cpp)
target_include_directories(avl_tree_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)