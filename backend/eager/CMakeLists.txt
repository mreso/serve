cmake_minimum_required(VERSION 3.10)

project(cpp_backend_poc_eager)
add_executable(cpp_backend_poc_eager main.cpp)
target_compile_features(cpp_backend_poc_eager PRIVATE cxx_std_14)

#
# Microsoft's C++ REST SDK
#

target_include_directories(cpp_backend_poc_eager PRIVATE $ENV{cpprestsdk_DIR}/include)
# target_link_libraries(cpp_backend_poc_eager PRIVATE cpprestsdk::cpprest)

#
# PyTorch
#

find_package(Torch REQUIRED)
target_include_directories(cpp_backend_poc_eager PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(cpp_backend_poc_eager PRIVATE ${TORCH_LIBRARIES})

# torch::deploy currently does not add itself to TORCH_LIBRARIES it seems,
# presumably because it is in beta.  We hunt down for, and separately link
# against, the library manually here.  This step should not be necessary it
# seems to me after torch:deploy is no longer in beta.

find_library(
    TORCH_DEPLOY
    torch_deploy
    PATHS ${TORCH_INSTALL_PREFIX}/../build/lib
    REQUIRED)

target_link_libraries(cpp_backend_poc_eager PRIVATE ${TORCH_DEPLOY})
