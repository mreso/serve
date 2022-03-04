cmake_minimum_required(VERSION 3.10)

project(cpp_backend_poc_eager)
add_executable(cpp_backend_poc_eager main.cpp)
target_compile_features(cpp_backend_poc_eager PRIVATE cxx_std_14)


add_executable(test_tokenizer test_tokenizer.cpp)
target_compile_features(test_tokenizer PRIVATE cxx_std_14)


#
# Microsoft's C++ REST SDK
#

find_package(cpprestsdk REQUIRED)
target_link_libraries(cpp_backend_poc_eager PRIVATE cpprestsdk::cpprest)

# C++ REST SDK #define's the upper case letter U as a macro (!!!) which is an
# extraordinarily irresponsible thing to do!  This conflicts, among many other
# things, with the upper case letter U that is commonly used in C++ to denote
# generic template parameters, resulting in cryptic errors that are time
# consuming to root cause.  We disable this behavior here.
target_compile_definitions(cpp_backend_poc_eager PRIVATE _TURN_OFF_PLATFORM_STRING)

# C++ REST SDK does not link against its dependencies it seems so we are doing it
# manually here.
target_link_libraries(cpp_backend_poc_eager PRIVATE OpenSSL::Crypto ${Boost_LIBRARIES} bert_tokenizer)

#
# PyTorch
#


#find_package(TorchVision REQUIRED)
#target_include_directories(cpp_backend_poc_eager PRIVATE ${TORCHVISION_INCLUDE_DIRS})

find_package(Torch REQUIRED)

target_include_directories(cpp_backend_poc_eager PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(cpp_backend_poc_eager PRIVATE ${TORCH_LIBRARIES})

#
# Google logging
# 
#find_package (glog 0.3.0 REQUIRED)
find_library(GLOG_LIBRARIES libglog.so.0)


#
# Tokenizer
#
add_compile_definitions(SPDLOG_FMT_EXTERNAL)
add_library(bert_tokenizer bert_tokenizer.cpp ${RADISH_INSTALL_PREFIX}/radish/bert/bert_tokenizer.cc ${RADISH_INSTALL_PREFIX}/radish/utils/text_tokenizer.cc)
include_directories(BEFORE ${RADISH_INSTALL_PREFIX} ${UTF8PROC_INSTALL_PREFIX} ${TORCH_INCLUDE_DIRS})

find_package(fmt REQUIRED)
#find_library(glog_LIBRARY_PATH libglog)
set(glog_LIBRARY_PATH /usr/lib/x86_64-linux-gnu/libglog.so)

target_link_libraries(bert_tokenizer fmt::fmt ${TORCH_LIBRARIES} ${glog_LIBRARY_PATH})

#find_library(utf8proc_LIBRARY_PATH libutf8proc)
set(utf8proc_LIBRARY_PATH /usr/lib/x86_64-linux-gnu/libutf8proc.so)
target_link_libraries(bert_tokenizer ${utf8proc_LIBRARY_PATH})

target_include_directories(test_tokenizer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(test_tokenizer PRIVATE ${TORCH_LIBRARIES})

target_link_libraries(cpp_backend_poc_eager PRIVATE bert_tokenizer ${GLOG_LIBRARIES})

target_link_libraries(test_tokenizer PRIVATE bert_tokenizer ${TORCH_DEPLOY})


add_executable(benchmark threading_benchmark.cpp)
target_compile_features(benchmark PRIVATE cxx_std_14)

find_package(Torch REQUIRED)
target_include_directories(benchmark PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(benchmark PRIVATE ${TORCH_LIBRARIES} ${TORCH_DEPLOY})

add_executable(batched_benchmark batched_threading_benchmark.cpp)
target_compile_features(batched_benchmark PRIVATE cxx_std_14)

find_package(Torch REQUIRED)
target_include_directories(batched_benchmark PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(batched_benchmark PRIVATE ${TORCH_LIBRARIES} ${TORCH_DEPLOY})

add_executable(optimize_for_inference_benchmark optimize_for_inference_benchmark.cpp)
target_compile_features(optimize_for_inference_benchmark PRIVATE cxx_std_14)
target_include_directories(optimize_for_inference_benchmark PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(optimize_for_inference_benchmark PRIVATE ${TORCH_LIBRARIES})