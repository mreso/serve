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
target_link_libraries(cpp_backend_poc_eager PRIVATE OpenSSL::Crypto)
target_link_libraries(cpp_backend_poc_eager PRIVATE ${Boost_LIBRARIES})

#
# PyTorch
#


#find_package(TorchVision REQUIRED)
#target_include_directories(cpp_backend_poc_eager PRIVATE ${TORCHVISION_INCLUDE_DIRS})

find_package(Torch REQUIRED)
target_include_directories(cpp_backend_poc_eager PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(cpp_backend_poc_eager PRIVATE ${TORCH_LIBRARIES})


#
# Tokenizer
#
add_compile_definitions(SPDLOG_FMT_EXTERNAL)
add_library(bert_tokenizer bert_tokenizer.cpp ${RADISH_INSTALL_PREFIX}/radish/bert/bert_tokenizer.cc ${RADISH_INSTALL_PREFIX}/radish/utils/text_tokenizer.cc)
include_directories(BEFORE ${RADISH_INSTALL_PREFIX} ${UTF8PROC_INSTALL_PREFIX} ${TORCH_INCLUDE_DIRS})

find_package(fmt REQUIRED)
find_library(glog_LIBRARY_PATH libglog)

target_link_libraries(bert_tokenizer fmt::fmt ${TORCH_LIBRARIES} ${glog_LIBRARY_PATH})

find_library(utf8proc_LIBRARY_PATH libutf8proc)
target_link_libraries(bert_tokenizer ${utf8proc_LIBRARY_PATH})

target_include_directories(test_tokenizer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(test_tokenizer PRIVATE ${TORCH_LIBRARIES})


# torch::deploy currently does not add itself to TORCH_LIBRARIES it seems,
# presumably because it is in beta.  We hunt down for, and separately link
# against, the library manually here.  This step should not be necessary it
# seems to me after torch:deploy is no longer in beta.

find_library(
    TORCH_DEPLOY
    torch_deploy
    PATHS ${TORCH_INSTALL_PREFIX}/../build/lib
    REQUIRED)

target_link_libraries(cpp_backend_poc_eager PRIVATE ${TORCH_DEPLOY} bert_tokenizer)

target_link_libraries(test_tokenizer PRIVATE ${TORCH_DEPLOY} bert_tokenizer)
