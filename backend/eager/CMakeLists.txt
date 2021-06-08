cmake_minimum_required(VERSION 3.10)

project(cpp_backend_poc_eager)
add_executable(cpp_backend_poc_eager main.cpp)
target_compile_features(cpp_backend_poc_eager PRIVATE cxx_std_14)

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
