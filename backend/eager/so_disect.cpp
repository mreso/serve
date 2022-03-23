#include <dlfcn.h>
#include <iostream>

#include "handler.h"


int main(void) {
    void *handle;
    handle = dlopen("libhandler.so", RTLD_LAZY);

    if(!handle){
        std::cerr << "Could not open library\n";
        return EXIT_FAILURE;
    }

    create_t* create_handler = (create_t*)dlsym(handle, "create");

    if(!create_handler) {
        std::cerr << "Could find create handler\n";
        return EXIT_FAILURE;
    }

    destroy_t* destroy_handler = (destroy_t*)dlsym(handle, "destroy");

    if(!destroy_handler) {
        std::cerr << "Could find destroy handler\n";
        return EXIT_FAILURE;
    }

    IHandler* h = create_handler();

    h->handle("this");


    destroy_handler(h);
}