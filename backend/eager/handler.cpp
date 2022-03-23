#include "handler.h"

#include <iostream>

class Handler : public IHandler {
public:
    Handler() {
        std::cout << "Init handler\n";
    }

    virtual ~Handler() {
        std::cout << "Destroy handler\n";
    }

    virtual void handle(const char* s) const{
        std::cout << s << " got handled!\n";
    }
};

extern "C" IHandler* create() {
    return new Handler;
}

extern "C" void destroy(IHandler* obj) {
    delete obj;
}