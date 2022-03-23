#pragma once

class IHandler {
    public:
    IHandler(){}

    virtual ~IHandler() {}

    virtual void handle(const char*) const = 0;
};

typedef IHandler* create_t();
typedef void destroy_t(IHandler*);