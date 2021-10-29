// CPP REST SDK
#include <cpprest/http_listener.h>
#include <cpprest/json.h>

// PyTorch
#include <torch/torch.h>
#include <torch/types.h>
#include <torch/csrc/deploy/deploy.h>

using namespace web::http::experimental::listener;
using namespace web::http;
using namespace web;

// C++ STD
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <thread>
#include <map>
#include <string>
#include <iostream>
#include <set>
using namespace std;

#define TRACE(msg)            cout << msg
#define TRACE_ACTION(a, k, v) cout << a << " (" << k << ", " << v << ")\n"

map<utility::string_t, utility::string_t> dictionary;

namespace {

volatile std::sig_atomic_t signal_{};

void signal_handler(const int signal) {
  signal_ = signal;
}

} // namespace

void display_json(
   json::value const & jvalue,
   utility::string_t const & prefix)
{
   cout << prefix << jvalue.serialize() << endl;
   // cout << jvalue.serialize() << endl;
}

void handle_request(
   http_request request,
   torch::deploy::ReplicatedObj &model_hander)
{
   utility::string_t answer;
   request
      .extract_string()
      .then([&answer, &model_hander](pplx::task<utility::string_t> task) {
         try
         {
            std::vector<c10::IValue> args{task.get()};
            std::unordered_map<std::string, c10::IValue> kwargs;

            auto ret = model_hander.callKwargs(args, kwargs);

            cout << ret.toIValue() << endl;

            answer = ret.toIValue().toString();
         }
         catch (http_exception const & e)
         {
            wcout << e.what() << endl;
         }
      })
      .wait();

   request.reply(status_codes::OK, answer);
}

int main(const int argc, const char* const argv[]) {
  if (argc != 5) {
    std::cout << "Usage: cpp_backend_poc_eager <uri> <model_to_serve> <python_path> <thread_count>" << std::endl
              << "Serve <model_to_serve> at <uri> with <thread_count> threads." << std::endl
              << std::endl
              << "Example: cpp_backend_poc_eager http://localhost:8090 ../models/resnet venv/lib/python3.8/site-packages/ 4" << std::endl;
    return EXIT_FAILURE;
  }

  if (std::signal(SIGINT, &signal_handler) == SIG_ERR) {
    std::cerr << "Failed to register signal handler!" << std::endl;
    return EXIT_FAILURE;
  }

  // Configurations
  const std::string uri = argv[1];
  const std::string model_to_serve = argv[2];
  const std::string python_path = argv[3];
  const size_t thread_count = std::stoul(argv[4]);

  std::cout << "Serving " << model_to_serve << " at " << uri << " with " << thread_count << " threads." << std::endl
            << "Press Ctrl+C to terminate." << std::endl;

  // Torch Deploy


  torch::deploy::InterpreterManager manager(thread_count, python_path);
  torch::deploy::Package package = manager.loadPackage(model_to_serve);
  torch::deploy::ReplicatedObj handler = package.loadPickle("handler", "handler.pkl");


   // HTTP Server
   http_listener listener(uri);
   listener.support(methods::GET, [](const http_request& request) {
      std::cout << "Received a GET request!"  << std::endl;
      if(request.relative_uri().path() == "/ping") {
         request.reply(status_codes::OK, "PING");
      }else {
         request.reply(status_codes::OK);
      }
   });

   listener.support(methods::POST, [&handler](const http_request& request) {
      std::cout << "Received a POST request!" << std::endl;
      handle_request(
         request,
         handler
      );

   });

   listener.support(methods::PUT, [&handler](const http_request& request) {
      std::cout << "Received a PUT request!" << std::endl;
      handle_request(
         request,
         handler);
   });

  listener.open().wait();

  while (signal_ != SIGINT) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }

  std::cout << "Shutting down ..." << std::endl;
  listener.close();

  return EXIT_SUCCESS;
}
