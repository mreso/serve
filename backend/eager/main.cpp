// CPP REST SDK
#include <cpprest/http_listener.h>
#include <cpprest/json.h>

// PyTorch
#include <torch/torch.h>
#include <torch/types.h>
#include <torch/csrc/deploy/deploy.h>
#include <torchvision/io/image/image.h>

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
            // auto const & jvalue = task.get();

            // std::tojvalue << endl;

            // std::string sequence_1 = "Apples are especially bad for your health";
            // std::string sequence_0 = "Eating apples is a health risk";

            // std::vector<c10::IValue> args{"{\"sequence_1\": \"Apples are especially bad for your health\", \"sequence_0\": \"Eating apples is a health risk\"}"};
            std::vector<c10::IValue> args{task.get()};
            std::unordered_map<std::string, c10::IValue> kwargs;

            cout << "WILL IT RETURN?"<<endl;

            auto ret = model_hander.call_kwargs(args, kwargs);

            cout << "YES?"<<endl;

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
  if (argc != 4) {
    std::cout << "Usage: cpp_backend_poc_eager <uri> <model_to_serve> <thread_count>" << std::endl
              << "Serve <model_to_serve> at <uri> with <thread_count> threads." << std::endl
              << std::endl
              << "Example: cpp_backend_poc_eager http://localhost:8090 ../models/resnet 4" << std::endl;
    return EXIT_FAILURE;
  }

  if (std::signal(SIGINT, &signal_handler) == SIG_ERR) {
    std::cerr << "Failed to register signal handler!" << std::endl;
    return EXIT_FAILURE;
  }

  // Configurations
  const std::string uri = argv[1];
  const std::string model_to_serve = argv[2];
  const size_t thread_count = std::stoul(argv[3]);

  std::cout << "Serving " << model_to_serve << " at " << uri << " with " << thread_count << " threads." << std::endl
            << "Press Ctrl+C to terminate." << std::endl;

  // Torch Deploy
  torch::deploy::InterpreterManager manager(thread_count);
  torch::deploy::Package package = manager.load_package(model_to_serve);
  torch::deploy::ReplicatedObj handler = package.load_pickle("handler", "handler.pkl");


// std::vector<c10::IValue> args{"{\"sequence_1\": \"Apples are especially bad for your health\", \"sequence_0\": \"Eating apples is a health risk\"}"};
// std::unordered_map<std::string, c10::IValue> kwargs;

// auto ret = handler.call_kwargs(args, kwargs);
// std::cout << ret.toIValue() << std::endl;

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
