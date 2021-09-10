// CPP REST SDK
#include <cpprest/http_listener.h>

// PyTorch
#include <torch/torch.h>
#include <torch/types.h>
#include <torch/csrc/deploy/deploy.h>
#include <torchvision/io/image/image.h>

// C++ STD
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <thread>

using namespace web::http::experimental::listener;
using namespace web::http;

namespace {

volatile std::sig_atomic_t signal_{};

void handler(const int signal) {
  signal_ = signal;
}

} // namespace

int main(const int argc, const char* const argv[]) {
  if (argc != 4) {
    std::cout << "Usage: cpp_backend_poc_eager <uri> <model_to_serve> <thread_count>" << std::endl
              << "Serve <model_to_serve> at <uri> with <thread_count> threads." << std::endl
              << std::endl
              << "Example: cpp_backend_poc_eager http://localhost:8090 ../models/resnet 4" << std::endl;
    return EXIT_FAILURE;
  }

  if (std::signal(SIGINT, &handler) == SIG_ERR) {
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
  torch::deploy::ReplicatedObj obj = package.load_pickle("model", "model.pkl");

  torch::deploy::InterpreterSession s = obj.acquire_session();

  std::vector<c10::IValue> args;
  s.self.attr("eval")(args);

  torch::Tensor data = vision::image::read_file("kitten_small.jpg");
  torch::Tensor image = vision::image::decode_jpeg(data);

  image = image.toType(torch::kFloat32);

  std::cout << image.index({0,0,0}).item<double>() << std::endl;

  image = image.toType(torch::kFloat32);

  std::cout << image.index({0,0,0}).item<double>()<< std::endl;

  image = image.div(255.);

  std::cout << image.index({0,0,0}).item<double>()<< std::endl;

  image = image
    .sub(torch::tensor(std::vector<float>{0.485, 0.456, 0.406}).unsqueeze(-1).unsqueeze(-1))
    .div(torch::tensor(std::vector<float>{0.229, 0.224, 0.225}).unsqueeze(-1).unsqueeze(-1));

  std::cout << image.dim() <<  std::endl;

  for( size_t i=0; i<image.dim(); ++i)
    std::cout << image.size(i) <<  " ";
  std::cout << std::endl;

  torch::Tensor result = obj({image.unsqueeze(0)}).toTensor();

  std::cout << result.index({0}).argmax(0).item<int>() << std::endl;

  // HTTP Server
  http_listener listener(uri);
  listener.support(methods::GET, [](const http_request& request) {
    std::cout << "Received a GET request!" << std::endl;
  });
  listener.open().wait();

  while (signal_ != SIGINT) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }

  std::cout << "Shutting down ..." << std::endl;
  listener.close();

  return EXIT_SUCCESS;
}
