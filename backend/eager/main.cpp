// CPP REST SDK
#include <cpprest/http_listener.h>

// PyTorch
#include <torch/torch.h>
#include <torch/csrc/deploy/deploy.h>

int main(const int argc, const char* const argv[]) {
  if (argc != 3) {
    std::cout << "Usage: cpp_backend_poc_eager <model_to_serve> <thread_count>" << std::endl;
    return -1;
  }

  // Configurations
  const std::string model_to_serve = argv[1];
  const size_t thread_count = std::stoul(argv[2]);

  std::cout << "Serving " << model_to_serve << " with " << thread_count << " threads." << std::endl;

  torch::deploy::InterpreterManager manager(thread_count);
  torch::deploy::Package package = manager.load_package(model_to_serve);
  torch::deploy::ReplicatedObj obj = package.load_pickle("model", "model.pkl");

  const auto input = torch::IValue(torch::ones({1, 3, 224, 224}));
  const auto output = obj(input);

  std::cout << output << std::endl;

  return 0;
}
