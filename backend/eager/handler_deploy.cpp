// PyTorch
#include <torch/torch.h>
#include <torch/types.h>
#include <torch/csrc/deploy/deploy.h>
#include <torch/csrc/deploy/path_environment.h>

#include <iostream>

int main(int argc, char* argv[]) {

    if (argc != 2) {
      std::cout << "Usage: "<< argv[0] <<" python_path " << std::endl;
      return EXIT_FAILURE;
   }

    auto env = std::make_shared<torch::deploy::PathEnvironment>(argv[1]);
    auto m = std::make_shared<torch::deploy::InterpreterManager>(2, env);

    auto I = m->acquireOne();

    auto realpath = I.global("os", "path").attr("expanduser")({"../handler.py"});
    
    //Alternative would be importlib.import_module which is also used in TorchServe Python backend
    auto h = I.global("runpy", "run_path")({realpath}).attr("__getitem__")({"Handler"})({"context"});

    auto ret = h.attr("handle")({"this"});

    std::cout << "Handled: " << ret.toIValue() << std::endl;
}