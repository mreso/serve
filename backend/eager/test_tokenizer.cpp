// Tokenizer
#include "bert_tokenizer.h"

// PyTorch
#include <torch/torch.h>
#include <torch/types.h>
#include <torch/csrc/deploy/deploy.h>

// C++ STD
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <string>
using namespace std;

const size_t NUM_ITER = 100;

int main(const int argc, const char* const argv[]) {
  if (argc != 4) {
    std::cout << "Usage: "<< argv[0] <<" <site_packages_folder> <torch_package_model> <torch_script_model>" << std::endl
              << std::endl
              << "Example: "<< argv[0] <<" ../transformers_venv/lib/python3.8/site-packages/ ../models/bert_model_only.pt ../models/bert_model_only_traced.pt" << std::endl;
    return EXIT_FAILURE;
  }

  // Configurations
  const std::string python_path = argv[1];
  const std::string torch_package_model = argv[2];
  const std::string torch_script_model = argv[3];

  string sequence_0 = "The company HuggingFace is based in New York City";
  string sequence_1 = "Apples are especially bad for your health";
  string sequence_2 = "HuggingFace's headquarters are situated in Manhattan";

  BertTokenizer tokenizer("vocab.txt");


  // Torch script
  auto traced = torch::jit::load(torch_script_model);

  at::set_num_interop_threads(1);
  at::set_num_threads(1);

  cout << "Inter-op threads:" << at::get_num_interop_threads() << endl;
  cout << "Intra-op threads:" << at::get_num_threads() << endl; 

  traced.eval();

  auto run_traced = [&tokenizer, &traced](string seq_1, string seq_2) {
    auto kwargs = tokenizer.encode_plus(seq_1, seq_2, 128);

    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
      std::cout << "CUDA is available! Training on GPU." << std::endl;
      device = torch::kCUDA;
    }

    kwargs["input_ids"] = kwargs["input_ids"].toTensor().to(device);
    kwargs["token_type_ids"] = kwargs["token_type_ids"].toTensor().to(device);
    kwargs["attention_mask"] = kwargs["attention_mask"].toTensor().to(device);

    auto ret = traced.forward({}, kwargs).toIValue().toTuple()->elements()[0];
    float paraphrased_percent = 100.0 * torch::softmax(ret.toTensor(),1)[0][1].item<float>();
    cout << round(paraphrased_percent) << "% paraphrase" << endl;

    // WARM UP
    for(size_t i=0; i<10; ++i)
      auto ret = traced.forward({}, kwargs).toIValue().toTuple()->elements()[0];

    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    for(size_t i=0; i<NUM_ITER; ++i)
      auto ret = traced.forward({}, kwargs).toIValue().toTuple()->elements()[0];
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << "ModelTime (ms): " << chrono::duration_cast<chrono::milliseconds>(end - begin).count()/float(NUM_ITER) << endl;
  };

  run_traced(sequence_0, sequence_1);


  // Torch Deploy
  torch::deploy::InterpreterManager manager(1, python_path);
  torch::deploy::Package package = manager.loadPackage(torch_package_model);
  torch::deploy::ReplicatedObj model = package.loadPickle("model", "model.pkl");

  auto run_model = [&tokenizer, &model](string seq_1, string seq_2) {
    auto kwargs = tokenizer.encode_plus(seq_1, seq_2, 128);

    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
      std::cout << "CUDA is available! Training on GPU." << std::endl;
      device = torch::kCUDA;
    }

    kwargs["input_ids"] = kwargs["input_ids"].toTensor().to(device);
    kwargs["token_type_ids"] = kwargs["token_type_ids"].toTensor().to(device);
    kwargs["attention_mask"] = kwargs["attention_mask"].toTensor().to(device);

    auto ret = model.callKwargs({}, kwargs).toIValue().toTuple()->elements()[0];
    float paraphrased_percent = 100.0 * torch::softmax(ret.toTensor(),1)[0][1].item<float>();
    cout << round(paraphrased_percent) << "% paraphrase" << endl;
    
    // WARM UP
    for(size_t i=0; i<10; ++i)
      auto ret = model.callKwargs({}, kwargs).toIValue().toTuple()->elements()[0];

    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    for(size_t i=0; i<NUM_ITER; ++i)
      auto ret = model.callKwargs({}, kwargs).toIValue().toTuple()->elements()[0];
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << "ModelTime (ms): " << chrono::duration_cast<chrono::milliseconds>(end - begin).count()/float(NUM_ITER) << endl;
  };

  run_model(sequence_0, sequence_1);
}
