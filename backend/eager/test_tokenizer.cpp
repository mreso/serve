// Tokenizer
#include "bert_tokenizer.h"

// PyTorch
#include <torch/torch.h>
#include <torch/types.h>
#include <torch/csrc/deploy/deploy.h>

// C++ STD
#include <cmath>
#include <iostream>
#include <map>
#include <string>
using namespace std;



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

  // Torch Deploy
  torch::deploy::InterpreterManager manager(1, python_path);
  torch::deploy::Package package = manager.loadPackage(torch_package_model);
  torch::deploy::ReplicatedObj model = package.loadPickle("model", "model.pkl");

  auto run_model = [&tokenizer, &model](string seq_1, string seq_2) {
    auto kwargs = tokenizer.encode_plus(seq_1, seq_2);
    auto ret = model.callKwargs({}, kwargs).toIValue().toTuple()->elements()[0];
    float paraphrased_percent = 100.0 * torch::softmax(ret.toTensor(),1)[0][1].item<float>();
    cout << round(paraphrased_percent) << "% paraphrase" << endl;
  };

  run_model(sequence_0, sequence_1);
  run_model(sequence_0, sequence_2);


  // Torch script
  auto traced = torch::jit::load(torch_script_model);

  auto run_traced = [&tokenizer, &traced](string seq_1, string seq_2) {
    auto kwargs = tokenizer.encode_plus(seq_1, seq_2);
    auto ret = traced.forward({}, kwargs).toIValue().toTuple()->elements()[0];
    float paraphrased_percent = 100.0 * torch::softmax(ret.toTensor(),1)[0][1].item<float>();
    cout << round(paraphrased_percent) << "% paraphrase" << endl;
  };

  run_traced(sequence_0, sequence_1);
  run_traced(sequence_0, sequence_2);
}
