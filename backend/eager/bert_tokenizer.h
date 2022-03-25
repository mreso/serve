#include <radish/bert/bert_tokenizer.h>

#include <torch/torch.h>

#include <map>
#include <string>

class BertTokenizer
{
  public:
  BertTokenizer(std::string vocab_file);

  ~BertTokenizer();

  std::unordered_map<std::string, c10::IValue> encode_plus(std::string sequence_1, std::string sequence_2, size_t sequence_length);

  std::unordered_map<std::string, torch::Tensor> encode_plus_tensor(std::string sequence_1, std::string sequence_2, size_t sequence_length);

  protected:
  radish::BertTokenizer tokenizer_;
};

BertTokenizer *createTokenizer();
