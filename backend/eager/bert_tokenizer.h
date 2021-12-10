#include <radish/bert/bert_tokenizer.h>

#include <torch/torch.h>

#include <map>
#include <string>

class BertTokenizer
{
  public:
  BertTokenizer(std::string vocab_file);

  ~BertTokenizer();

  c10::Dict<std::string, torch::Tensor> encode_plus(std::string sequence_1, std::string sequence_2, size_t sequence_length);

  protected:
  radish::BertTokenizer tokenizer_;
};
