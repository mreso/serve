
#include "bert_tokenizer.h"

using namespace std;

BertTokenizer::BertTokenizer(string vocab_file)
{
    assert(tokenizer_.Init("vocab.txt"));
}

BertTokenizer::~BertTokenizer()
{
}

unordered_map<string, c10::IValue> BertTokenizer::encode_plus(string sequence_1, string sequence_2, size_t sequence_length)
{
    vector<int> input_ids;
    input_ids.push_back(tokenizer_.ClsId());
    vector<int> token_type_ids;
    vector<int> attention_mask;

    auto ids = tokenizer_.Encode(sequence_1);

    for(int i : ids){
        input_ids.push_back(i);
    }
    input_ids.push_back(tokenizer_.SepId());
    token_type_ids.resize(input_ids.size(), 0);

    ids = tokenizer_.Encode(sequence_2);

    for(int i : ids)
        input_ids.push_back(i);
    input_ids.push_back(tokenizer_.SepId());

    input_ids.resize(sequence_length, tokenizer_.PadId());
    token_type_ids.resize(input_ids.size(), 1);
    attention_mask.resize(input_ids.size(), 1);

    unordered_map<string, c10::IValue> kwargs;
    kwargs["input_ids"] = torch::tensor(input_ids);
    kwargs["token_type_ids"] = torch::tensor(token_type_ids);
    kwargs["attention_mask"] = torch::tensor(attention_mask);

    return kwargs;
}
