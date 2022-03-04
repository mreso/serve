// PyTorch
#include <torch/torch.h>
#include <torch/types.h>
#include <torch/script.h>

// C++ STD
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <list>
#include <map>
#include <string>
#include <thread>

using namespace std;


typedef  unordered_map<string, c10::IValue> KWARG;

int warmup_batch_num = 100;

struct BatchQueue {
   BatchQueue() {
   }
   
   ~BatchQueue() {
   }

   void enqueue(vector<KWARG> batch) {
         batches_.push(std::move(batch));
   }

   vector<KWARG> dequeue()
   {
       if(batches_.size()==0)
        return vector<KWARG>();
       
       vector<KWARG> batch;
       bool terminate = true;
       
       batch = std::move(batches_.front());
       batches_.pop();
    //    if(batches_.size() % 100 == 0)
    //     cout << "Queue size: " << batches_.size() << endl;

       return batch;
   }

   size_t size() {
       return batches_.size();
   }
   
   queue<vector<KWARG>> batches_;
};


class TorchScriptWorker{
    public:
    TorchScriptWorker(shared_ptr<BatchQueue> queue, bool optimize)
    :queue_(queue){
        for(int i=0; i<1;++i)
            threads_.emplace_back([this, i, optimize]{process(i, optimize);});
    }

    ~TorchScriptWorker(){
        for(auto &t : threads_)
            t.join();

        float sum = accumulate(model_times.begin(), model_times.end(), 0);

        float mean = sum / float(model_times.size());

        float stdev = 0;

        for(float v : model_times) {
            stdev += pow(v - mean, 2);
        }

        stdev = sqrt(stdev / float(model_times.size()));


        cout << "Average batch time: " <<  mean  << " ms (+- "<< stdev <<")\n";
    }

    void process(int idx, bool optimize){

        auto model = torch::jit::load("../models/bert_model_only_traced.pt");
        model.eval();
        if(optimize) {
            cout << "Applying optimize_for_inference()\n";
            model = torch::jit::optimize_for_inference(model);
        }

        int batch_idx=0;

        while(true) {
            vector<KWARG> batch = queue_->dequeue();
            if(batch.size() == 0)
                break;

            do_work_on_batch(model, move(batch), idx, batch_idx++);
        }
    }

    void do_work_on_batch(torch::jit::script::Module& model, vector<KWARG> batch, int idx, int batch_idx){
         vector<torch::Tensor> input_ids, token_type_ids, attention_mask;

        for(auto &kw : batch) {
            input_ids.push_back(kw["input_ids"].toTensor());
            token_type_ids.push_back(kw["token_type_ids"].toTensor());
            attention_mask.push_back(kw["attention_mask"].toTensor());
        }

        KWARG input_data;
        
        input_data["input_ids"] = torch::stack(input_ids).pin_memory().to(at::Device(torch::kCUDA, idx));
        input_data["token_type_ids"] = torch::stack(token_type_ids).pin_memory().to(at::Device(torch::kCUDA, idx));
        input_data["attention_mask"] = torch::stack(attention_mask).pin_memory().to(at::Device(torch::kCUDA, idx));

        auto start_time = chrono::steady_clock::now();
        auto ret = model.forward({}, input_data).toIValue().toTuple()->elements()[0];
        auto stop_time = chrono::steady_clock::now();
        if(batch_idx >= warmup_batch_num)
            model_times.emplace_back(chrono::duration_cast<chrono::milliseconds>(stop_time - start_time).count());


        auto res = torch::softmax(ret.toTensor(),1);
        
        vector<string> answers;
        for(int i=0; i<batch.size(); ++i) {
            float paraphrased_percent = 100.0 * res[i][1].item<float>();
            answers.push_back(to_string((int)round(paraphrased_percent)) + "% paraphrase");
        }
    }

    vector<float> model_times;

    shared_ptr<BatchQueue> queue_;
    vector<thread> threads_;

};


int main(const int argc, const char* const argv[]) {
    shared_ptr<BatchQueue> batch_queue = make_shared<BatchQueue>();

    int batch_num = argc > 1 ? stoi(argv[1]) : 10000;

    warmup_batch_num = max(batch_num / 10, 100);

    bool enable_optimization = argc > 2 && (strcmp(argv[2], "--optimize_for_inference") == 0);

    for(int i=0; i< batch_num + warmup_batch_num; ++i) { 
        vector<KWARG> batch;
        for(int j=0; j<8; ++j) 
        {
            KWARG kwargs;

            kwargs["input_ids"] = torch::tensor(std::vector<int64_t>{
                101,  1109,  1419, 20164, 10932,  2271,  7954,  1110,  1359,  1107,
                1203,  1365,  1392,   102,  7302,  1116,  1132,  2108,  2213,  1111,
                1240,  2332,   102});
            kwargs["token_type_ids"] = torch::tensor(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1});
            kwargs["attention_mask"] = torch::tensor(std::vector<int64_t>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
            batch.push_back(kwargs);
        }

        batch_queue->enqueue(batch);
    }

    cout << "Total number of batches: " << batch_queue->size() << endl;
    
    TorchScriptWorker  worker(batch_queue, enable_optimization);
    
}
