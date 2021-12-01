// Tokenizer
#include "bert_tokenizer.h"

// CPP REST SDK
#include <cpprest/http_listener.h>
#include <cpprest/json.h>

// Google Log
#include <glog/logging.h>

// PyTorch
#include <torch/torch.h>
#include <torch/types.h>
#include <torch/csrc/deploy/deploy.h>

using namespace web::http::experimental::listener;
using namespace web::http;
using namespace web;

// C++ STD
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <thread>
#include <map>
#include <string>
#include <iostream>
#include <set>
#include <mutex>
#include <condition_variable>

using namespace std;

typedef web::json::value JsonValue;

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

void log_metric(string name, float value, string unit="ms") {
   LOG(INFO) << name << "." << unit << ":" << value << "|" << endl;
}

class SequenceClassifier {
   public:
   SequenceClassifier(){

   }
   torch::Tensor classify(string sequence_0, string sequence_1) {
      // acquire();

      // auto handler_time_begin = chrono::steady_clock::now();

      // chrono::steady_clock::time_point begin = chrono::steady_clock::now();
      // auto kwargs = tokenizer_.encode_plus(sequence_0, sequence_1);
      // chrono::steady_clock::time_point end = chrono::steady_clock::now();

      // log_metric("TokenizationTime", chrono::duration_cast<chrono::milliseconds>(end - begin).count());

      

      // if (torch::cuda::is_available()) {
      //    torch::Device device = torch::kCUDA;

      //    kwargs["input_ids"] = kwargs["input_ids"].toTensor().to(device);
      //    kwargs["token_type_ids"] = kwargs["token_type_ids"].toTensor().to(device);
      //    kwargs["attention_mask"] = kwargs["attention_mask"].toTensor().to(device);
      // }

      // begin = chrono::steady_clock::now();

      // auto ret = apply_model(kwargs);

      // auto res = torch::softmax(ret.toTensor(),1);
      // end = chrono::steady_clock::now();

      // log_metric("ModelTime", chrono::duration_cast<chrono::milliseconds>(end - begin).count());


      // auto handler_time_end = chrono::steady_clock::now();

      // log_metric("HandlerTime", chrono::duration_cast<chrono::milliseconds>(handler_time_end - handler_time_begin).count());

      // release();

      // return res;
      return torch::ones(1);
   }

   virtual  c10::IValue apply_model(std::unordered_map<std::string, c10::IValue> kwargs) = 0;

   // protected:

   // void release() {
   //    std::lock_guard<decltype(mtx_)> lock(mtx_);
   //    ++count_;
   //    cv_.notify_one();
   // }

   // void acquire() {
   //    std::unique_lock<decltype(mtx_)> lock(mtx_);
   //    while(!count_) {
   //       cv_.wait(lock);
   //    }
   //    --count_;
   // }


   // std::mutex mtx_;
   // std::condition_variable cv_;
   // BertTokenizer &tokenizer_;
};

class TorchPackageSequenceClassifier: public SequenceClassifier {
   public:
   TorchPackageSequenceClassifier(std::string model_to_serve,std::string python_path)
   : manager_(4, python_path) {
      torch::deploy::Package package = manager_.loadPackage(model_to_serve);
      model_handler_ = package.loadPickle("model", "model.pkl");
   }

   ~TorchPackageSequenceClassifier() {

   }
   protected:

   c10::IValue apply_model(std::unordered_map<std::string, c10::IValue> kwargs){
      auto ret = model_handler_.callKwargs({}, kwargs).toIValue().toTuple()->elements()[0];
      return ret;
   }

   torch::deploy::InterpreterManager manager_;
   torch::deploy::ReplicatedObj model_handler_;
};

class TorchScriptSequenceClassifier: public SequenceClassifier {
   public:
   TorchScriptSequenceClassifier(std::string model_to_serve)
   : model_(torch::jit::load(model_to_serve)) {
      model_.eval();
   }

   ~TorchScriptSequenceClassifier() {

   }

   c10::IValue apply_model(std::unordered_map<std::string, c10::IValue> kwargs){
      auto ret = model_.forward({}, kwargs).toIValue().toTuple()->elements()[0];
      return ret;
   }

   protected:

   torch::jit::script::Module model_;
};


struct Request {

   Request(http_request request, string sequence_0, string sequence_1)
   :request_(request), sequence_0_(sequence_0), sequence_1_(sequence_1) {
      // PredictionTime is not 100 accurate as data fetching from request is not included
      creation_time_ = chrono::steady_clock::now();
   }

   void reply(int code, string message) {
      request_.reply(code, message);
      log_metric("PredictionTime", chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - creation_time_).count());
   }

   void log_queue_time() {
      log_metric("QueueTime", chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - creation_time_).count());
   }

   http_request request_;
   string sequence_0_;
   string sequence_1_;
   std::chrono::time_point<std::chrono::steady_clock> creation_time_;
};


class Batcher {
public:
   Batcher(unique_ptr<SequenceClassifier> classifier, BertTokenizer tokenizer, size_t max_batch_size, size_t sequence_length) 
   : classifier_(std::move(classifier)), tokenizer_(std::move(tokenizer)), max_batch_size_(max_batch_size), sequence_length_(sequence_length){

   }

   void enqueue(Request request) {
      cout << "Queuing request...\n";
      lock_guard<mutex> lock(m_);
      queue_.push(request);
   }

   size_t get_queue_size() {
      lock_guard<mutex> lock(m_);
      return queue_.size();
   }

   void process_batch() {
      auto begin = chrono::steady_clock::now();

      vector<Request> requests;
      size_t sample_num;

      {
         lock_guard<mutex> lock(m_);
         if (queue_.size() == 0)
            return;

         sample_num = min(max_batch_size_, queue_.size());
         
         cout << "Processing a batch of size " << to_string(sample_num) << "\n";

         for(int i=0; i<sample_num; ++i) {
            requests.push_back(std::move(queue_.front()));
            queue_.pop();
            requests.back().log_queue_time();
         }
      }

      vector<torch::Tensor> input_ids, token_type_ids, attention_mask;

      for(int i=0; i<sample_num; ++i) {

         string sequence_0 = requests[i].sequence_0_;
         string sequence_1 = requests[i].sequence_1_;

         auto kwargs = tokenizer_.encode_plus(sequence_0, sequence_1, sequence_length_);
         input_ids.push_back(kwargs["input_ids"].toTensor());
         token_type_ids.push_back(kwargs["token_type_ids"].toTensor());
         attention_mask.push_back(kwargs["attention_mask"].toTensor());
      }

      unordered_map<string, c10::IValue> batch;
      
      batch["input_ids"] = torch::stack(input_ids);
      batch["token_type_ids"] = torch::stack(token_type_ids);
      batch["attention_mask"] = torch::stack(attention_mask);
      // chrono::steady_clock::time_point end = chrono::steady_clock::now();

      // log_metric("TokenizationTime", chrono::duration_cast<chrono::milliseconds>(end - begin).count());

      if (torch::cuda::is_available()) {
         torch::Device device = torch::kCUDA;
         batch["input_ids"] = batch["input_ids"].toTensor().to(device);
         batch["token_type_ids"] = batch["token_type_ids"].toTensor().to(device);
         batch["attention_mask"] = batch["attention_mask"].toTensor().to(device);
      }

      auto ret = classifier_->apply_model(batch);

      log_metric("ModelTime", chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count());

      auto res = torch::softmax(ret.toTensor(),1);
      
      for(int i=0; i<sample_num; ++i) {
         float paraphrased_percent = 100.0 * res[i][1].item<float>();
         string answer = to_string((int)round(paraphrased_percent)) + "% paraphrase";

         requests[i].reply(status_codes::OK, answer);

         log_metric("HandlerTime", chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count());
      }
   }

protected:
   queue<Request> queue_;
   unique_ptr<SequenceClassifier> classifier_;
   BertTokenizer tokenizer_;

   size_t max_batch_size_;
   size_t sequence_length_;

   mutex m_;
};


void handle_request(
   http_request request,
   Batcher &batcher) {

   status_code code = status_codes::OK;
   utility::string_t answer;

   string sequence_0, sequence_1;

   request
      .extract_json()
      .then([&](pplx::task<JsonValue> task) {
         try
         {
            // auto begin = chrono::steady_clock::now();
            JsonValue json = task.get();

            if(! (json.has_field("sequence_0") && json.has_field("sequence_1")) ) {
               code = status_codes::InternalError;
               answer = "JSON object does not contain necessary fields!";
               return;
            }

            sequence_0 = json["sequence_0"].as_string();
            sequence_1 = json["sequence_1"].as_string();

            // torch::Tensor ret = classifier->classify(json["sequence_0"].as_string(), json["sequence_1"].as_string());

            // float paraphrased_percent = 100.0 * ret[0][1].item<float>();
            // answer = to_string((int)round(paraphrased_percent)) + "% paraphrase";

            auto end = chrono::steady_clock::now();

            // log_metric("PredictionTime", chrono::duration_cast<chrono::milliseconds>(end - begin).count());
            // log_metric("QueueTime", 0);
            log_metric("WorkerThreadTime", 0);
         }
         catch (http_exception const & e)
         {
            wcout << e.what() << endl;
         }
      })
      .wait();

   if(code !=  status_codes::OK){
      request.reply(code, answer);
   }else {

      Request r(std::move(request), sequence_0, sequence_1);
      batcher.enqueue(std::move(r));
   }

   // request.reply(code, answer);
}

int main(const int argc, const char* const argv[]) {
   if (!(argc == 6 or argc == 5)) {
      std::cout << "Usage: cpp_backend_poc_eager <uri> <batch_size> <batch_delay> <model_to_serve> [<python_path>] " << std::endl
               << "Serve <model_to_serve> at <uri> with threads." << std::endl
               << std::endl
               << "Example: cpp_backend_poc_eager http://localhost:8090 8 100 ../models/resnet venv/lib/python3.8/site-packages/" << std::endl;
      return EXIT_FAILURE;
   }

   if (std::signal(SIGINT, &signal_handler) == SIG_ERR) {
      std::cerr << "Failed to register signal handler!" << std::endl;
      return EXIT_FAILURE;
   }

   google::InitGoogleLogging("log.txt");

   // Configurations
   const std::string uri = argv[1];
   const size_t max_batch_size = atol(argv[2]);
   const long max_batch_delay(atol(argv[3]));
   const std::string model_to_serve = argv[4];
   const size_t sequence_length = 128;

   BertTokenizer tokenizer("vocab.txt");

   unique_ptr<SequenceClassifier> classifier;

   // const size_t thread_count = std::stoul(argv[3]);

   if(argc > 5) {
      const std::string python_path = argv[5];
      classifier = unique_ptr<SequenceClassifier>(new TorchPackageSequenceClassifier(model_to_serve, python_path));
   }else
   {
      classifier = unique_ptr<SequenceClassifier>(new TorchScriptSequenceClassifier(model_to_serve));
   }

   Batcher batcher(std::move(classifier), std::move(tokenizer), max_batch_size, sequence_length);

   at::set_num_interop_threads(1);
   at::set_num_threads(1);

   // cout << "Serving " << model_to_serve << " at " << uri << " with " << thread_count << " threads." << std::endl;
   cout << "Inter-op threads:" << at::get_num_interop_threads() << endl;
   cout << "Intra-op threads:" << at::get_num_threads() << endl; 

   cout << "Press Ctrl+C to terminate." << std::endl;

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

   listener.support(methods::POST, [&batcher](const http_request& request) {
      std::cout << "Received a POST request!" << std::endl;
      handle_request(
         request,
         // handler,
         // tokenizer
         batcher
      );

   });

   listener.support(methods::PUT, [&batcher](const http_request& request) {
      std::cout << "Received a PUT request!" << std::endl;
      handle_request(
         request,
         // handler,
         // tokenizer
         batcher
      );
   });

  listener.open().wait();

  auto last_batch_time = chrono::steady_clock::now();
  while (signal_ != SIGINT) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    const bool max_time_passed = std::chrono::duration_cast<std::chrono::milliseconds>(chrono::steady_clock::now() - last_batch_time).count() >= max_batch_delay;
    if(max_time_passed || batcher.get_queue_size() >= max_batch_size) {
       batcher.process_batch();
       last_batch_time = chrono::steady_clock::now();
    }
  }

  std::cout << "Shutting down ..." << std::endl;
  listener.close();

  return EXIT_SUCCESS;
}
