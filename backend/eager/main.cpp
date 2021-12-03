// Tokenizer
#include "bert_tokenizer.h"

// CPP REST SDK
#include <cpprest/http_listener.h>
#include <cpprest/json.h>
#include <pplx/threadpool.h>        // crossplat::threadpool

// Google Log
#include <glog/logging.h>

// PyTorch
#include <torch/torch.h>
#include <torch/types.h>
#include <torch/csrc/deploy/deploy.h>
#include <c10/util/Optional.h>
#include <c10/core/Device.h>

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

typedef unordered_map<string, c10::IValue> KWARG;

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
}

void log_metric(string name, float value, string unit="ms") {
   LOG(INFO) << name << "." << unit << ":" << value << "|" << endl;
}


void move_to_cuda(KWARG &kwargs, c10::DeviceIndex idx) {
    if (torch::cuda::is_available()) {
         torch::Device device(torch::kCUDA, idx);

         for(auto &p : kwargs)
            p.second = p.second.toTensor().to(device);
    }
}

class SequenceClassifier {
   public:
   SequenceClassifier(){

   }

   virtual  c10::IValue apply_model(std::unordered_map<std::string, c10::IValue> kwargs) = 0;
};

class TorchPackageSequenceClassifier: public SequenceClassifier {
   public:
   TorchPackageSequenceClassifier(torch::deploy::ReplicatedObj model_handler, c10::optional<c10::DeviceIndex> device_idx)
   : model_handler_(std::move(model_handler)), device_idx_(device_idx) {
      if(device_idx_.has_value())
         model_handler_.acquireSession().self.attr("to")({"cuda:" + to_string(*device_idx_)});
   }

   ~TorchPackageSequenceClassifier() {

   }
   protected:

   c10::IValue apply_model(std::unordered_map<std::string, c10::IValue> kwargs){
      if(device_idx_.has_value())
         move_to_cuda(kwargs, *device_idx_);
      auto ret = model_handler_.callKwargs({}, kwargs).toIValue().toTuple()->elements()[0];
      return ret;
   }

   c10::optional<c10::DeviceIndex> device_idx_;
   torch::deploy::ReplicatedObj model_handler_;
};

class TorchScriptSequenceClassifier: public SequenceClassifier {
   public:
   TorchScriptSequenceClassifier(torch::jit::script::Module model, c10::optional<c10::DeviceIndex> device_idx)
   : model_(std::move(model)), device_idx_(device_idx) {
      model_.eval();
      if(device_idx_.has_value() && torch::cuda::is_available())
         model_.to(at::Device(torch::kCUDA, *device_idx_));
   }

   ~TorchScriptSequenceClassifier() {

   }

   c10::IValue apply_model(std::unordered_map<std::string, c10::IValue> kwargs){
      if(device_idx_.has_value())
         move_to_cuda(kwargs, *device_idx_);
      auto ret = model_.forward({}, kwargs).toIValue().toTuple()->elements()[0];
      return ret;
   }

   protected:

   c10::optional<c10::DeviceIndex> device_idx_;
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
   }

   void log_queue_time() {
      log_metric("QueueTime", chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - creation_time_).count());
   }

   http_request request_;
   string sequence_0_;
   string sequence_1_;
   std::chrono::time_point<std::chrono::steady_clock> creation_time_;
};

struct BatchQueue {
   BatchQueue() {
   }

   void enqueue(vector<Request> batch) {
      {
         lock_guard<mutex> lock(m_);
         batches_.push(std::move(batch));
      }
      cv_.notify_one();
   }

   vector<Request> dequeue() {
      lock_guard<mutex> lock(m_);
      if(batches_.size() == 0)
         return vector<Request>();
         
      vector<Request> batch = std::move(batches_.front());
      batches_.pop();
      return batch;
   }
   
   condition_variable cv_;
   mutex m_;
   queue<vector<Request>> batches_;
};


class Batcher {
public:
   Batcher(size_t max_batch_size, BatchQueue &batch_queue) 
   :max_batch_size_(max_batch_size), batch_queue_(batch_queue){
      last_batch_time = chrono::steady_clock::now();
   }

   void enqueue(Request request) {
      cout << "Queuing request...\n";
      bool should_push = false;
      {
         lock_guard<mutex> lock(m_);
         queue_.push(request);
         if(queue_.size() >= max_batch_size_) {
            should_push = true;
         }
      }
      if(should_push)
         push_batch();
   }

   void push_batch() {
      vector<Request> requests;

      {
         lock_guard<mutex> lock(m_);
         size_t sample_num;

         if (queue_.size() == 0)
            return ;

         sample_num = min(max_batch_size_, queue_.size());
         
         for(int i=0; i<sample_num; ++i) {
            requests.push_back(std::move(queue_.front()));
            queue_.pop();
            requests.back().log_queue_time();
         }
         last_batch_time = chrono::steady_clock::now();
      }

      batch_queue_.enqueue(std::move(requests));
   }

   chrono::time_point<chrono::steady_clock> get_last_batch_time() {
      lock_guard<mutex> lock(m_);
      return last_batch_time;
   }

protected:
   queue<Request> queue_;
   size_t max_batch_size_;

   chrono::time_point<chrono::steady_clock> last_batch_time;

   mutex m_;
   
   BatchQueue &batch_queue_;
};

struct WorkerThead {

   WorkerThead(unique_ptr<SequenceClassifier> classifier, BertTokenizer& tokenizer, size_t sequence_length, BatchQueue& batch_queue, atomic_bool &terminate)
   :classifier_(std::move(classifier)), tokenizer_(tokenizer), sequence_length_(sequence_length), batch_queue_(batch_queue), terminate_(terminate) {

   }

   void do_work() {

      while(!terminate_) {
         unique_lock<mutex> lock(batch_queue_.m_);
         batch_queue_.cv_.wait(lock, [&]{return this->terminate_ || this->batch_queue_.batches_.size();});

         if(terminate_)
            return;

         vector<Request> requests = std::move(batch_queue_.batches_.front());
         batch_queue_.batches_.pop();
         lock.unlock();

         auto begin = chrono::steady_clock::now();
         process_batch(requests);
         log_metric("PredictionTime", chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count());
      }

   }


   void process_batch(vector<Request> requests) {
      auto begin = chrono::steady_clock::now();

      size_t sample_num = requests.size();

      if (sample_num == 0)
         return;

      cout << "Processing batch of size " << to_string(sample_num) << "\n";

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

      auto model_begin = chrono::steady_clock::now();
      auto ret = classifier_->apply_model(batch);

      log_metric("ModelTime", chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - model_begin).count());

      auto res = torch::softmax(ret.toTensor(),1);

      log_metric("HandlerTime", chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count());
      
      for(int i=0; i<sample_num; ++i) {
         float paraphrased_percent = 100.0 * res[i][1].item<float>();
         string answer = to_string((int)round(paraphrased_percent)) + "% paraphrase";

         requests[i].reply(status_codes::OK, answer);
      }
   }

   unique_ptr<SequenceClassifier> classifier_;
   BertTokenizer& tokenizer_;

   size_t sequence_length_;
   BatchQueue &batch_queue_;
   atomic_bool &terminate_;
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

            auto end = chrono::steady_clock::now();

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
   crossplat::threadpool::initialize_with_threads(100);

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

   atomic_bool terminate;
   terminate = false;

   BatchQueue batch_queue;

   Batcher batcher(max_batch_size, batch_queue);

   vector<thread> worker_threads;

   c10::optional<torch::deploy::InterpreterManager> manager;
   c10::optional<torch::deploy::Package> package;

   for(int i=0; i<4; ++i) {
      unique_ptr<SequenceClassifier> classifier;
      c10::optional<c10::DeviceIndex> device_idx(torch::cuda::is_available() ? c10::optional<c10::DeviceIndex>(i) : c10::nullopt);
      if(argc > 5) {
         if(!manager.has_value()) {
            const std::string python_path = argv[5];
            manager.emplace(4, python_path);
            package.emplace(manager->loadPackage(model_to_serve));
         }
         
         classifier = unique_ptr<SequenceClassifier>(new TorchPackageSequenceClassifier(std::move(package->loadPickle("model", "model.pkl")), device_idx));
      }else
      {
         classifier = unique_ptr<SequenceClassifier>(new TorchScriptSequenceClassifier(std::move(torch::jit::load(model_to_serve)), device_idx));
      }

      worker_threads.emplace_back(&WorkerThead::do_work, WorkerThead(std::move(classifier), tokenizer, sequence_length, batch_queue, terminate));
   }

   thread timer_thread = thread([&batcher, &terminate, &max_batch_delay](){
       while(!terminate) {
         bool should_push = false;
         chrono::milliseconds sleep_time(max_batch_delay);
         chrono::time_point<chrono::steady_clock> last_batch_time = batcher.get_last_batch_time();
         chrono::time_point<chrono::steady_clock> now =chrono::steady_clock::now();
         chrono::milliseconds diff = chrono::duration_cast<chrono::milliseconds>(now - last_batch_time);

         if(diff.count() > max_batch_delay) {
            should_push = true;
         }else {
            sleep_time = diff;
         }
         if(should_push)
            batcher.push_batch();
         this_thread::sleep_for(sleep_time);
         }
   });

   at::set_num_interop_threads(1);
   at::set_num_threads(1);

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

  while (signal_ != SIGINT) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  terminate = true;

  batch_queue.cv_.notify_all();

  for(auto &t : worker_threads)
      t.join();

   timer_thread.join();

  std::cout << "Shutting down ..." << std::endl;
  listener.close();

  return EXIT_SUCCESS;
}
