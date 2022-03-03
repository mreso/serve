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
#include <torch/csrc/deploy/path_environment.h>
#include <c10/util/Optional.h>
#include <c10/core/Device.h>
#include "c10/cuda/CUDAGuard.h"

#include <cuda_runtime.h>

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

#include "blockingconcurrentqueue.h"


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
   SequenceClassifier(c10::optional<c10::DeviceIndex> device_idx)
   :device_idx_(device_idx){

   }

   c10::optional<c10::DeviceIndex> get_device_idx() {
      return device_idx_;
   }

   virtual  c10::IValue apply_model(std::unordered_map<std::string, c10::IValue> kwargs) = 0;
   c10::optional<c10::DeviceIndex> device_idx_;
};

class TorchPackageSequenceClassifier: public SequenceClassifier {
   public:
   TorchPackageSequenceClassifier(
      shared_ptr<torch::deploy::ReplicatedObj> model_handler,
      shared_ptr<torch::deploy::InterpreterManager> manager,
      c10::optional<c10::DeviceIndex> device_idx)
   :model_handler_(model_handler),
   manager_(manager),
   SequenceClassifier(device_idx) {
      if(device_idx_.has_value()) {
         auto device = "cuda:" + to_string(*device_idx_);
         
         auto I = model_handler_->acquireSession(&manager_->allInstances().at(*device_idx_));
         I.self.attr("to")({device});
      }
      if(device_idx_.has_value()) {
         model_handler_->acquireSession(&manager_->allInstances().at(*device_idx_)).self.attr("eval")(vector<c10::IValue>());
      }else {
         model_handler_->acquireSession().self.attr("eval")(vector<c10::IValue>());
      }
   }

   ~TorchPackageSequenceClassifier() {

   }
   protected:

   c10::IValue apply_model(std::unordered_map<std::string, c10::IValue> kwargs){
      shared_ptr<torch::deploy::InterpreterSession> I;
      if(device_idx_.has_value())
         I = make_shared<torch::deploy::InterpreterSession>(model_handler_->acquireSession(&manager_->allInstances().at(*device_idx_)));
      else
         I = make_shared<torch::deploy::InterpreterSession>(model_handler_->acquireSession());

      return I->self.callKwargs({}, kwargs).toIValue().toTuple()->elements()[0];
   }

   shared_ptr<torch::deploy::ReplicatedObj> model_handler_;
   shared_ptr<torch::deploy::InterpreterManager> manager_;

};

class TorchScriptSequenceClassifier: public SequenceClassifier {
   public:
   TorchScriptSequenceClassifier(torch::jit::script::Module model, c10::optional<c10::DeviceIndex> device_idx)
   : model_(std::move(model)), SequenceClassifier(device_idx) {
      if(device_idx_.has_value() && torch::cuda::is_available())
      {
         model_.to(at::Device(torch::kCUDA, *device_idx_)); 
         std::cout << "Optimizing for inference...\n";
         model_ = torch::jit::optimize_for_inference(model_);
      }
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

   Request() {

   }


   void reply(int code, string message) {
      request_.reply(code, message);
   }
   void reply(int code) {
      request_.reply(code, reply_message_);
   }
   void set_reply_message(string reply_message){
      reply_message_ = reply_message;
   }
   void mark_batching(){
      batch_time_ = chrono::steady_clock::now();
   }
   void mark_reply(){
      reply_time_ = chrono::steady_clock::now();
   }
   void log_times(){
      long queue_duration = chrono::duration_cast<chrono::milliseconds>(batch_time_ - creation_time_).count();
      long batch_queue_duration = chrono::duration_cast<chrono::milliseconds>(processing_time_ - batch_time_).count();
      long processing_duration = chrono::duration_cast<chrono::milliseconds>(reply_time_ - processing_time_).count();
      long reply_duration = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - reply_time_).count();

      LOG(INFO) << "AllTimes," << queue_duration << "," << batch_queue_duration << "," << processing_duration << "," << reply_duration << "," << queue_duration + batch_queue_duration << endl;
   }

   void log_queue_time() {
      processing_time_ = chrono::steady_clock::now();
      log_metric("QueueTime", chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - creation_time_).count());
   }

   http_request request_;
   string sequence_0_;
   string sequence_1_;
   std::chrono::time_point<std::chrono::steady_clock> creation_time_;
   std::chrono::time_point<std::chrono::steady_clock> batch_time_;
   std::chrono::time_point<std::chrono::steady_clock> processing_time_;
   std::chrono::time_point<std::chrono::steady_clock> reply_time_;
   string reply_message_;
};

class Batcher {
public:
   Batcher(size_t max_batch_size, size_t max_batch_delay) 
   :max_batch_size_(max_batch_size), max_batch_delay_(max_batch_delay){
   }

   void enqueue(Request request) {
      cc_request_queue_.enqueue(request);
   }

   vector<Request> poll_batch() {
      chrono::time_point<chrono::steady_clock>last_batch_time = chrono::steady_clock::now();
      vector<Request> requests;

      for(size_t i=0; i<max_batch_size_; ++i)
      {  
         Request r;

         chrono::milliseconds time_since_last_batch = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now()-last_batch_time);

         chrono::milliseconds max_deplay = max(chrono::milliseconds(0), chrono::milliseconds(max_batch_delay_) - time_since_last_batch);

         if(cc_request_queue_.wait_dequeue_timed(r, max_deplay)) {
         
            requests.push_back(std::move(r));
            requests.back().mark_batching();
         }else {
            break;
         }
      }

      return requests;
   }

protected:
   queue<Request> queue_;
   moodycamel::BlockingConcurrentQueue<Request> cc_request_queue_;
   size_t max_batch_size_;
   size_t max_batch_delay_;
};

class RequestReplyQueue {
public:
   RequestReplyQueue() {
      terminate_ = false;
      for(int i=0; i<4; i++)
         threads_.emplace_back([this]{do_work();});

   }
   ~RequestReplyQueue() {
      terminate_ = true;

      for(auto &t : threads_)
         t.join();
   }

   void enqueue_batch(vector<Request> requests) {
      cc_reply_queue_.enqueue(requests);
   }

   void do_work() {

      while(!terminate_) {
         vector<Request> requests;

         if(cc_reply_queue_.wait_dequeue_timed(requests, 200)) {
            for(auto &r : requests){
               r.reply(status_codes::OK);
               r.log_times();
            }
         }
      }
   }

protected:
   moodycamel::BlockingConcurrentQueue<vector<Request>> cc_reply_queue_;

   atomic_bool terminate_;
   
   mutex m_;
   condition_variable cv_;
   vector<thread> threads_;

};

struct WorkerThead {

   WorkerThead(unique_ptr<SequenceClassifier> classifier, BertTokenizer& tokenizer, size_t sequence_length, Batcher& batcher, atomic_bool &terminate,
   RequestReplyQueue& reply_queue)
   :classifier_(std::move(classifier)), tokenizer_(tokenizer), sequence_length_(sequence_length), batcher_(batcher), terminate_(terminate),
   reply_queue_(reply_queue) {

   }

   void do_work() {

      auto device_idx = classifier_->get_device_idx();

      
      while(!terminate_) {

         vector<Request> requests = batcher_.poll_batch();
         if(requests.size() == 0)
            continue;
         
         auto w_begin = chrono::steady_clock::now();

         for(auto &r : requests)
            r.log_queue_time();

         auto begin = chrono::steady_clock::now();
         process_batch(move(requests));
         auto duration = chrono::steady_clock::now() - begin;
         log_metric("PredictionTime", chrono::duration_cast<chrono::milliseconds>(duration).count());
         log_metric("WorkerThreadTime", chrono::duration_cast<chrono::milliseconds>((chrono::steady_clock::now() - w_begin) - duration).count());
      }

   }


   void process_batch(vector<Request> requests) {
      auto begin = chrono::steady_clock::now();

      size_t sample_num = requests.size();

      if (sample_num == 0)
         return;

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
      auto device_idx = classifier_->get_device_idx();

      if(device_idx.has_value()) {
         batch["input_ids"] = torch::stack(input_ids).pin_memory();
         batch["token_type_ids"] = torch::stack(token_type_ids).pin_memory();
         batch["attention_mask"] = torch::stack(attention_mask).pin_memory();
         move_to_cuda(batch, *device_idx);
      }else {
         batch["input_ids"] = torch::stack(input_ids);
         batch["token_type_ids"] = torch::stack(token_type_ids);
         batch["attention_mask"] = torch::stack(attention_mask);
      }
      
      auto ret = classifier_->apply_model(batch);
      auto res = torch::softmax(ret.toTensor(),1);


      for(int i=0; i<sample_num; ++i) {
         float paraphrased_percent = 100.0 * res[i][1].item<float>();
         requests[i].set_reply_message(to_string((int)round(paraphrased_percent)) + "% paraphrase");
      }
      log_metric("HandlerTime", chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count());

      for(auto &r : requests)
            r.mark_reply();
      
      reply_queue_.enqueue_batch(move(requests));
   }

   unique_ptr<SequenceClassifier> classifier_;
   BertTokenizer& tokenizer_;

   size_t sequence_length_;
   
   Batcher &batcher_;
   atomic_bool &terminate_;
   RequestReplyQueue& reply_queue_;
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
}

int main(const int argc, const char* const argv[]) {
   crossplat::threadpool::initialize_with_threads(160);

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
   const size_t sequence_length = 64;

   BertTokenizer tokenizer("vocab.txt");

   atomic_bool terminate;
   terminate = false;

   Batcher batcher(max_batch_size, max_batch_delay);

   RequestReplyQueue reply_queue;

   vector<thread> worker_threads;

   shared_ptr<torch::deploy::InterpreterManager> manager;
   shared_ptr<torch::deploy::Package> package;
   shared_ptr<torch::deploy::Environment> env;
   shared_ptr<torch::deploy::ReplicatedObj> model;

   for(int i=0; i<4; ++i) {
      unique_ptr<SequenceClassifier> classifier;
      c10::optional<c10::DeviceIndex> device_idx(torch::cuda::is_available() ? c10::optional<c10::DeviceIndex>(i) : c10::nullopt);
      if(argc > 5) {
         if(manager == nullptr) {
            const string python_path = argv[5];
            env = make_shared<torch::deploy::PathEnvironment>(python_path);
            manager = make_shared<torch::deploy::InterpreterManager>(4, env);
            package = make_shared<torch::deploy::Package>(manager->loadPackage(model_to_serve));
            model = make_shared<torch::deploy::ReplicatedObj>(package->loadPickle("model", "model.pkl"));
         }
         classifier = unique_ptr<SequenceClassifier>(new TorchPackageSequenceClassifier(model, manager, device_idx));
      }else
      {
         classifier = unique_ptr<SequenceClassifier>(new TorchScriptSequenceClassifier(std::move(torch::jit::load(model_to_serve)), device_idx));
      }

      worker_threads.emplace_back(&WorkerThead::do_work, WorkerThead(std::move(classifier), tokenizer, sequence_length, batcher, terminate, reply_queue));
   }

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
      // std::cout << "Received a POST request!" << std::endl;
      handle_request(
         request,
         // handler,
         // tokenizer
         batcher
      );

   });

   listener.support(methods::PUT, [&batcher](const http_request& request) {
      // std::cout << "Received a PUT request!" << std::endl;
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

  for(auto &t : worker_threads)
      t.join();

  std::cout << "Shutting down ..." << std::endl;
  listener.close();

  return EXIT_SUCCESS;
}
