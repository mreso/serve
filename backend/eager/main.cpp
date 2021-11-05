// Tokenizer
#include "bert_tokenizer.h"

// CPP REST SDK
#include <cpprest/http_listener.h>
#include <cpprest/json.h>

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

class ISequenceClassifier {
   public:
   virtual torch::Tensor classify(string sequence_0, string sequence_1) = 0;

   protected:

   void release() {
      std::lock_guard<decltype(mtx_)> lock(mtx_);
      ++count_;
      cv_.notify_one();
   }

   void acquire() {
      std::unique_lock<decltype(mtx_)> lock(mtx_);
      while(!count_) {
         cv_.wait(lock);
      }
      --count_;
   }

   std::mutex mtx_;
   std::condition_variable cv_;
   size_t count_;
};

class TorchPackageSequenceClassifier: public ISequenceClassifier {
   public:
   TorchPackageSequenceClassifier(std::string model_to_serve, size_t thread_count, std::string python_path, BertTokenizer &tokenizer)
   : manager_(thread_count, python_path),tokenizer_(tokenizer) {
      count_ = thread_count;
      torch::deploy::Package package = manager_.loadPackage(model_to_serve);
      model_handler_ = package.loadPickle("model", "model.pkl");
   }

   ~TorchPackageSequenceClassifier() {

   }

   torch::Tensor classify(string sequence_0, string sequence_1) {
      acquire();
      chrono::steady_clock::time_point begin = chrono::steady_clock::now();
      auto kwargs = tokenizer_.encode_plus(sequence_0, sequence_1);
      chrono::steady_clock::time_point end = chrono::steady_clock::now();

      cout << "TokenizationTime: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;

      begin = chrono::steady_clock::now();
      auto ret = model_handler_.callKwargs({}, kwargs).toIValue().toTuple()->elements()[0];

      auto res = torch::softmax(ret.toTensor(),1);
      end = chrono::steady_clock::now();

      cout << "ModelTime: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;

      release();

      return res;
   }

   protected:
   torch::deploy::InterpreterManager manager_;
   torch::deploy::ReplicatedObj model_handler_;
   BertTokenizer &tokenizer_;
};

class TorchScriptSequenceClassifier: public ISequenceClassifier {
   public:
   TorchScriptSequenceClassifier(std::string model_to_serve, BertTokenizer &tokenizer, size_t thread_count)
   : model_(torch::jit::load(model_to_serve)),tokenizer_(tokenizer) {
      count_ = thread_count;
      model_.eval();
   }

   ~TorchScriptSequenceClassifier() {

   }

   torch::Tensor classify(string sequence_0, string sequence_1) {
      acquire();

      chrono::steady_clock::time_point begin = chrono::steady_clock::now();
      auto kwargs = tokenizer_.encode_plus(sequence_0, sequence_1);
      chrono::steady_clock::time_point end = chrono::steady_clock::now();

      cout << "TokenizationTime: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;

      begin = chrono::steady_clock::now();
      auto ret = model_.forward({}, kwargs);
      end = chrono::steady_clock::now();

      cout << "ModelTime: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;

      auto ret2 = ret.toIValue().toTuple()->elements()[0];

      auto res = torch::softmax(ret2.toTensor(),1);


      release();

      return res;
   }

   protected:

   torch::jit::script::Module model_;
   BertTokenizer &tokenizer_;

};

void handle_request(
   http_request request,
   unique_ptr<ISequenceClassifier> &classifier) {

   status_code code = status_codes::OK;
   utility::string_t answer;
   request
      .extract_json()
      .then([&](pplx::task<JsonValue> task) {
         try
         {
            JsonValue json = task.get();

            if(! (json.has_field("sequence_0") && json.has_field("sequence_1")) ) {
               code = status_codes::InternalError;
               answer = "JSON object does not contain necessary fields!";
               return;
            }

            torch::Tensor ret = classifier->classify(json["sequence_0"].as_string(), json["sequence_1"].as_string());

            float paraphrased_percent = 100.0 * ret[0][1].item<float>();
            answer = to_string((int)round(paraphrased_percent)) + "% paraphrase";

         }
         catch (http_exception const & e)
         {
            wcout << e.what() << endl;
         }
      })
      .wait();

   request.reply(code, answer);
}

int main(const int argc, const char* const argv[]) {
   if (!(argc == 5 or argc == 4)) {
      std::cout << "Usage: cpp_backend_poc_eager <uri> <model_to_serve> <thread_count> [<python_path>] " << std::endl
               << "Serve <model_to_serve> at <uri> with <thread_count> threads." << std::endl
               << std::endl
               << "Example: cpp_backend_poc_eager http://localhost:8090 4 ../models/resnet venv/lib/python3.8/site-packages/" << std::endl;
      return EXIT_FAILURE;
   }

   if (std::signal(SIGINT, &signal_handler) == SIG_ERR) {
      std::cerr << "Failed to register signal handler!" << std::endl;
      return EXIT_FAILURE;
   }

   // Configurations
   const std::string uri = argv[1];
   const std::string model_to_serve = argv[2];

   BertTokenizer tokenizer("vocab.txt");

   unique_ptr<ISequenceClassifier> classifier;


   const size_t thread_count = std::stoul(argv[3]);

   if(argc > 4) {
      const std::string python_path = argv[4];
      classifier = unique_ptr<ISequenceClassifier>(new TorchPackageSequenceClassifier(model_to_serve, thread_count, python_path, tokenizer));
   }else
   {
      classifier = unique_ptr<ISequenceClassifier>(new TorchScriptSequenceClassifier(model_to_serve, tokenizer, thread_count));
   }

   string sequence_0 = "Eating apples is a health risk";
   string sequence_1 = "Apples are especially bad for your health";

   classifier->classify(sequence_0, sequence_1);
   classifier->classify(sequence_0, sequence_1);


   return EXIT_SUCCESS;
   std::cout << "Serving " << model_to_serve << " at " << uri << " with " << thread_count << " threads." << std::endl;
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

   listener.support(methods::POST, [&classifier](const http_request& request) {
      std::cout << "Received a POST request!" << std::endl;
      handle_request(
         request,
         // handler,
         // tokenizer
         classifier
      );

   });

   listener.support(methods::PUT, [&classifier](const http_request& request) {
      std::cout << "Received a PUT request!" << std::endl;
      handle_request(
         request,
         // handler,
         // tokenizer
         classifier
      );
   });

  listener.open().wait();

  while (signal_ != SIGINT) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }

  std::cout << "Shutting down ..." << std::endl;
  listener.close();

  return EXIT_SUCCESS;
}
