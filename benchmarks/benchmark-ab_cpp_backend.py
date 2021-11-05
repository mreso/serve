from abc import ABC, abstractmethod
import csv
import json
import re
import shutil
import signal
import time
from subprocess import Popen, PIPE

import click
import click_config_file
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import tempfile
import os
from urllib.parse import urlparse


default_ab_params = {'url': "https://torchserve.pytorch.org/mar_files/resnet-18.mar",
                     'gpus': '',
                     'exec_env': 'local',
                     'batch_size': 1,
                     'batch_delay': 200,
                     'workers': 1,
                     'concurrency': 10,
                     'requests': 100,
                     'input': '../examples/image_classifier/kitten.jpg',
                     'content_type': 'application/jpg',
                     'image': '',
                     'docker_runtime': '',
                     'backend_profiling': False,
                     'config_properties': 'config.properties',
                     'inference_model_url': 'predictions/benchmark',
                     'backend_parameters': '',
                     }
TMP_DIR = tempfile.gettempdir()
execution_params = default_ab_params.copy()
result_file = os.path.join(TMP_DIR, "benchmark/result.txt")
metric_log = os.path.join(TMP_DIR, "benchmark/logs/model_metrics.log")


def json_provider(file_path, cmd_name):
    with open(file_path) as config_data:
        return json.load(config_data)


@click.command()
@click.argument('test_plan', default='custom')
@click.option('--url', '-u', default='https://torchserve.pytorch.org/mar_files/resnet-18.mar',
              help='Input model url')
@click.option('--exec_env', '-e', type=click.Choice(['local', 'docker'], case_sensitive=False), default='local',
              help='Execution environment')
@click.option('--gpus', '-g', default='',
              help='Number of gpus to run docker container with.  Leave empty to run CPU based docker container')
@click.option('--concurrency', '-c', default=10, help='Number of concurrent requests to run')
@click.option('--requests', '-r', default=100, help='Number of requests')
@click.option('--batch_size', '-bs', default=1, help='Batch size of model')
@click.option('--batch_delay', '-bd', default=200, help='Batch delay of model')
@click.option('--input', '-i', default='../examples/image_classifier/kitten.jpg',
              type=click.Path(exists=True), help='The input file path for model')
@click.option('--content_type', '-ic', default='application/jpg', help='Input file content type')
@click.option('--workers', '-w', default=1, help='Number model workers')
@click.option('--image', '-di', default='', help='Use custom docker image for benchmark')
@click.option('--docker_runtime', '-dr', default='', help='Specify required docker runtime')
@click.option('--backend_profiling', '-bp', default=False,
              help='Enable backend profiling using CProfile. Default False')
@click.option('--config_properties', '-cp', default='config.properties',
              help='config.properties path, Default config.properties')
@click.option('--inference_model_url', '-imu', default='predictions/benchmark',
              help='Inference function url - can be either for predictions or explanations. Default predictions/benchmark')
@click.option('--backend_parameters', '-bp', default='',
              help='Additional parameters for backend')

@click_config_file.configuration_option(provider=json_provider, implicit=False,
                                        help="Read configuration from a JSON file")

def benchmark(test_plan, url, gpus, exec_env, concurrency, requests, batch_size, batch_delay, input, workers,
              content_type, image, docker_runtime, backend_profiling, config_properties, inference_model_url, backend_parameters):
    input_params = {'url': url,
                    'gpus': gpus,
                    'exec_env': exec_env,
                    'batch_size': batch_size,
                    'batch_delay': batch_delay,
                    'workers': workers,
                    'concurrency': concurrency,
                    'requests': requests,
                    'input': input,
                    'content_type': content_type,
                    'image': image,
                    'docker_runtime': docker_runtime,
                    'backend_profiling': backend_profiling,
                    'config_properties': config_properties,
                    'inference_model_url': inference_model_url,
                    'backend_parameters': backend_parameters,
                    }

    # set ab params
    update_plan_params[test_plan]()
    update_exec_params(input_params)
    click.secho("Starting AB benchmark suite...", fg='green')
    click.secho(f"\n\nConfigured execution parameters are:", fg='green')
    click.secho(f"{execution_params}", fg="blue")

    system_under_test = CppBackend(execution_params)

    try:
        system_under_test.start()

        system_under_test.check_health()
        run_benchmark()
        generate_report()

    except Exception as e:
        click.secho("Exception occurred!" + str(e), fg='red')

    system_under_test.stop()

def run_benchmark():
    click.secho("\n\nExecuting Apache Bench tests ...", fg='green')
    click.secho("*Executing inference performance test...", fg='green')
    ab_cmd = f"ab -c {execution_params['concurrency']}  -n {execution_params['requests']} -p {TMP_DIR}/benchmark/input -T " \
             f"{execution_params['content_type']} {execution_params['inference_url']}/{execution_params['inference_model_url']} > {result_file}"

    execute(ab_cmd, wait=True)


def register_model():
    click.secho("*Registering model...", fg='green')
    url = execution_params['management_url'] + "/models"
    data = {'model_name': 'benchmark', 'url': execution_params['url'], 'batch_delay': execution_params['batch_delay'],
            'batch_size': execution_params['batch_size'], 'initial_workers': execution_params['workers'],
            'synchronous': 'true'}
    resp = requests.post(url, params=data)
    if not resp.status_code == 200:
        failure_exit(f"Failed to register model.\n{resp.text}")
    click.secho(resp.text)


def unregister_model():
    click.secho("*Unregistering model ...", fg='green')
    resp = requests.delete(execution_params['management_url'] + "/models/benchmark")
    if not resp.status_code == 200:
        failure_exit(f"Failed to unregister model. \n {resp.text}")
    click.secho(resp.text)


def execute(command, wait=False, stdout=None, stderr=None, shell=True, **kwargs):
    print(command)
    cmd = Popen(command, shell=shell, close_fds=True, stdout=stdout, stderr=stderr, universal_newlines=True, **kwargs)
    if wait:
        cmd.wait()
    return cmd


def execute_return_stdout(cmd):
    proc = execute(cmd, stdout=PIPE)
    return proc.communicate()[0].strip()


def docker_torchserve_start():
    prepare_docker_dependency()
    enable_gpu = ''
    if execution_params['image']:
        docker_image = execution_params['image']
        if execution_params['gpus']:
            enable_gpu = f"--gpus {execution_params['gpus']}"
    else:
        if execution_params['gpus']:
            docker_image = "pytorch/torchserve:latest-gpu"
            enable_gpu = f"--gpus {execution_params['gpus']}"
        else:
            docker_image = "pytorch/torchserve:latest"
        execute(f"docker pull {docker_image}", wait=True)

    backend_profiling = ''
    if execution_params['backend_profiling']:
        backend_profiling = '-e TS_BENCHMARK=True'

    # delete existing ts conatiner instance
    click.secho("*Removing existing ts conatiner instance...", fg='green')
    execute('docker rm -f ts', wait=True)

    click.secho(f"*Starting docker container of image {docker_image} ...", fg='green')
    inference_port = urlparse(execution_params['inference_url']).port
    management_port = urlparse(execution_params['management_url']).port
    docker_run_cmd = f"docker run {execution_params['docker_runtime']} {backend_profiling} --name ts --user root -p {inference_port}:{inference_port} -p {management_port}:{management_port} " \
                     f"-v {TMP_DIR}:/tmp {enable_gpu} -itd {docker_image} " \
                     f"\"torchserve --start --model-store /home/model-server/model-store " \
                         f"--ts-config /tmp/benchmark/conf/{execution_params['config_properties_name']} > /tmp/benchmark/logs/model_metrics.log\""
    execute(docker_run_cmd, wait=True)
    time.sleep(5)

def getAPIS():
    MANAGEMENT_API = "http://127.0.0.1:8081"
    INFERENCE_API = "http://127.0.0.1:8080"

    with open(execution_params['config_properties'], "r") as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if "management_address" in line:
            MANAGEMENT_API = line.split("=")[1]
        if "inference_address" in line:
            INFERENCE_API = line.split("=")[1]

    execution_params['inference_url'] = INFERENCE_API
    execution_params['management_url'] = MANAGEMENT_API
    execution_params['config_properties_name'] = execution_params['config_properties'].strip().split("/")[-1]


def update_exec_params(input_param):
    for k, v in input_param.items():
        if default_ab_params[k] != input_param[k]:
            execution_params[k] = input_param[k]
    getAPIS()


def generate_report():
    click.secho("\n\nGenerating Reports...", fg='green')
    extract_metrics()
    generate_csv_output()
    generate_latency_graph()
    generate_profile_graph()
    click.secho("\nTest suite execution complete.", fg='green')


metrics = {"predict.txt": "PredictionTime",
           "handler_time.txt": "HandlerTime",
           "waiting_time.txt": "QueueTime",
           "worker_thread.txt": "WorkerThreadTime"}


def extract_metrics():
    with open(metric_log) as f:
        lines = f.readlines()

    for k, v in metrics.items():
        all_lines = []
        pattern = re.compile(v)
        for line in lines:
            if pattern.search(line):
                all_lines.append(line.split("|")[0].split(':')[3].strip())

        out_fname = f'{TMP_DIR}/benchmark/{k}'
        click.secho(f"\nWriting extracted {v} metrics to {out_fname} ", fg='green')
        with open(out_fname, 'w') as outf:
            all_lines = map(lambda x: x + '\n', all_lines)
            outf.writelines(all_lines)


def generate_csv_output():
    click.secho("*Generating CSV output...", fg='green')
    batched_requests = execution_params['requests'] / execution_params['batch_size']
    line50 = int(batched_requests / 2)
    line90 = int(batched_requests * 9 / 10)
    line99 = int(batched_requests * 99 / 100)
    artifacts = {}
    with open(f'{TMP_DIR}/benchmark/result.txt') as f:
        data = f.readlines()
    artifacts['Benchmark'] = "AB"
    artifacts['Model'] = execution_params['url']
    artifacts['Concurrency'] = execution_params['concurrency']
    artifacts['Requests'] = execution_params['requests']
    artifacts['TS failed requests'] = extract_entity(data, 'Failed requests:', -1)
    artifacts['TS throughput'] = extract_entity(data, 'Requests per second:', -3)
    artifacts['TS latency P50'] = extract_entity(data, '50%', -1)
    artifacts['TS latency P90'] = extract_entity(data, '90%', -1)
    artifacts['TS latency P99'] = extract_entity(data, '99%', -1)
    artifacts['TS latency mean'] = extract_entity(data, 'Time per request:.*mean\)', -3)
    artifacts['TS error rate'] = int(artifacts['TS failed requests']) / execution_params['requests'] * 100

    with open(os.path.join(TMP_DIR, 'benchmark/predict.txt')) as f:
        lines = f.readlines()
        lines.sort(key=float)
        artifacts['Model_p50'] = lines[line50].strip()
        artifacts['Model_p90'] = lines[line90].strip()
        artifacts['Model_p99'] = lines[line99].strip()

    for m in metrics:
        df = pd.read_csv(f"{TMP_DIR}/benchmark/{m}", header=None, names=['data'])
        artifacts[m.split('.txt')[0] + "_mean"] = df['data'].values.mean().round(2)

    with open(os.path.join(TMP_DIR, 'benchmark/ab_report.csv'), 'w') as csv_file:
        csvwriter = csv.writer(csv_file)
        csvwriter.writerow(artifacts.keys())
        csvwriter.writerow(artifacts.values())

    return artifacts


def extract_entity(data, pattern, index, delim=" "):
    pattern = re.compile(pattern)
    for line in data:
        if pattern.search(line):
            return line.split(delim)[index].strip()


def generate_latency_graph():
    click.secho("*Preparing graphs...", fg='green')
    df = pd.read_csv(os.path.join(TMP_DIR, 'benchmark/predict.txt'), header=None, names=['latency'])
    iteration = df.index
    latency = df.latency
    a4_dims = (11.7, 8.27)
    plt.figure(figsize=(a4_dims))
    plt.xlabel('Requests')
    plt.ylabel('Prediction time')
    plt.title('Prediction latency')
    plt.bar(iteration, latency)
    plt.savefig(f"{TMP_DIR}/benchmark/predict_latency.png")


def generate_profile_graph():
    click.secho("*Preparing Profile graphs...", fg='green')

    plot_data = {}
    for m in metrics:
        df = pd.read_csv(f'{TMP_DIR}/benchmark/{m}', header=None)
        m = m.split('.txt')[0]
        plot_data[f"{m}_index"] = df.index
        plot_data[f"{m}_values"] = df.values

    if execution_params['requests'] > 100:
        sampling = int(execution_params['requests'] / 100)
    else:
        sampling = 1
    print(f"Working with sampling rate of {sampling}")

    a4_dims = (11.7, 8.27)
    grid = plt.GridSpec(3, 2, wspace=0.2, hspace=0.2)
    plt.figure(figsize=a4_dims)
    fig1 = plt.subplot(grid[0, 0])
    fig2 = plt.subplot(grid[0, 1])
    fig3 = plt.subplot(grid[1, 0])
    fig4 = plt.subplot(grid[1, 1])
    fig5 = plt.subplot(grid[2, 0:])

    def plot_line(fig, data, color='blue', title=None):
        fig.set_title(title)
        fig.set_ylabel('Time (ms)')
        fig.set_xlabel('Percentage of queries')
        fig.grid()
        plot_points = np.arange(0, 100, 100 / len(data))
        x = plot_points[:len(data):sampling]
        y = data[::sampling]
        fig.plot(x, y, f'tab:{color}')

    # Queue Time
    plot_line(fig1, data=plot_data["waiting_time_values"], color='pink', title='Queue Time')

    # handler Predict Time
    plot_line(fig2, data=plot_data["handler_time_values"], color='orange',
              title='Handler Time(pre & post processing + inference time)')

    # Worker time
    plot_line(fig3, data=plot_data["worker_thread_values"], color='green', title='Worker Thread Time')

    # Predict Time
    plot_line(fig4, data=plot_data["predict_values"], color='red',
              title='Prediction time(handler time+python worker overhead)')

    # Plot in one graph
    plot_line(fig5, data=plot_data["waiting_time_values"], color='pink')
    plot_line(fig5, data=plot_data["handler_time_values"], color='orange')
    plot_line(fig5, data=plot_data["predict_values"], color='red')
    plot_line(fig5, data=plot_data["worker_thread_values"], color='green', title='Combined Graph')
    fig5.grid()
    plt.savefig("api-profile1.png", bbox_inches='tight')


def stop_torchserve():
    if execution_params['exec_env'] == 'local':
        click.secho("*Terminating Torchserve instance...", fg='green')
        execute("torchserve --stop", wait=True)
    else:
        click.secho("*Removing benchmark container 'ts'...", fg='green')
        execute('docker rm -f ts', wait=True)
    click.secho("Apache Bench Execution completed.", fg='green')


# Test plans (soak, vgg11_1000r_10c,  vgg11_10000r_100c,...)
def soak():
    execution_params['requests'] = 100000

    execution_params['concurrency'] = 10


def vgg11_1000r_10c():
    execution_params['url'] = 'https://torchserve.pytorch.org/mar_files/vgg11.mar'
    execution_params['requests'] = 1000
    execution_params['concurrency'] = 10


def vgg11_10000r_100c():
    execution_params['url'] = 'https://torchserve.pytorch.org/mar_files/vgg11.mar'
    execution_params['requests'] = 10000
    execution_params['concurrency'] = 100


def resnet152_batch():
    execution_params['url'] = 'https://torchserve.pytorch.org/mar_files/resnet-152-batch.mar'
    execution_params['requests'] = 1000
    execution_params['concurrency'] = 10
    execution_params['batch_size'] = 4


def resnet152_batch_docker():
    execution_params['url'] = 'https://torchserve.pytorch.org/mar_files/resnet-152-batch.mar'
    execution_params['requests'] = 1000
    execution_params['concurrency'] = 10
    execution_params['batch_size'] = 4
    execution_params['exec_env'] = 'docker'


def custom():
    pass


update_plan_params = {
    "soak": soak,
    "vgg11_1000r_10c": vgg11_1000r_10c,
    "vgg11_10000r_100c": vgg11_10000r_100c,
    "resnet152_batch": resnet152_batch,
    "resnet152_batch_docker": resnet152_batch_docker,
    "custom": custom
}


def failure_exit(msg):
    import sys
    click.secho(f"{msg}", fg='red')
    click.secho(f"Test suite terminated due to above failure", fg='red')
    sys.exit()


class SystemUnderTest(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def start(self):
        raise NotImplementedError

    @abstractmethod
    def check_health(self):
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        raise NotImplementedError

class CppBackend(SystemUnderTest):
    def __init__(self, execution_params):
        self._execution_params = execution_params
        self._handle = None

    def start(self):
        click.secho("\n\nPreparing local execution...", fg='green')
        click.secho("*Setting up environment...", fg='green')
        self.prepare_common_dependency()
        click.secho("*Starting local instance...", fg='green')
        # self._handle = execute(f"../backend/eager/build/cpp_backend_poc_eager {execution_params['inference_url']}"
        parameters = f"{execution_params['inference_url']} {execution_params['url']} {execution_params['workers']}"
        parameters += f" {execution_params['backend_parameters']}" if execution_params['backend_parameters'] else ""
        self._handle = execute(f"../backend/eager/build/cpp_backend_poc_eager {parameters}"
                f" > {TMP_DIR}/benchmark/logs/model_metrics.log", preexec_fn=os.setsid)
        time.sleep(3)

    def check_health(self):
        return self._check_torchserve_health()

    def stop(self):
        if self._handle:
            click.secho("\n\nStopping local instance...", fg="green")
            os.killpg(os.getpgid(self._handle.pid), signal.SIGINT)
            self._handle.wait()

    def prepare_common_dependency(self):
        input = self._execution_params['input']
        print(input)
        shutil.rmtree(os.path.join(TMP_DIR, "benchmark"), ignore_errors=True)
        os.makedirs(os.path.join(TMP_DIR, "benchmark/conf"), exist_ok=True)
        os.makedirs(os.path.join(TMP_DIR, "benchmark/logs"), exist_ok=True)

        shutil.copy(self._execution_params['config_properties'], os.path.join(TMP_DIR, 'benchmark/conf/'))
        shutil.copyfile(input, os.path.join(TMP_DIR, 'benchmark/input'))

    def _check_torchserve_health(self):
        attempts = 3
        retry = 0
        click.secho("*Testing system health...", fg='green')
        click.secho(TMP_DIR)
        while retry < attempts:
            try:
                click.secho(self._execution_params['inference_url'] + "/ping")
                resp = requests.get("http://localhost:8090/ping", headers={'User-Agent': 'Mozilla/5.0'})
                if resp.status_code == 200:
                    click.secho(resp.text)
                    return True
                else:
                    click.secho(f"Got wrong return code: {resp.status_code}")
            except Exception as e:
                retry += 1
            time.sleep(3)
            click.secho("*Retrying testing system health...", fg='green')
        failure_exit("Could not connect to Tochserve instance at " + execution_params['inference_url'])
        return False


if __name__ == '__main__':
    benchmark()
