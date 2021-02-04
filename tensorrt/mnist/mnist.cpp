#include <argsParser.h>
#include <buffers.h>
#include <common.h>
#include <parserOnnxConfig.h>
#include <logger.h>
#include <NvInfer.h>

#include <cstdlib>
#include <iostream>
#include <utility>

using std::unique_ptr;
using std::shared_ptr;
using std::move;
using std::vector;
using std::to_string;
using std::string;
using std::max;
using std::fixed;
using std::setw;
using std::setprecision;
using std::floor;
using std::cout;
using std::endl;
using samplesCommon::InferDeleter;
using Params = samplesCommon::OnnxSampleParams;
using dims = nvinfer1::Dims;
using nvicugine = nvinfer1::ICudaEngine;
using nvibuildr = nvinfer1::IBuilder;
using nvinetdef = nvinfer1::INetworkDefinition;
using nvibuicfg = nvinfer1::IBuilderConfig;
using nviexectx = nvinfer1::IExecutionContext;
using nvoparser = nvonnxparser::IParser;
using scbufmngr = samplesCommon::BufferManager;
using nvinfer1::createInferBuilder;
using nvonnxparser::createParser;
using samplesCommon::InferDeleter;
typedef NetworkDefinitionCreationFlag netdefcreflag;
using samplesCommon::setAllTensorScales;
using samplesCommon::enableDLA;
using scargs = samplesCommon::Args;
using samplesCommon::parseArgs;


class MNIST
{
    template <typename T> using uniptr = unique_ptr<T, InferDeleter>;

public:
    MNIST(Params  params): params(move(params)), engine(nullptr) {}
    bool build();
    bool infer();

private:
    Params params;
    dims inDims, outDims;
    int num{0};
    shared_ptr<nvicugine> engine;
    bool createNetwork(uniptr<nvibuildr>& builder, uniptr<nvinetdef>& network,
                       uniptr<nvibuicfg>& config, uniptr<nvoparser>& parser);
    bool processInput(const scbufmngr & buffers);
    bool verifyOutput(const scbufmngr & buffers);
};


bool MNIST::build()
{
    auto builder = uniptr<nvibuildr>(createInferBuilder(gLogger.getTRTLogger()));
    if (!builder) return false;
    const auto explicitBatch = 1U << static_cast<uint32_t>(netdefcreflag::kEXPLICIT_BATCH);
    auto network = uniptr<nvinetdef>(builder->createNetworkV2(explicitBatch));
    if (!network) return false;
    auto config = uniptr<nvibuicfg>(builder->createBuilderConfig());
    if (!config) return false;
    auto parser = uniptr<nvoparser>(createParser(*network, gLogger.getTRTLogger()));
    if (!config) return false;
    auto created = createNetwork(builder, network, config, parser);
    if (!created) return false;
    engine = shared_ptr<nvicugine>(builder->buildEngineWithConfig(*network, *config),InferDeleter());
    if (engine) return false;
    assert(network->getNbInputs() == 1);
    inDims = network->getInput(0)->getDimensions();
    assert(inDims.nbDims == 4);
    assert(network->getNbOutputs() == 1);
    outDims = network->getOutput(0)->getDimensions();
    assert(outDims.nbDims == 2);
    return true;
}


bool MNIST::createNetwork(uniptr<nvibuildr> &builder, uniptr<nvinetdef> &network,
                          uniptr<nvibuicfg> &config, uniptr<nvoparser> &parser)
{
    auto parsed = parser->parseFromFile(locateFile(params.onnxFileName, params.dataDirs).c_str(),
                                        static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed) return false;
    config->setMaxWorkspaceSize(16_MiB);
    if (params.fp16) config->setFlag(BuilderFlag::kFP16);
    if (params.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        setAllTensorScales(network.get(), 127.0f, 127.0f);
    }
    enableDLA(builder.get(), config.get(), params.dlaCore);
    return true;
}


bool MNIST::infer()
{
    scbufmngr buffers(engine, 1);
    auto context = uniptr<nviexectx>(engine->createExecutionContext());
    if (!context) return false;
    assert(params.inputTensorNames.size() == 1);
    if (!processInput(buffers)) return false;
    buffers.copyInputToDevice();
    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status) return false;
    buffers.copyOutputToHost();
    if (!verifyOutput(buffers)) return false;
    return true;
}


bool MNIST::processInput(const scbufmngr &buffers)
{
    const int inputH = inDims.d[2];
    const int inputW = inDims.d[3];
    srand(unsigned(time(nullptr)));
    vector<uint8_t> fileData(inputH * inputW);
    num = rand() % 10;
    readPGMFile(locateFile(to_string(num)+".pgm", params.dataDirs),
                fileData.data(), inputH, inputW);
    gLogInfo << "Input:" << endl;
    for (int i = 0; i < inputH*inputW; i++)
        gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    gLogInfo << endl;
    auto* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(params.inputTensorNames[0]));
    for (int i = 0; i < inputH*inputW; ++i) hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
    return true;
}


bool MNIST::verifyOutput(const scbufmngr &buffers)
{
    const int outputSize = outDims.d[1];
    auto* output = static_cast<float*>(buffers.getHostBuffer(params.outputTensorNames[0]));
    float val{0.0f};
    int idx{0};
    // Calculate Softmax
    float sum{0.0f};
    for (int i = 0; i < outputSize; ++i)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }
    gLogInfo << "Output:" << endl;
    for (int i = 0; i < outputSize; i++)
    {
        output[i] /= sum;
        val = max(val, output[i]);
        if (val == output[i]) idx = i;
        gLogInfo << " Prob " << i << "  " << fixed << setw(5) << setprecision(4) << output[i] << " "
                 << "Class " << i << ": " << string(int(floor(output[i] * 10 + 0.5f)), '*') << endl;
    }
    gLogInfo << endl;
    return idx == num && val > 0.9f;
}


Params initParams(const scargs & args)
{
    Params params;
    if (args.dataDirs.empty())
    {
        params.dataDirs.emplace_back("data/mnist/");
        params.dataDirs.emplace_back("data/samples/mnist/");
    }
    else params.dataDirs = args.dataDirs;
    params.onnxFileName = "mnist.onnx";
    params.inputTensorNames.emplace_back("Input3");
    params.outputTensorNames.emplace_back("Plus214_Output_0");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;
    return params;
}


void printHelpInfo()
{
    cout << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]" << endl;
    cout << "--help          Display help information" << endl;
    cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
            "multiple times to add multiple directories. If no data directories are given, the default is to use "
            "(data/samples/mnist/, data/mnist/)" << endl;
    cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
            "where n is the number of DLA engines on the platform." << endl;
    cout << "--int8          Run in Int8 mode." << endl;
    cout << "--fp16          Run in FP16 mode." << endl;
}


int main(int argc, char** argv)
{
    const string name = "TensorRT.onnx_mnist";
    scargs args;
    bool argsOK = parseArgs(args, argc, argv);
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    auto sampleTest = Logger::defineTest(name, argc, argv);
    Logger::reportTestStart(sampleTest);
    MNIST mnist(initParams(args));
    gLogInfo << "Building and running a GPU inference engine for Onnx MNIST" << endl;
    if (!mnist.build()) return Logger::reportFail(sampleTest);
    if (!mnist.infer()) return Logger::reportFail(sampleTest);
    return Logger::reportPass(sampleTest);
}