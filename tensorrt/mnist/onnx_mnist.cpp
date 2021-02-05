#include <common.h>
#include <logger.h>
#include <buffers.h>
#include <argsParser.h>
#include <parserOnnxConfig.h>
#include <NvInfer.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>

class OnnxMNIST
{
    template<typename T> using uniptr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    explicit OnnxMNIST(samplesCommon::OnnxSampleParams  params): params(std::move(params)), engine(nullptr) {};
    bool build();
    bool infer();

private:
    samplesCommon::OnnxSampleParams params;
    nvinfer1::Dims inDims{};
    nvinfer1::Dims outDims{};
    int num{0};
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    bool createNetwork(uniptr<nvinfer1::IBuilder>& builder,
                       uniptr<nvinfer1::INetworkDefinition>& network,
                       uniptr<nvinfer1::IBuilderConfig>& config,
                       uniptr<nvonnxparser::IParser>& parser);
    bool processInput(const samplesCommon::BufferManager& buffers);
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};


bool OnnxMNIST::build()
{
    auto builder = uniptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        gLogError << "Infer Builder creation failed !" << std::endl;
        return false;
    }
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = uniptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        gLogError << "Network Definition creation failed !" << std::endl;
        return false;
    }
    auto config = uniptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        gLogError << "Builder Config creation failed !" << std::endl;
        return false;
    }
    auto parser = uniptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser)
    {
        gLogError << "Onnx Parser creation failed !" << std::endl;
        return false;
    }
    auto created = createNetwork(builder, network, config, parser);
    if (!created)
    {
        gLogError << "Network creation failed !" << std::endl;
        return false;
    }
    engine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config),
                                                    samplesCommon::InferDeleter());
    if (!engine)
    {
        gLogError << "Engine creation failed !" << std::endl;
    }
    gLogInfo << "Checking num of inputs ..." << std::endl;
    assert(network->getNbInputs() == 1);
    inDims = network->getInput(0)->getDimensions();
    gLogInfo << "Checking dim of inputs ..." << std::endl;
    assert(inDims.nbDims == 4);
    gLogInfo << "Checking num of outputs ..." << std::endl;
    assert(network->getNbOutputs() == 1);
    outDims = network->getOutput(0)->getDimensions();
    gLogInfo << "Checking dim of outputs ..." << std::endl;
    assert(outDims.nbDims == 2);
    return true;
}


bool OnnxMNIST::createNetwork(uniptr<nvinfer1::IBuilder> &builder,
                              uniptr<nvinfer1::INetworkDefinition> &network,
                              uniptr<nvinfer1::IBuilderConfig> &config,
                              uniptr<nvonnxparser::IParser> &parser)
{
    auto parsed = parser->parseFromFile(locateFile(params.onnxFileName, params.dataDirs).c_str(),
                                        static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed)
    {
        gLogError << "Onnx file parsing failed !" << std::endl;
        return false;
    }
    builder->setMaxBatchSize(params.batchSize);
    config->setMaxWorkspaceSize(16_MiB);
    if (params.fp16) config->setFlag(BuilderFlag::kFP16);
    if (params.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }
    samplesCommon::enableDLA(builder.get(), config.get(), params.dlaCore);
    return true;
}


bool OnnxMNIST::infer()
{
    samplesCommon::BufferManager buffers(engine, params.batchSize);
    auto context = uniptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context)
    {
        gLogError << "Execution Context creation failed !" << std::endl;
        return false;
    }
    assert(params.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        gLogError << "Input Processing failed !" << std::endl;
        return false;
    }
    buffers.copyOutputToHost();
    if (!verifyOutput(buffers))
    {
        gLogError << "Output Verifying failed !" << std::endl;
        return false;
    }
    return true;
}


bool OnnxMNIST::processInput(const samplesCommon::BufferManager &buffers)
{
    const int inputH = inDims.d[2];
    const int inputW = inDims.d[3];
    srand(unsigned(time(nullptr)));
    std::vector<uint8_t> fileData(inputH * inputW);
    num = rand() % 10;
    readPGMFile(locateFile(std::to_string(num)+".pgm", params.dataDirs),
                fileData.data(), inputH, inputW);
    gLogInfo << "Input:" << std::endl;
    for (int i = 0; i < inputH*inputW; ++i)
        gLogInfo << (" .:-=+*#%@"[fileData[i]/26]) << (((i+1) % inputW) ? "" : "\n");
    gLogInfo << std::endl;
    float* hostbuf = static_cast<float*>(buffers.getHostBuffer(params.inputTensorNames[0]));
    for (int i = 0; i < inputH*inputW; ++i)
        hostbuf[i] = 1.0 - float(fileData[i] / 255.);
    return true;
}


bool OnnxMNIST::verifyOutput(const samplesCommon::BufferManager &buffers)
{
    const int outputSize = outDims.d[1];
    float* output = static_cast<float*>(buffers.getHostBuffer(params.outputTensorNames[0]));
    int idx{0};
    float sum{0.0f}, val{0.0f};
    for (int i = 0; i < outputSize; ++i)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }
    gLogInfo << "Output" << std::endl;
    for (int i = 0; i < )
    {
        output[i] /= sum;
        val = std::max(val, output[i]);
        if (val == output[i]) idx = i;
        gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i] << " "
                 << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*') << std::endl;
    }
    gLogInfo << std::endl;
    return idx == num && val > 0.9f;
}


samplesCommon::OnnxSampleParams initParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty())
    {
        params.dataDirs.push_back("data/mnist");
        params.dataDirs.push_back("data/samples/mnist");
    }
    else params.dataDirs = args.dataDirs;
    params.onnxFileName = "mnist.onnx";
    params.inputTensorNames.push_back("Input3");
    params.batchSize = 1;
    params.outputTensorNames.push_back("Plus214_Output_0");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;
    return params;
}


void printHelpInfo()
{
    std::cout << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]" << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)" << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform." << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}


int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        gLogError << "Invalid arguments !" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_FAILURE;
    }
    auto test = gLogger.defineTest("onnx_mnist", argc, argv);
    gLogger.reportTestStart(test);
    OnnxMNIST mnist(initParams(args));
    gLogInfo << "Building and running a GPU inference engine for Onnx MNIST" << std::endl;
    if (!mnist.build())
    {
        gLogError << "Model building failed !!" << std::endl;
        return gLogger.reportFail(test);
    }
    if(!mnist.infer())
    {
        gLogError << "Model inferring failed !!!" << std::endl;
        return gLogger.reportFail(test);
    }
    return gLogger.reportPass(test);
}