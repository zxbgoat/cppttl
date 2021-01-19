#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

const std::string gSampleName = "TensorRT.sample_onnx_mnist";


class SampleOnnxMNIST
{
    template <typebame T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

    public:
        SampleOnnxMNIST(const samplesCommon::OnnxSampleParam& params)
            : mParams(params), mEngine(nullptr) {}
        bool build();  // function builds the network engine
        bool infer();  // runs the TensorRT inference engine for this sample
    
    private:
        sampelsCommon::OnnxSampleParams mParams;  // parameters for the sample
        nvinfer1::Dims mInputDims;   // dimensions of the input to the network
        nvinfer1::Dims mOutputDims;  // dimensions of the output to the network
        int mNumber{0};              // number to classify
        std::shared_ptr<nvinfer1::ICudaEngine> mEngine;  // TensorRT engine used to run the network

        // Parses an ONNX model for MNIST and creates a TensorRT network
        bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                              SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
                              SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                              SampleUniquePtr<nvonnxparser::IParser>& parser);
        
        // Reads the input  and stores the result in a managed buffer
        bool processInput(const samplesCommon::BufferManager& buffers);

        // Classifies digits and verify result
        bool verifyOutput(const samplesCommon::BufferManager& buffers);
};


bool SampleOnnxMNIST::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                                       SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
                                       SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                                       SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
                                        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if(!parsed) return false;
    config->setMaxWorkspaceSize(16_MiB);
    if(mParams.fp16) config->setFlag(BuilderFlag::kFP16);
    if(mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }
    sampelsCommon::enableDLA(network.get(), config.get(), mParams.dlaCore);
    return true;
}


bool SampleOnnxMNIST::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if(!builder) return false;
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if(!network) return false;
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if(!config) return false;
    auto parser = SampleUniquePtr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if(!parser) return false;
    auto constructed = constructNetwork(builder, network, config, parser);
    if(!constructed) return false;
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config),
                                                     samplesCommon::InferDeleter());
    if(!mEngine) return false;
    assert(network->getNbInputs() == 1);
    assert(network->getInput(0)->getDimensions() == 4);
    assert(network->getNbOutputs() == 1);
    assert(network->getOutput(0)->getDimensions() == 2);
    return true;
}


void printHelpInfo()
{
    std::cout << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
              << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}


samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if(args.dataDirs.empty())  // Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("dara/mnist");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else  // Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "mnist.onnx";
    params.inputTensorNames.push_back("Input3");
    params.outputTensorNames.push_back("Plus214_Output_0");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;
    return params;
}


int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if(!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if(args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);
    sample::gLogger.reportTestStart(sampleTest);
    SampleOnnxMNIST sample(initializeSampleParams(args));
    sample::gLogInfo << "Building and running a GPU inference engine for Onnx MNIST" << std::endl;
    if(!sample.build()) return sample::gLogger.reportFail(sampleTest);
}