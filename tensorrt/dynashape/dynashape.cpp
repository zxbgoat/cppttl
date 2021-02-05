#include <buffers.h>
#include <argsParser.h>
#include <common.h>
#include <logger.h>
#include <parserOnnxConfig.h>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <random>


class DynamicReshape
{
    template<typename T> using uniptr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    DynamicReshape(const samplesCommon::OnnxSampleParams& params): params(params) {}
    bool build();
    bool prepare();
    bool infer();

private:
    samplesCommon::OnnxSampleParams params;
};


samplesCommon::OnnxSampleParams initParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else params.dataDirs = args.dataDirs;
    params.onnxFileName = "mnist.onnx";
    params.inputTensorNames.push_back("Input3");
    params.outputTensorNames.push_back("Plus214_Output_0");
    params.batchSize = 1;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;
    return params;
}


void printHelpInfo()
{
    std::cout << "Usage: ./sample_dynamic_reshape [-h or --help] [-d or --datadir=<path to data directory>]" << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)" << std::endl;
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
    auto test = gLogger.defineTest("dynamic_reshape", argc, argv);
    gLogger.reportTestStart(test);
    DynamicReshape reshape(initParams(args));
    reshape.build();
    reshape.prepare();
    reshape.infer();
}