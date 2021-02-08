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
    explicit DynamicReshape(const samplesCommon::OnnxSampleParams& params): params(params) {}
    bool build();
    bool prepare();
    bool infer();

private:
    samplesCommon::OnnxSampleParams params;
    nvinfer1::Dims inputDims, outputDims;
    bool buildPredictEngine(const uniptr<nvinfer1::IBuilder>& builder);
    bool buildPreprocEngine(const uniptr<nvinfer1::IBuilder>& builder);
    Dims loadPGMFile(const std::string& filename);
    bool validateOutput(int digit);
    uniptr<nvinfer1::ICudaEngine> predictEngine{nullptr};
    uniptr<nvinfer1::ICudaEngine> preprocEngine{nullptr};
    uniptr<nvinfer1::IExecutionContext> predictContext{nullptr};
    uniptr<nvinfer1::IExecutionContext> preprocContext{nullptr};
    samplesCommon::ManagedBuffer input{};
    samplesCommon::DeviceBuffer predictInput{};
    samplesCommon::ManagedBuffer output{};
    template<typename T> uniptr<T> makeUnique(T* t)
    {
        if (!t) throw std::runtime_error{"Failed to create TensorRT object !"};
        return uniptr<T>{t};
    }
};


bool DynamicReshape::buildPreprocEngine(const uniptr<nvinfer1::IBuilder> &builder)
{
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto preprocNetwork = makeUnique(builder->createNetworkV2(explicitBatch));
    auto input = preprocNetwork->addInput("input", nvinfer1::DataType::kFLOAT, Dims4{1, 1, -1, -1});
    auto resizeLayer = preprocNetwork->addResize(*input);
    resizeLayer->setOutputDimensions(inputDims);
    preprocNetwork->markOutput(*resizeLayer->getOutput(0));
    auto preprocConfig = makeUnique(builder->createBuilderConfig());
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(input->getName(), OptProfileSelector::kMIN, Dims4{1, 1, 1, 1});
    profile->setDimensions(input->getName(), OptProfileSelector::kOPT, Dims4{1, 1, 28, 28});
    profile->setDimensions(input->getName(), OptProfileSelector::kMAX, Dims4{1, 1, 56, 56});
    preprocConfig->addOptimizationProfile(profile);
    preprocEngine = makeUnique(builder->buildEngineWithConfig(*preprocNetwork, *preprocConfig));
    gLogInfo << "Profile dimensions in preprocessor engine:" << std::endl;
    gLogInfo << "    Minimum = " << preprocEngine->getProfileDimensions(0, 0, OptProfileSelector::kMIN) << std::endl;
    gLogInfo << "    Optimum = " << preprocEngine->getProfileDimensions(0, 0, OptProfileSelector::kOPT) << std::endl;
    gLogInfo << "    Maximum = " << preprocEngine->getProfileDimensions(0, 0, OptProfileSelector::kMAX) << std::endl;
    return true;
}


bool DynamicReshape::buildPredictEngine(const uniptr<nvinfer1::IBuilder> &builder)
{
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = makeUnique(builder->createNetworkV2(explicitBatch));
    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());
    bool parseSuccess = parser->parseFromFile(locateFile(params.onnxFileName, params.dataDirs).c_str(),
                                              static_cast<int>(gLogger.getReportableSeverity()));
    if (!parseSuccess) throw std::runtime_error{"Failed to parse model !"};
    auto softmax = network->addSoftMax(*network->getOutput(0));
    softmax->setAxes(1 << 1);
    network->unmarkOutput(*network->getOutput(0));
    network->markOutput(*softmax->getOutput(0));
    inputDims = network->getInput(0)->getDimensions();
    outputDims = network->getOutput(0)->getDimensions();
    auto config = makeUnique(builder->createBuilderConfig());
    config->setMaxWorkspaceSize(16_MiB);
    if (params.fp16) config->setFlag(BuilderFlag::kFP16);
    if (params.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }
    predictEngine = makeUnique(builder->buildEngineWithConfig(*network, *config));
    return true;
}


bool DynamicReshape::build()
{
    auto builder = makeUnique(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!buildPredictEngine(builder))
    {
        gLogError << "Build Prediction Engine failed !" << std::endl;
        return false;
    }
    if (!buildPreprocEngine(builder))
    {
        gLogError << "Build Preprosessor Engine failed !" << std::endl;
        return false;
    }
    return true;
}


bool DynamicReshape::prepare()
{
    preprocContext = makeUnique(preprocEngine->createExecutionContext());
    predictContext = makeUnique(predictEngine->createExecutionContext());
    predictInput.resize(inputDims);
    output.hostBuffer.resize(outputDims);
    output.deviceBuffer.resize(outputDims);
    return true;
}


Dims DynamicReshape::loadPGMFile(const std::string &filename)
{
    std::ifstream infile(filename, std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from file that is not opened");
    std::string magic;
    int h, w, max;
    infile >> magic >> h >> w >> max;
    infile.seekg(1, infile.cur);
    Dims4 inputDims{1, 1, h, w};
    size_t vol = samplesCommon::volume(inputDims);
    std::vector<uint8_t> fileData(vol);
    infile.read(reinterpret_cast<char*>(fileData.data()), vol);
    gLogInfo << "Input:" << std::endl;
    for (size_t i = 0; i < vol; ++i) gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % w) ? "" : "\n");
    gLogInfo << std::endl;
    input.hostBuffer.resize(inputDims);
    float* hostDataBuffer = static_cast<float*>(input.hostBuffer.data());
    std::transform(fileData.begin(), fileData.end(), hostDataBuffer, [](uint8_t x){return 1.0-static_cast<float>(x/255.0);});
    return inputDims;
}


bool DynamicReshape::validateOutput(int digit)
{
    ;
}


bool DynamicReshape::infer()
{
    std::random_device rd{};
    std::default_random_engine generator{rd()};
    std::uniform_int_distribution<int> digitDistribution{0, 9};
    int digit = digitDistribution(generator);
    Dims inputDims = loadPGMFile(locateFile(std::to_string(digit)+".pgm", params.dataDirs));
    input.deviceBuffer.resize(inputDims);
    CHECK(cudaMemcpy(input.deviceBuffer.data(), input.hostBuffer.data(), input.hostBuffer.nbBytes(), cudaMemcpyHostToDevice));
    preprocContext->setBindingDimensions(0, inputDims);
    if (!preprocContext->allInputDimensionsSpecified()) return false;
    std::vector<void*> preprocBingings = {input.deviceBuffer.data(), predictInput.data()};
    bool status = preprocContext->executeV2(preprocBingings.data());
    if (!status) return false;
    std::vector<void*> predictBindings = {predictInput.data(), output.deviceBuffer.data()};
    status = predictContext->executeV2(predictBindings.data());
    if (!status) return false;
    CHECK(cudaMemcpy(output.hostBuffer.data(), output.deviceBuffer.data(), output.deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost));
    return validateOutput(digit);
}


samplesCommon::OnnxSampleParams initParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty())
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
    auto test = Logger::defineTest("dynamic_reshape", argc, argv);
    Logger::reportTestStart(test);
    DynamicReshape reshape(initParams(args));
    if (!reshape.build())
    {
        gLogError << "Building failed !" << std::endl;
        return gLogger.reportFail(test);
    }
    if (!reshape.prepare())
    {
        gLogError << "Preparing failed !!" << std::endl;
        return gLogger.reportFail(test);
    }
    if (!reshape.infer())
    {
        gLogError << "Inferring failed !!!" << std::endl;
        return gLogger.reportFail(test);
    }
    return gLogger.reportPass(test);
}