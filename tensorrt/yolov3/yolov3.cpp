#include <buffers.h>
#include <argsParser.h>
#include <common.h>
#include <logger.h>
#include <parserOnnxConfig.h>
#include <NvInfer.h>
#include <memory>
#include <utility>
#include <opencv2/opencv.hpp>


class YOLOv3
{
    template<typename T> using uniptr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    explicit YOLOv3(samplesCommon::OnnxSampleParams params): params(std::move(params)) {};
    bool build();
    bool infer();

protected:
    static cv::Mat letterBox(const cv::Mat& img, int longsz);
    bool preproc(void *buffers);
    bool postproc();

private:
    samplesCommon::OnnxSampleParams params;
    std::shared_ptr<nvinfer1::ICudaEngine> engine{nullptr};
    nvinfer1::Dims inDims{}, outDims{};
};


bool YOLOv3::build()
{
    auto builder = uniptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        gLogError << "Builder creation failed !" << std::endl;
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
    auto parsed = parser->parseFromFile(locateFile(params.onnxFileName, params.dataDirs).c_str(),
                                        static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed)
    {
        gLogError << "Onnx Parsing failed !" << std::endl;
        return false;
    }
    builder->setMaxBatchSize(params.batchSize);
    config->setMaxWorkspaceSize(256_MiB);
    if (params.fp16) config->setFlag(BuilderFlag::kFP16);
    if (params.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }
    samplesCommon::enableDLA(builder.get(), config.get(), params.dlaCore);
    engine = shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config),
                                               samplesCommon::InferDeleter());
    if (!engine)
    {
        gLogError << "Engine Creation failed !" << std::endl;
        return false;
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


cv::Mat YOLOv3::letterBox(const cv::Mat &img, int longsz)
{
    cv::Mat resized;
    if (img.cols >= img.rows)
    {
        float ratio = (float)longsz / (float)img.cols;
        int new_height = round(ratio * img.rows - 1e-5);
        float pad_h = (longsz - new_height) % 32 / 2.;
        if (new_height != longsz) resize(img, resized, cv::Size(longsz, new_height));
        int top = round(pad_h - 0.1);
        int bottom = round(pad_h + 0.1);
        copyMakeBorder(resized, resized, top, bottom, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
    }
    else
    {
        float ratio = (float)longsz / (float)img.rows;
        int new_width = round(ratio * img.cols - 1e-5);
        float pad_w = (longsz - new_width) % 32 / 2.;
        if (new_width != longsz) resize(img, resized, cv::Size(new_width, longsz));
        int left = round(pad_w - 0.1);
        int right = round(pad_w + 0.1);
        copyMakeBorder(resized, resized, 0, 0, left, right, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
    }
    return resized;
}


bool YOLOv3::preproc(void *buffers)
{
    int inputH = inDims.d[2];
    int inputW = inDims.d[3];
    cv::Mat img = cv::imread(locateFile("test.jpg", params.dataDirs));
    if (img.empty())
    {
        gLogError << "Reading Image failed !" << std::endl;
        return false;
    }
    cv::Mat transed;
    cv::cvtColor(img, transed, cv::COLOR_BGR2RGB);
    cv::Mat resized = letterBox(img, std::max(inputH, inputW));
    resized.convertTo(resized, CV_32FC3, 1.0, 1.0);
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    for (auto& channel: channels)
    {
        int copyNum = (float*)channel.dataend - (float*)channel.datastart;
        memcpy((void*)buffers, channel.data, copyNum * sizeof(float));
    }
}


bool YOLOv3::infer()
{
    samplesCommon::BufferManager buffers(engine, params.batchSize);
    auto context = uniptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context)
    {
        gLogError << "Execution Context creation failed !" << std::endl;
        return false;
    }
    assert(params.inputTensorNames.size() == 1);
    if (!preproc(buffers))
    {
        gLogError << "Preprocessing failed !" << std::endl;
        return false;
    }
}


samplesCommon::OnnxSampleParams initParams(samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) params.dataDirs.emplace_back("data/yolov3");
    else params.dataDirs = args.dataDirs;
    params.onnxFileName = "model/yolov3.onnx";
    params.inputTensorNames.emplace_back("Input3");
    params.batchSize = 1;
    params.outputTensorNames.emplace_back("Plus214_Output_0");
    params.dlaCore = args.useDLACore;
    params.fp16 = args.runInFp16;
    params.int8 = args.runInInt8;
    return params;
}


int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        gLogError << "Invalid arguments !" << std::endl;
        return EXIT_FAILURE;
    }
    auto test = Logger::defineTest("yolov3", argc, argv);
    Logger::reportTestStart(test);
    YOLOv3 yolov3(initParams(args));
    gLogInfo << "Building and running a GPU inference engine for ONNX YOLOv3" << std::endl;
    if (yolov3.build())
    {
        gLogError << "Model Building failed !" << std::endl;
        return Logger::reportFail(test);
    }
    if (yolov3.infer())
    {
        gLogError << "Model inferring failed !" << std::endl;
        return Logger::reportFail(test);
    }
    return Logger::reportPass(test);
}