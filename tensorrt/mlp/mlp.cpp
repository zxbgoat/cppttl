#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

using std::string;


struct MLPParams: public samplesCommon::SampleParams
{
    int input_h;
    int input_;
    int output_size;
    string wts_file;
};


class MLP
{};