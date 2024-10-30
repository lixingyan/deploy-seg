#include "TrtModel.h"
#include <algorithm>
#include <cstring>
#include <memory>
#include <type_traits>
#include "cuda_runtime.h"
#include <sys/stat.h>


inline bool file_exists(const std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}


//!初始化推理引擎，如果没有推理引擎，则从onnx模型构建推理引擎
TrtModel::TrtModel(std::string onnxfilepath, bool fp16, int maxbatch)
    :onnx_file_path{onnxfilepath}, FP16(fp16), MaxBatch(maxbatch)
{
    auto idx = onnx_file_path.find(".onnx");
    auto basename = onnx_file_path.substr(0, idx);
    m_enginePath = basename + ".engine";

    if (file_exists(m_enginePath)){
        std::cout << "start building model from engine file: " << m_enginePath;
        this->Runtime();
    }else{
        std::cout << "start building model from onnx file: " << onnx_file_path;
        this->genEngine();
        this->Runtime();
    }
    this->trtIOMemory();
}


bool TrtModel::constructNetwork(
    std::unique_ptr<nvinfer1::IBuilder>& builder,
    std::unique_ptr<nvinfer1::INetworkDefinition>& network,
    std::unique_ptr<nvinfer1::IBuilderConfig>& config,
    std::unique_ptr<nvonnxparser::IParser>& parser)
{
    // 读取onnx模型文件开始构建模型
    auto parsed = parser->parseFromFile(onnx_file_path.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));
    if(!parsed){
        std::cout<<" (T_T)~~~ ,Failed to parse onnx file."<<std::endl;
        return false;
    }

    auto input = network->getInput(0);
    auto input_dims = input->getDimensions();
    auto profile = builder->createOptimizationProfile(); 

    // 配置最小、最优、最大范围
    input_dims.d[0] = 1;                                                         
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
    input_dims.d[0] = MaxBatch;
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    config->addOptimizationProfile(profile);


    // 判断是否使用半精度优化模型
    if(FP16)  config->setFlag(nvinfer1::BuilderFlag::kFP16);


    config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);


    // 设置默认设备类型为 DLA
    config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);

    // 获取 DLA 核心支持情况
    int numDLACores = builder->getNbDLACores();
    if (numDLACores > 0) {
        std::cout << "DLA is available. Number of DLA cores: " << numDLACores << std::endl;

        // 设置 DLA 核心
        int coreToUse = 0; // 选择第一个 DLA 核心（可以根据实际需求修改）
        config->setDLACore(coreToUse);
        std::cout << "Using DLA core: " << coreToUse << std::endl;
    } else {
        std::cerr << "DLA not available on this platform, falling back to GPU." << std::endl;
        
        // 如果 DLA 不可用，则设置 GPU 回退
        config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        config->setDefaultDeviceType(nvinfer1::DeviceType::kGPU);
    }
    return true;
}


bool TrtModel::genEngine(){

    // 打印模型编译过程的日志
    sample::gLogger.setReportableSeverity(nvinfer1::ILogger::Severity::kVERBOSE);

    // 创建builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if(!builder){
        std::cout<<" (T_T)~~~, Failed to create builder."<<std::endl;
        return false;
    }

    // 声明显性batch，创建network
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if(!network){
        std::cout<<" (T_T)~~~, Failed to create network."<<std::endl;
        return false;
    }

    // 创建 config
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if(!config){
        std::cout<<" (T_T)~~~, Failed to create config."<<std::endl;
        return false;
    }

    // 创建parser 从onnx自动构建模型，否则需要自己构建每个算子
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if(!parser){
        std::cout<<" (T_T)~~~, Failed to create parser."<<std::endl;
        return false;
    }

    // 为网络设置config, 以及parse
    auto constructed = this->constructNetwork(builder, network, config, parser);
    if (!constructed)
    { 
        std::cout<<" (T_T)~~~,  Failed to Create an optimization profile and calibration configuration. (•_•)~ "<<std::endl;
        return false;
    }

    builder->setMaxBatchSize(1);
    // config->setMaxWorkspaceSize(1<<30);     /*在比较新的版本中，这个接口已经被弃用*/
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30);      /*在新的版本中被使用*/

    auto profileStream = samplesCommon::makeCudaStream();
    if(!profileStream){
        std::cout<<" (T_T)~~~, Failed to makeCudaStream."<<std::endl;
        return false;
    }
    config->setProfileStream(*profileStream);

    // 创建序列化引擎文件
    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if(!plan){
        std::cout<<" (T_T)~~~, Failed to SerializedNetwork."<<std::endl;
        return false;
    }

    //! 检查输入部分是否符合要求
    if(network->getNbInputs() == 1){
        auto mInputDims = network->getInput(0)->getDimensions();
        std::cout<<" ✨~ model input dims: "<<mInputDims.nbDims <<std::endl;
        for(size_t ii=0; ii<mInputDims.nbDims; ++ii){
            std::cout<<" ✨^_^ model input dim"<<ii<<": "<<mInputDims.d[ii] <<std::endl;
        }
    } else {
        std::cout<<" (T_T)~~~, please check model input shape "<<std::endl;
        return false;
    }

    //! 检查输出部分是否符合要求
    if(network->getNbOutputs() == 2){
        for(size_t i=0; i<network->getNbOutputs(); ++i){
            auto mOutputDims = network->getOutput(i)->getDimensions();
            std::cout<<" ✨~ model output dims: "<<mOutputDims.nbDims <<std::endl;
            for(size_t jj=0; jj<mOutputDims.nbDims; ++jj){
                std::cout<<" ✨^_^ model output dim"<<jj<<": "<<mOutputDims.d[jj] <<std::endl;
            }
        }
        
    } else {
        std::cout<<" (T_T)~~~, please check model output shape "<<std::endl;
        return false;
    }

    // 序列化保存推理引擎文件文件
    std::ofstream engine_file(m_enginePath, std::ios::binary);
    if(!engine_file.good()){
        std::cout<<" (T_T)~~~, Failed to open engine file"<<std::endl;
        return false;
    }
    engine_file.write((char *)plan->data(), plan->size());
    engine_file.close();

    std::cout << " ~~Congratulations! 🎉🎉🎉~  Engine build success!!! ✨✨✨~~ " << std::endl;

}


std::vector<unsigned char>TrtModel::load_engine_file()
{
    std::vector<unsigned char> engine_data;
    std::ifstream engine_file(m_enginePath, std::ios::binary);
    if(!engine_file.is_open()){
        std::cout<<" (T_T)~~~, Unable to load engine file O_O."<<std::endl;
        return engine_data;
    }
    engine_file.seekg(0, engine_file.end);
    int length = engine_file.tellg();
    engine_data.resize(length);
    engine_file.seekg(0, engine_file.beg);
    engine_file.read(reinterpret_cast<char *>(engine_data.data()), length);
    return engine_data;
}


inline std::string printDims(const nvinfer1::Dims dims){
    int n = 0;
    char buff[100];
    std::string result;

    n += snprintf(buff + n, sizeof(buff) - n, "[ ");
    for (int i = 0; i < dims.nbDims; i++){
        n += snprintf(buff + n, sizeof(buff) - n, "%d", dims.d[i]);
        if (i != dims.nbDims - 1) {
            n += snprintf(buff + n, sizeof(buff) - n, ", ");
        }
    }
    n += snprintf(buff + n, sizeof(buff) - n, " ]");
    result = buff;
    return result;
}


bool TrtModel::trtIOMemory() {

    m_inputDims   = m_context->getBindingDimensions(0);
    m_segmentDims = m_context->getBindingDimensions(1);
    m_detectDims  = m_context->getBindingDimensions(2);
    this->kInputH = m_inputDims.d[2];
    this->kInputW = m_inputDims.d[3];
    this->ImageC  = m_inputDims.d[1];

    std::cout<<"after optimizer input shape: "<<m_context->getBindingDimensions(0)<<std::endl;
    std::cout<<"after optimizer output1 shape: "<<m_context->getBindingDimensions(1)<<std::endl;
    std::cout<<"after optimizer output2 shape: "<<m_context->getBindingDimensions(2)<<std::endl;

    CUDA_CHECK(cudaStreamCreate(&m_stream));

    m_inputSize = m_inputDims.d[0] * m_inputDims.d[1] * m_inputDims.d[2] * m_inputDims.d[3] * sizeof(float);
    m_imgArea = m_inputDims.d[2] * m_inputDims.d[3];
    m_detectSize = m_detectDims.d[0] * m_detectDims.d[1] * m_detectDims.d[2] * sizeof(float);
    m_segmentSize = m_segmentDims.d[1] * m_segmentDims.d[2] * m_segmentDims.d[3] * sizeof(float);

    // 改进这里的内存分配错误处理
    if (cudaMallocHost(&m_inputMemory[0], m_inputSize) != cudaSuccess) {
        std::cerr << "Failed to allocate host memory for input. Error code: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        return false;
    }
    if (cudaMallocHost(&m_detectMemory[0], m_detectSize) != cudaSuccess) {
        std::cerr << "Failed to allocate host memory for detection. Error code: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        cudaFreeHost(m_inputMemory[0]);
        return false;
    }
    if (cudaMallocHost(&m_segmentMemory[0], m_segmentSize) != cudaSuccess) {
        std::cerr << "Failed to allocate host memory for segmentation. Error code: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        cudaFreeHost(m_inputMemory[0]);
        cudaFreeHost(m_detectMemory[0]);
        return false;
    }
    if (cudaMalloc(&m_inputMemory[1], m_inputSize) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for input. Error code: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        cudaFreeHost(m_inputMemory[0]);
        cudaFreeHost(m_detectMemory[0]);
        cudaFreeHost(m_segmentMemory[0]);
        return false;
    }
    if (cudaMalloc(&m_detectMemory[1], m_detectSize) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for detection. Error code: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        cudaFreeHost(m_inputMemory[0]);
        cudaFreeHost(m_detectMemory[0]);
        cudaFreeHost(m_segmentMemory[0]);
        cudaFree(m_inputMemory[1]);
        return false;
    }
    if (cudaMalloc(&m_segmentMemory[1], m_segmentSize) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for segmentation. Error code: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        cudaFreeHost(m_inputMemory[0]);
        cudaFreeHost(m_detectMemory[0]);
        cudaFreeHost(m_segmentMemory[0]);
        cudaFree(m_inputMemory[1]);
        cudaFree(m_detectMemory[1]);
        return false;
    }


    m_bindings[0] = m_inputMemory[1];
    m_bindings[1] = m_segmentMemory[1];
    m_bindings[2] = m_detectMemory[1];

    return true; // 成功时返回true
}


bool TrtModel::Runtime(){

    // 初始化trt插件
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");
    
    // 加载序列化的推理引擎
    auto plan = this->load_engine_file();

    // / 打印模型推理过程的日志
    sample::setReportableSeverity(sample::Severity::kINFO);

    // 创建推理引擎
    m_runtime.reset(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if(!m_runtime){
        std::cout<<" (T_T)~~~, Failed to create runtime."<<std::endl;
        return false;
    }

    // 反序列化推理引擎
    m_engine.reset(m_runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if(!m_engine){
        std::cout<<" (T_T)~~~, Failed to deserialize."<<std::endl;
        return false;
    }

    // 获取优化后的模型的输入维度和输出维度
    // int nbBindings = m_engine->getNbBindings(); // trt8.5 以前版本
    int nbBindings = m_engine->getNbIOTensors();  // trt8.5 以后版本
    for (int i = 0; i < nbBindings; i++)
    {
        auto dims = m_engine->getBindingDimensions(i);

        auto size = dims.d[0]*dims.d[1]*dims.d[2]*dims.d[3]*sizeof(float);

        auto name = m_engine->getBindingName(i);
        auto bingdingType = m_engine->getBindingDataType(i);

        std::cout << "Binding " << i << ": " << name << ", size: " << size << ", dims: " << dims << ", type: " << int(bingdingType) << std::endl;
    }

    // 推理执行上下文
    m_context.reset(m_engine->createExecutionContext());
    if(!m_context){
        std::cout<<" (T_T)~~~, Failed to create ExecutionContext."<<std::endl;
        return false;
    }

    auto input_dims = m_context->getBindingDimensions(0);
    input_dims.d[0] = MaxBatch;

    // 设置当前推理时，input大小
    m_context->setBindingDimensions(0, input_dims);

    std::cout << " ~~Congratulations! 🎉🎉🎉~  runtime deserialize success!!! ✨✨✨~~ " << std::endl;

}


cv::Mat TrtModel::letterbox(cv::Mat srcimg, int *newh, int *neww, int *top, int *left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->kInputH;
	*neww = this->kInputW;
	cv::Mat dstimg;
	if (srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->kInputH; 
			*neww = int(this->kInputW / hw_scale);
			cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*left = int((this->kInputW - *neww) * 0.5);
			cv::copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->kInputW - *neww - *left, cv::BORDER_CONSTANT, 0);
		} else {
			*newh = (int)this->kInputH * hw_scale;
			*neww = this->kInputW;
			cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*top = (int)(this->kInputH - *newh) * 0.5;
			cv::copyMakeBorder(dstimg, dstimg, *top, this->kInputH - *newh - *top, 0, 0, cv::BORDER_CONSTANT, 0);
		}
	}
	else {
		cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
	}
	return dstimg;
}


void TrtModel::processConvert(cv::Mat& frame, int* newh, int* neww, int* top, int* left)
{

    if (frame.data == nullptr)  {std::cerr<<"ERROR: Image file not founded! Program terminated";}
    auto letterboxImage = this->letterbox(frame, newh, neww, top, left);

    /*Preprocess -- host端进行normalization和BGR2RGB, NHWC->NCHW*/
    if (letterboxImage.channels()==3){
        int index {0};
        int offset_ch0 = m_imgArea * 0;
        int offset_ch1 = m_imgArea * 1;
        int offset_ch2 = m_imgArea * 2;
        for (int i = 0; i < m_inputDims.d[2]; i++) {
            for (int j = 0; j < m_inputDims.d[3]; j++) {
                index = i * m_inputDims.d[3] * m_inputDims.d[1] + j * m_inputDims.d[1];
                m_inputMemory[0][offset_ch2++] = letterboxImage.data[index + 0] / 255.0f;
                m_inputMemory[0][offset_ch1++] = letterboxImage.data[index + 1] / 255.0f;
                m_inputMemory[0][offset_ch0++] = letterboxImage.data[index + 2] / 255.0f;
            }
    }
    /*Preprocess -- 将host的数据移动到device上*/
    CUDA_CHECK(cudaMemcpyAsync(m_inputMemory[1], m_inputMemory[0], m_inputSize, cudaMemcpyHostToDevice, m_stream));
    }else if (letterboxImage.channels()==1){
        int index {0};
        int offset_ch = m_imgArea*0 ;

        for (int i = 0; i < m_inputDims.d[2]; i++) {
            for (int j = 0; j < m_inputDims.d[3]; j++) {
                index = i * m_inputDims.d[3] * m_inputDims.d[1] + j * m_inputDims.d[1];
                m_inputMemory[0][offset_ch++] = letterboxImage.data[index + 0] / 255.0f;
            }
    }
    /*Preprocess -- 将host的数据移动到device上*/
    CUDA_CHECK(cudaMemcpyAsync(m_inputMemory[1], m_inputMemory[0], m_inputSize, cudaMemcpyHostToDevice, m_stream));
    }

}


void TrtModel::nms(std::vector<BoxInfo>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; }); // 降序排列

	std::vector<bool> remove_flags(input_boxes.size(),false);

	auto iou = [](const BoxInfo& box1,const BoxInfo& box2)
	{
		float xx1 = std::max(box1.x1, box2.x1);
		float yy1 = std::max(box1.y1, box2.y1);
		float xx2 = std::min(box1.x2, box2.x2);
		float yy2 = std::min(box1.y2, box2.y2);
		// 交集
		float w = std::max(0.0f, xx2 - xx1 + 1);
		float h = std::max(0.0f, yy2 - yy1 + 1);
		float inter_area = w * h;
		// 并集
		float union_area = std::max(0.0f,box1.x2-box1.x1) * std::max(0.0f,box1.y2-box1.y1)
						   + std::max(0.0f,box2.x2-box2.x1) * std::max(0.0f,box2.y2-box2.y1) - inter_area;
		return inter_area / union_area;
	};
	for (int i = 0; i < input_boxes.size(); ++i)
	{
		if(remove_flags[i]) continue;
		for (int j = i + 1; j < input_boxes.size(); ++j)
		{
			if(remove_flags[j]) continue;
			if(input_boxes[i].label == input_boxes[j].label && iou(input_boxes[i],input_boxes[j])>=this->kNmsThresh)
			{
				remove_flags[j] = true;
			}
		}
	}

	int idx_t = 0;
    // remove_if()函数 remove_if(beg, end, op) //移除区间[beg,end)中每一个“令判断式:op(elem)获得true”的元素
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &remove_flags](const BoxInfo& f) { return remove_flags[idx_t++]; }), input_boxes.end());
}


inline float computeIoU(const cv::Rect& box1, const cv::Rect& box2) {
    int intersectionArea = (box1 & box2).area();
    int unionArea = (box1 | box2).area();
    return static_cast<float>(intersectionArea) / static_cast<float>(unionArea);
}


void TrtModel::NMS(const std::vector<cv::Rect>& boxes,
                const std::vector<float>& confidences,
                std::vector<int>& nms_result,
                float confThresh,
                float nmsThresh) {
    // 根据置信度进行筛选
    std::vector<int> indices;
    for (size_t i = 0; i < confidences.size(); ++i) {
        if (confidences[i] > confThresh) {
            indices.push_back(i);
        }
    }

    // 按照置信度从高到低排序
    std::sort(indices.begin(), indices.end(),
              [&confidences](int a, int b) { return confidences[a] > confidences[b]; });

    // 执行 NMS
    while (!indices.empty()) {
        int currentIndex = indices.front(); // 取出置信度最高的框
        nms_result.push_back(currentIndex);
        indices.erase(indices.begin()); // 移除该框

        // 移除与当前框重叠度高的框
        for (auto it = indices.begin(); it != indices.end();) {
            if (computeIoU(boxes[currentIndex], boxes[*it]) > nmsThresh) {
                it = indices.erase(it); // 移除重叠度高的框
            } else {
                ++it;
            }
        }
    }
}


std::vector<OutputSeg>  TrtModel::doInference(cv::Mat& frame) {

    std::vector<int> classIds;      // 实例分割的id
	std::vector<float> confidences; // 实例分割的置信度
	std::vector<cv::Rect> boxes;    // 检测出来的框坐标
    std::vector<std::vector<float>> picked_proposals;  // 挑选出来output0[:,:, 5 + _className.size():m_detectDims.d[2]]的map-mask信息值

    int newh = 0, neww = 0, padh = 0, padw = 0; // 做letterbox的padding，用于解码坐标值

    this->processConvert(frame, &newh, &neww, &padh, &padw);

    bool status = this->m_context->enqueueV2((void**)m_bindings, m_stream, nullptr);
    if(!status){
        std::cout<<"(T_T)~~~, Failed to execute inference, Please check your input and output."<<std::endl;
    }

    CUDA_CHECK(cudaMemcpyAsync(m_segmentMemory[0], m_segmentMemory[1], m_segmentSize, cudaMemcpyKind::cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaMemcpyAsync(m_detectMemory[0], m_detectMemory[1], m_detectSize, cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));

	float ratio_h = (float)frame.rows / newh;
	float ratio_w = (float)frame.cols / neww;

	// step1: 先处理好bbox部分，完成bbox的解码任务
    float* pdata = m_detectMemory[0];

	for (int j = 0; j < m_detectDims.d[1]; ++j) {
		float box_score = pdata[4]; // 获取分割的每个目标的置信度
		if (box_score >= this->kConfThresh) {
			cv::Mat scores(1, m_detectDims.d[2] - 5 - m_segmentDims.d[1], CV_32FC1, pdata + 5);
			cv::Point classIdPoint;
			double max_class_socre;
			cv::minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
			max_class_socre = (float)max_class_socre;
			if (max_class_socre >= this->kConfThresh) {

				std::vector<float> temp_proto(pdata + m_detectDims.d[2] - m_segmentDims.d[1], pdata + m_detectDims.d[2]);
				picked_proposals.push_back(temp_proto);

				float x = (pdata[0] - padw) * ratio_w;  //x
				float y = (pdata[1] - padh) * ratio_h;  //y
				float w = pdata[2] * ratio_w;  //w
				float h = pdata[3] * ratio_h;  //h

				int left = std::max(int(x - 0.5 * w) , 0);
				int top = std::max(int(y - 0.5 * h) , 0);
				classIds.push_back(classIdPoint.x);
				confidences.push_back(max_class_socre * box_score);
				boxes.push_back(cv::Rect(left, top, int(w ), int(h )));
			}
		}
		pdata += m_detectDims.d[2];
	}


    // 对检测出来的框做nms操作
    std::vector<int> nms_result; // 保存符合要求的检测框的索引
	// cv::dnn::NMSBoxes(boxes, confidences, this->kConfThresh, this->kNmsThresh, nms_result);
    this->NMS(boxes, confidences, nms_result, this->kConfThresh, this->kNmsThresh);


    std::vector<std::vector<float>> temp_mask_proposals;

    cv::Rect holeImgRect(0, 0, frame.cols, frame.rows);

	std::vector<OutputSeg> output; // 存放检测的结果

    for (int i = 0; i < nms_result.size(); ++i) {
		int idx = nms_result[i];
		OutputSeg result;
		result.id = classIds[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx] & holeImgRect;
		output.push_back(result);
		temp_mask_proposals.push_back(picked_proposals[idx]);
	}


	// step2: 处理mask-map，完成的解码
    pdata = m_segmentMemory[0];
	cv::Mat maskProposals;


    if(temp_mask_proposals.size()!=0){ 
        for (int i = 0; i < temp_mask_proposals.size(); ++i){
            // std::cout<<"ppppppqqqqqqqqqqqq"<< cv::Mat(temp_mask_proposals[i]).t().size()<<std::endl;
            maskProposals.push_back( cv::Mat(temp_mask_proposals[i]).t() );
        }

        // for (size_t i = 0; i < 10; ++i) {
        // std::cout << pdata[i] << " ";
        // }

        std::vector<float> mask(pdata, pdata + m_segmentDims.d[0] * m_segmentDims.d[1] * m_segmentDims.d[2] * m_segmentDims.d[3]);

        cv::Mat mask_protos = cv::Mat(mask);

        cv::Mat protos = mask_protos.reshape(0, { m_segmentDims.d[1], m_segmentDims.d[2] * m_segmentDims.d[3] });//mask-map mask_protos

        cv::Mat matmulRes = (maskProposals * protos).t(); // n*32 32*25600 

        cv::Mat masks = matmulRes.reshape(output.size(), {m_segmentDims.d[2] , m_segmentDims.d[3]});

        std::vector<cv::Mat> maskChannels;

        cv::split(masks, maskChannels);

        for (int i = 0; i < output.size(); ++i) {

            cv::Mat dest, mask;
            // sigmoid操作
            cv::exp(-maskChannels[i], dest);
            
            dest = 1.0 / (1.0 + dest);//160 * 160

            cv::Rect roi(int((float)padw / this->kInputW * m_segmentDims.d[3]), int((float)padh / this->kInputH * m_segmentDims.d[2]), \
            int(m_segmentDims.d[3] - padw / 2), int(m_segmentDims.d[2] - padh / 2));

            dest = dest(roi);
            
            cv::resize(dest, mask, frame.size(), cv::INTER_NEAREST);

            // crop ----提取大于置信度的掩码部分
            cv::Rect temp_rect = output[i].box;
            mask = mask(temp_rect) > this->kMaskThresh;

            output[i].boxMask = mask;
        }
    }
    return output;
}


bool TrtModel::drawResult(cv::Mat& img, std::vector<OutputSeg> result) {

    // 为掩码创建随机颜色列表
    std::vector<cv::Scalar> colors(10000);
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < 10000; i++) {
        colors[i] = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
    }

    // 创建一个与原图相同大小的黑色掩码
    cv::Mat mask = cv::Mat::zeros(img.size(), img.type());

    for (const auto& seg : result) {

        // 绘制分割框
        cv::rectangle(img, seg.box, colors[seg.id], 2);

        if(seg.boxMask.empty()) continue;

        // 在掩码上填充当前检测框的颜色，仅限于当前对象的区域
        mask(seg.box).setTo(colors[seg.id], seg.boxMask);

        // 计算轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(seg.boxMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::drawContours(mask(seg.box), contours, -1, cv::Scalar(255,255,255), 3);
        double area = cv::contourArea(contours[0]); // 计算当前轮廓的面积
        // 准备标签
        std::string label = std::to_string(seg.id) + ": " + std::to_string(seg.confidence) + 
                            " Area: " + std::to_string(area);
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top = std::max(seg.box.y, labelSize.height);

        // 绘制标签
        cv::putText(img, label, cv::Point(seg.box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, colors[seg.id], 1);

    }

    // 把原图与掩码合并
    cv::addWeighted(img, 0.5, mask, 0.5, 0, img);

    return true; // 如果需要返回状态，可以返回true
}


