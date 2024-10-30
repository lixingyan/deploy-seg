#ifndef TRTMODEL_H
#define TRTMODEL_H

#include "NvInfer.h"
#include "NvOnnxParser.h" // onnxparser头文件
#include "logger.h"
#include "common.h"
#include "buffers.h"
#include "cassert"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <deque>
#include <array>
#include <cuda_runtime_api.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif  // CUDA_CHECK


typedef struct BoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int label;
} BoxInfo;


typedef struct OutputSeg {
	int id;             //id
	float confidence;   //conf
	cv::Rect box;       //box
	cv::Mat boxMask;    //mask-map
    // 默认构造函数
    OutputSeg() : id(-1), confidence(0.0f), box(), boxMask() {}
} OutputSeg;



class TrtModel
{
public:

    std::vector<OutputSeg> doInference(cv::Mat& frame);

    bool drawResult(cv::Mat& img, std::vector<OutputSeg> result);

    TrtModel(std::string onnxfilepath, bool fp16, int maxbatch);  
    ~TrtModel() = default;                                          /*使用默认析构函数*/


private:
    bool trtIOMemory();

    /* 创建网络 */
    bool constructNetwork(
        std::unique_ptr<nvinfer1::IBuilder>& builder,
        std::unique_ptr<nvinfer1::INetworkDefinition>& network, 
        std::unique_ptr<nvinfer1::IBuilderConfig>& config,
        std::unique_ptr<nvonnxparser::IParser>& parser);



	bool genEngine();                                               /*onnx转为engine */
    std::vector<unsigned char> load_engine_file();                  /*加载engine模型*/

    bool Runtime();                                                 /*从engine穿件推理运行时，执行上下文*/
    void processConvert(cv::Mat& frame, int* newh, int* neww, int* top, int* left);

    void nms(std::vector<BoxInfo>& input_boxes);                    /*模型后处理NMS*/
    void NMS(const std::vector<cv::Rect>& boxes,
                const std::vector<float>& confidences,
                std::vector<int>& nms_result, float confThresh,float nmsThresh);
    cv::Mat letterbox(cv::Mat srcimg, int *newh, int *neww, int *top, int *left);   /*对图像letterbox预处理*/

    std::shared_ptr<nvinfer1::IRuntime> m_runtime;                   /*声明模型的推理运行时指针*/
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;                 /*声明模型反序列化指针*/
    std::shared_ptr<nvinfer1::IExecutionContext> m_context;          /*声明模型执行上下文指针*/

    float* m_bindings[3];                                           /*为模型申请输入输出的缓冲区地址*/
    float* m_inputMemory[2];                                        /*声明输入缓冲区地址*/
    float* m_detectMemory[2];                                       
    float* m_segmentMemory[2];                                       

    nvinfer1::Dims m_inputDims;                                     /*声明输入图片属性的索引*/
    nvinfer1::Dims m_detectDims;
    nvinfer1::Dims m_segmentDims;

    cudaStream_t   m_stream;                                        /*声明cuda流*/

    int m_iClassNums {};                                            /*声明检测类别数量变量，用于后处理*/
    int m_iBoxNums {};                                              /*声明检测的anchor数量变量*/

    std::string onnx_file_path {};                                  /*指定输入的onnx模型的地址*/
    std::string m_enginePath {};                                    /*指定生成的engine模型的地址*/
    bool FP16 {true};                                               /* 判断是否使用半精度进行面模型优化*/
    int m_inputSize {};                                             /*图像需要预处理的大小*/
    int m_imgArea {};                                               /*使用指针对图像预处理的偏移量大小，不同图像通道*/
    int m_detectSize {};
    int m_segmentSize {};

    int kInputH = 544;                                               /*模型预处理的图像的高度，最好是32的整数倍*/
    int kInputW = 640;                                               /*模型预处理的图像的宽度，最好是32的整数倍*/
    int ImageC = 3;                                                    /*原始输入图像的通道数量*/
    int MaxBatch {};
    int m_rawImgArea {};                                            /*使用指针对原始图像预处理的偏移量大小，不同图像通道*/
    int m_outputSize {};                                            /*输入图像的大小*/
    int rawImageSize {};                                            /*复制图像的大小*/
    int m_imageArea {};


    float kNmsThresh = 0.2f;                                        /*后处理的NMS的阈值*/
    float kConfThresh = 0.5f;                                       /*模型检测的置信度的阈值*/
    float kMaskThresh = 0.5;


    const std::vector<std::string> CLASS_NAMES = {  /*需要检测的目标类别*/   
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse","sheep", "cow", "elephant", "bear", "zebra", 
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis","snowboard", "sports ball", 
    "kite", "baseball bat", "baseball glove","skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich","orange", "broccoli", "carrot", "hot dog", 
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", 
    "book", "clock", "vase","scissors", "teddy bear", "hair drier", "toothbrush"};

    const std::vector<std::vector<unsigned int>> COLORS_HEX = {     /*对不同的检测目标绘制不同的颜色*/
    {0x00, 0x72, 0xBD}, {0xD9, 0x53, 0x19}, {0xED, 0xB1, 0x20}, {0x7E, 0x2F, 0x8E}, {0x77, 0xAC, 0x30}, {0x4D, 0xBE, 0xEE},
    {0xA2, 0x14, 0x2F}, {0x4C, 0x4C, 0x4C}, {0x99, 0x99, 0x99}, {0xFF, 0x00, 0x00}, {0xFF, 0x80, 0x00}, {0xBF, 0xBF, 0x00},
    {0x00, 0xFF, 0x00}, {0x00, 0x00, 0xFF}, {0xAA, 0x00, 0xFF}, {0x55, 0x55, 0x00}, {0x55, 0xAA, 0x00}, {0x55, 0xFF, 0x00},
    {0xAA, 0x55, 0x00}, {0xAA, 0xAA, 0x00}, {0xAA, 0xFF, 0x00}, {0xFF, 0x55, 0x00}, {0xFF, 0xAA, 0x00}, {0xFF, 0xFF, 0x00},
    {0x00, 0x55, 0x80}, {0x00, 0xAA, 0x80}, {0x00, 0xFF, 0x80}, {0x55, 0x00, 0x80}, {0x55, 0x55, 0x80}, {0x55, 0xAA, 0x80},
    {0x55, 0xFF, 0x80}, {0xAA, 0x00, 0x80}, {0xAA, 0x55, 0x80}, {0xAA, 0xAA, 0x80}, {0xAA, 0xFF, 0x80}, {0xFF, 0x00, 0x80},
    {0xFF, 0x55, 0x80}, {0xFF, 0xAA, 0x80}, {0xFF, 0xFF, 0x80}, {0x00, 0x55, 0xFF}, {0x00, 0xAA, 0xFF}, {0x00, 0xFF, 0xFF},
    {0x55, 0x00, 0xFF}, {0x55, 0x55, 0xFF}, {0x55, 0xAA, 0xFF}, {0x55, 0xFF, 0xFF}, {0xAA, 0x00, 0xFF}, {0xAA, 0x55, 0xFF},
    {0xAA, 0xAA, 0xFF}, {0xAA, 0xFF, 0xFF}, {0xFF, 0x00, 0xFF}, {0xFF, 0x55, 0xFF}, {0xFF, 0xAA, 0xFF}, {0x55, 0x00, 0x00},
    {0x80, 0x00, 0x00}, {0xAA, 0x00, 0x00}, {0xD4, 0x00, 0x00}, {0xFF, 0x00, 0x00}, {0x00, 0x2B, 0x00}, {0x00, 0x55, 0x00},
    {0x00, 0x80, 0x00}, {0x00, 0xAA, 0x00}, {0x00, 0xD4, 0x00}, {0x00, 0xFF, 0x00}, {0x00, 0x00, 0x2B}, {0x00, 0x00, 0x55},
    {0x00, 0x00, 0x80}, {0x00, 0x00, 0xAA}, {0x00, 0x00, 0xD4}, {0x00, 0x00, 0xFF}, {0x00, 0x00, 0x00}, {0x24, 0x24, 0x24},
    {0x49, 0x49, 0x49}, {0x6D, 0x6D, 0x6D}, {0x92, 0x92, 0x92}, {0xB6, 0xB6, 0xB6}, {0xDB, 0xDB, 0xDB}, {0x00, 0x72, 0xBD},
    {0x50, 0xB7, 0xBD}, {0x80, 0x80, 0x00}};

};
#endif // TRTMODEL_H

