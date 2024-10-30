#include <iostream>
#include "TrtModel.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <deque>
#include <array>


int main() {

    cv::VideoCapture cap("media/lihua1.avi");

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file.\n";
        return -1;
    }

    cv::Size frameSize(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double video_fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "width: " << frameSize.width << " height: " << frameSize.height << " fps: " << video_fps << std::endl;

    //初始化模型
    TrtModel trtmodel("weights/lihua.onnx",true,1);

    cv::Mat frame;
    int frame_nums = 0;

    while (cap.read(frame)) {
        if (frame.empty()) {
            std::cerr << "Error: Frame is empty." << std::endl;
            break;
        }

        auto start = std::chrono::high_resolution_clock::now();
        //推理图片结果
        auto results = trtmodel.doInference(frame);
        auto enddo = std::chrono::high_resolution_clock::now();

        auto durationdo = std::chrono::duration_cast<std::chrono::milliseconds>(enddo- start);
        std::cout << " Durationdo: " << durationdo.count() << "ms" << std::endl;

        trtmodel.drawResult(frame, results);


        auto enddw = std::chrono::high_resolution_clock::now();
        auto durationdw = std::chrono::duration_cast<std::chrono::milliseconds>(enddw- enddo);
        // std::cout << " Durationdw: " << durationdw.count() << "ms" << std::endl;


        cv::imshow("Processed Video1", frame);
        // cv::imshow("Processed Video", frame);
        frame_nums += 1;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        // std::cout << "Processed frame: " << frame_nums << " Duration: " << duration.count() << "ms" << std::endl;

        if (cv::waitKey(22) == 27) {
            break; // 按Esc键退出
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
