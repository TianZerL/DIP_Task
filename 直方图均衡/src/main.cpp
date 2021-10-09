#include <cmath>

#include "DIP.hpp"

void histogramEqualization(cv::Mat& src, cv::Mat& dst)
{
    std::uint8_t table[256] = { 0 };
    std::uint64_t count[256] = { 0 };
    double probability[256] = { 0.0 };

    //统计灰度值
    DIP::changEachPixel1To1(src, dst,
        [&](int i, int j, std::uint8_t* srcData, std::uint8_t* dstData)
        {
            count[*srcData]++ ;
        });

    for (size_t i = 0; i < 256; i++)
    {
        double value = 0;
        //计算概率
        probability[i] = static_cast<double>(count[i]) / static_cast<double>(src.size().area());

        //累加概率
        for (size_t j = 0; j < i; j++)
            value += probability[j];

        //计算新的灰度值
        table[i] =  DIP::clamp(std::round(value * 255.0));
    }

    DIP::changEachPixel1To1(src, dst,
        [&](int i, int j, std::uint8_t* srcData, std::uint8_t* dstData)
        {
            *dstData = table[*srcData];
        });
}


int main()
{
    cv::Mat src = cv::imread("F:/Temp/Anime4K/p1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat dst;
    cv::Mat ref;

    histogramEqualization(src, dst);
    cv::equalizeHist(src, ref);
    cv::imshow("原图", src);
    cv::imshow("直方图均衡", dst);
    cv::imshow("OpenCV自带实现", ref);
    cv::waitKey();
    return 0;
}
