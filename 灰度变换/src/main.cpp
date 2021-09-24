#include <cmath>

#include "DIP.hpp"

//对数变换
void logVariation(cv::Mat& src, cv::Mat& dst, double c = 1.0f, double v = 1.0)
{
    std::uint8_t table[256];
    for (std::size_t i = 0; i < 256; i++)
        //使用换底公式计算，公式d = c * log(1 + v * s), s和d范围[0, 1.0]，构建转换表
        table[i] = static_cast<std::uint8_t>(std::round(c * 255.0 * (std::log2(1.0 + v * i / 255.0) / std::log2(1 + v))));

    DIP::changEachPixel1To1(src, dst,
        [&](int i, int j, std::uint8_t* srcData, std::uint8_t* dstData)
        {
            *dstData = table[*srcData];
        });
}

//分段线性变换
void lineVariation(cv::Mat& src, cv::Mat& dst, std::vector<std::pair<std::uint8_t, std::uint8_t>>&& ref)
{
    std::uint8_t table[256];
    std::size_t begin = 0;
    //当前灰度转换后的值
    double value = 0.0;
    //起始点(0, 0)
    auto p0 = std::make_pair<std::uint8_t, std::uint8_t>(0, 0);
    //中止点(255, 255)
    ref.emplace_back(std::make_pair<std::uint8_t, std::uint8_t>(255, 255));

    table[0] = 0;
    table[255] = 255;

    //计算并构建转换表
    for (auto& p : ref)
    {
        for (std::size_t i = begin; i < p.first; i++)
            table[i] = static_cast<std::uint8_t>(
                std::round(value += static_cast<double>(p.second - p0.second) / static_cast<double>(p.first - p0.first)));

        p0 = p;
        begin = p.first;
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

    logVariation(src, dst, 1.0, 1.0);
    cv::imshow("src", src);
    cv::imshow("dst", dst);
    cv::waitKey();
   
    lineVariation(src, dst, { {0.3*255, 0.15*255}, {0.7*255, 0.85*255} });
    cv::imshow("src", src);
    cv::imshow("dst", dst);
    cv::waitKey();
    return 0;
}
