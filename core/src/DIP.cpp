#include "DIP.hpp"

void DIP::changEachPixel1To1(
    cv::Mat& src, cv::Mat& dst, 
    std::function<void(int, int, std::uint8_t*, std::uint8_t*)>&& callback)
{
    const int w = src.cols;
    const int h = src.rows;
    const std::size_t step = src.step;
    dst.create(h, w, src.type());

    for (int i = 0; i < h; i++)
    {
        std::uint8_t* srcLineData = src.data + static_cast<std::size_t>(i) * step;
        std::uint8_t* dstLineData = dst.data + static_cast<std::size_t>(i) * step;
        for (int j = 0; j < w; j++)
        {
            callback(i, j, srcLineData + j, dstLineData + j);
        }
    }
}
