#ifndef CORE_DIP_HPP
#define CORE_DIP_HPP

#include <functional>
#include <cstdint>

#include <opencv2/opencv.hpp>

namespace DIP
{
    template<typename T>
    inline std::uint8_t clamp(T v)
    {
        return
            v < 0 ? 0 : (
                v > 255 ? 255 : static_cast<std::uint8_t>(v));
    }

    void changEachPixel1To1(
        cv::Mat& src, cv::Mat& dst,
        std::function<void(int, int, std::uint8_t*, std::uint8_t*)>&& callback);
}

#endif
