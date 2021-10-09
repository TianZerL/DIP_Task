#include <algorithm>

#include "DIP.hpp"

//3x3均值滤波
void meanFilter3x3(cv::Mat& src, cv::Mat& dst)
{
    DIP::changEachPixel1To1(src, dst,
        [&](int i, int j, std::uint8_t* srcData, std::uint8_t* dstData)
        {
            std::size_t in = i > 0 ? -src.cols : 0;
            std::size_t ip = i < src.rows - 1 ? src.cols : 0;
            std::size_t jn = j > 0 ? -1 : 0;
            std::size_t jp = j < src.cols - 1 ? 1 : 0;

            *dstData = DIP::clamp((
                static_cast<double>(*(srcData + in + jn)) +
                static_cast<double>(*(srcData + in)) +
                static_cast<double>(*(srcData + in + jp)) +
                static_cast<double>(*(srcData + jn)) +
                static_cast<double>(*srcData) +
                static_cast<double>(*(srcData + jp)) +
                static_cast<double>(*(srcData + ip + jn)) +
                static_cast<double>(*(srcData + ip)) +
                static_cast<double>(*(srcData + ip + jp))
                ) / 9.0);
        });
}

//中值滤波
void medianFilter3x3(cv::Mat& src, cv::Mat& dst)
{
    DIP::changEachPixel1To1(src, dst,
        [&](int i, int j, std::uint8_t* srcData, std::uint8_t* dstData)
        {
            std::size_t in = i > 0 ? -src.cols : 0;
            std::size_t ip = i < src.rows - 1 ? src.cols : 0;
            std::size_t jn = j > 0 ? -1 : 0;
            std::size_t jp = j < src.cols - 1 ? 1 : 0;

            std::vector<std::uint8_t> vec = {
                *(srcData + in + jn),
                *(srcData + in),
                *(srcData + in + jp),
                *(srcData + jn),
                *srcData,
                *(srcData + jp),
                *(srcData + ip + jn),
                *(srcData + ip),
                *(srcData + ip + jp)
            };

            std::nth_element(vec.begin(), vec.begin() + 4, vec.end());

            *dstData = vec[4];
        });
}

//最大值滤波
void maxFilter3x3(cv::Mat& src, cv::Mat& dst)
{
    DIP::changEachPixel1To1(src, dst,
        [&](int i, int j, std::uint8_t* srcData, std::uint8_t* dstData)
        {
            std::size_t in = i > 0 ? -src.cols : 0;
            std::size_t ip = i < src.rows - 1 ? src.cols : 0;
            std::size_t jn = j > 0 ? -1 : 0;
            std::size_t jp = j < src.cols - 1 ? 1 : 0;

            *dstData = std::max({
                *(srcData + in + jn),
                *(srcData + in),
                *(srcData + in + jp),
                *(srcData + jn),
                *srcData,
                *(srcData + jp),
                *(srcData + ip + jn),
                *(srcData + ip),
                *(srcData + ip + jp)
                });
        });
}

//最小值滤波
void minFilter3x3(cv::Mat& src, cv::Mat& dst)
{
    DIP::changEachPixel1To1(src, dst,
        [&](int i, int j, std::uint8_t* srcData, std::uint8_t* dstData)
        {
            std::size_t in = i > 0 ? -src.cols : 0;
            std::size_t ip = i < src.rows - 1 ? src.cols : 0;
            std::size_t jn = j > 0 ? -1 : 0;
            std::size_t jp = j < src.cols - 1 ? 1 : 0;

            *dstData = std::min({
                *(srcData + in + jn),
                *(srcData + in),
                *(srcData + in + jp),
                *(srcData + jn),
                *srcData,
                *(srcData + jp),
                *(srcData + ip + jn),
                *(srcData + ip),
                *(srcData + ip + jp)
                });
        });
}

//sobel边缘检测
void sobelFilter(cv::Mat& src, cv::Mat& dst)
{
    DIP::changEachPixel1To1(src, dst,
        [&](int i, int j, std::uint8_t* srcData, std::uint8_t* dstData)
        {
            std::size_t in = i > 0 ? -src.cols : 0;
            std::size_t ip = i < src.rows - 1 ? src.cols : 0;
            std::size_t jn = j > 0 ? -1 : 0;
            std::size_t jp = j < src.cols - 1 ? 1 : 0;

            double gradY = 0.0 +
                *(srcData + in + jn) + *(srcData + in) + *(srcData + in) + *(srcData + in + jp) -
                *(srcData + ip + jn) - *(srcData + ip) - *(srcData + ip) - *(srcData + ip + jp);

            double gradX = 0.0 +
                *(srcData + in + jn) + *(srcData + jn) + *(srcData + jn) + *(srcData + ip + jn) -
                *(srcData + in + jp) - *(srcData + jp) - *(srcData + jp) - *(srcData + ip + jp);

            *dstData = DIP::clamp(std::sqrt(gradX * gradX + gradY * gradY));
        });
}

//sobel锐化
void sobelSharpening(cv::Mat& src, cv::Mat& dst)
{
    DIP::changEachPixel1To1(src, dst,
        [&](int i, int j, std::uint8_t* srcData, std::uint8_t* dstData)
        {
            std::size_t in = i > 0 ? -src.cols : 0;
            std::size_t ip = i < src.rows - 1 ? src.cols : 0;
            std::size_t jn = j > 0 ? -1 : 0;
            std::size_t jp = j < src.cols - 1 ? 1 : 0;

            double gradY = 0.0 +
                *(srcData + in + jn) + *(srcData + in) + *(srcData + in) + *(srcData + in + jp) -
                *(srcData + ip + jn) - *(srcData + ip) - *(srcData + ip) - *(srcData + ip + jp);

            double gradX = 0.0 +
                *(srcData + in + jn) + *(srcData + jn) + *(srcData + jn) + *(srcData + ip + jn) -
                *(srcData + in + jp) - *(srcData + jp) - *(srcData + jp) - *(srcData + ip + jp);

            *dstData = DIP::clamp(*srcData - std::sqrt(gradX * gradX + gradY * gradY));
        });
}

//拉普拉斯算子锐化
void laplacianSharpening(cv::Mat& src, cv::Mat& dst)
{
    DIP::changEachPixel1To1(src, dst,
        [&](int i, int j, std::uint8_t* srcData, std::uint8_t* dstData)
        {
            std::size_t in = i > 0 ? -src.cols : 0;
            std::size_t ip = i < src.rows - 1 ? src.cols : 0;
            std::size_t jn = j > 0 ? -1 : 0;
            std::size_t jp = j < src.cols - 1 ? 1 : 0;

            *dstData = DIP::clamp(
                5 * (*srcData) +
                -(*(srcData + in)) +
                -(*(srcData + jn)) +
                -(*(srcData + jp)) +
                -(*(srcData + ip)));
        });
}

int main()
{
    cv::Mat src = cv::imread("F:/Temp/Anime4K/p1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat dst;
    cv::Mat ref;
    cv::Mat ref2;

    meanFilter3x3(src, dst);
    cv::blur(src, ref, { 3,3 });
    cv::blur(src, ref2, { 5,5 });
    cv::imshow("原图", src);
    cv::imshow("3x3均值滤波", dst);
    cv::imshow("OpenCV自带均值滤波实现", ref);
    cv::imshow("5x5均值滤波", ref2);
    cv::waitKey();

    medianFilter3x3(src, dst);
    cv::medianBlur(src, ref, 3);
    cv::medianBlur(src, ref2, 5);
    cv::imshow("原图", src);
    cv::imshow("3x3中值滤波", dst);
    cv::imshow("OpenCV自带中值滤波实现", ref);
    cv::imshow("5x5中值滤波", ref2);
    cv::waitKey();

    maxFilter3x3(src, dst);
    cv::imshow("原图", src);
    cv::imshow("最大值滤波", dst);
    cv::waitKey();

    minFilter3x3(src, dst);
    cv::imshow("原图", src);
    cv::imshow("最小值滤波", dst);
    cv::waitKey();

    sobelFilter(src, dst);
    cv::imshow("原图", src);
    cv::imshow("sobel滤波", dst);
    cv::waitKey();

    sobelSharpening(src, dst);
    cv::imshow("原图", src);
    cv::imshow("sobel锐化", dst);
    cv::waitKey();

    laplacianSharpening(src, dst);
    cv::imshow("原图", src);
    cv::imshow("laplacian锐化", dst);
    cv::waitKey();
    return 0;
}
