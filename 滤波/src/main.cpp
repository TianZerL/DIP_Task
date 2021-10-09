#include <algorithm>

#include "DIP.hpp"

//3x3��ֵ�˲�
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

//��ֵ�˲�
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

//���ֵ�˲�
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

//��Сֵ�˲�
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

//sobel��Ե���
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

//sobel��
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

//������˹������
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
    cv::imshow("ԭͼ", src);
    cv::imshow("3x3��ֵ�˲�", dst);
    cv::imshow("OpenCV�Դ���ֵ�˲�ʵ��", ref);
    cv::imshow("5x5��ֵ�˲�", ref2);
    cv::waitKey();

    medianFilter3x3(src, dst);
    cv::medianBlur(src, ref, 3);
    cv::medianBlur(src, ref2, 5);
    cv::imshow("ԭͼ", src);
    cv::imshow("3x3��ֵ�˲�", dst);
    cv::imshow("OpenCV�Դ���ֵ�˲�ʵ��", ref);
    cv::imshow("5x5��ֵ�˲�", ref2);
    cv::waitKey();

    maxFilter3x3(src, dst);
    cv::imshow("ԭͼ", src);
    cv::imshow("���ֵ�˲�", dst);
    cv::waitKey();

    minFilter3x3(src, dst);
    cv::imshow("ԭͼ", src);
    cv::imshow("��Сֵ�˲�", dst);
    cv::waitKey();

    sobelFilter(src, dst);
    cv::imshow("ԭͼ", src);
    cv::imshow("sobel�˲�", dst);
    cv::waitKey();

    sobelSharpening(src, dst);
    cv::imshow("ԭͼ", src);
    cv::imshow("sobel��", dst);
    cv::waitKey();

    laplacianSharpening(src, dst);
    cv::imshow("ԭͼ", src);
    cv::imshow("laplacian��", dst);
    cv::waitKey();
    return 0;
}
