#ifndef NO_INFER_DATA_TENSOR_UTIL_HPP
#define NO_INFER_DATA_TENSOR_UTIL_HPP

#include "data/tensor.hpp"

namespace no_infer {
/**
 * 对张量进行形状上的扩展
 * 张量（tensor）的广播操作，即将两个不同形状的张量调整为兼容的形状以便进行元素级操作。
 * 具体来说，代码处理了以下几种情况：
 * - 相同形状的张量：如果两个张量的形状相同，直接返回这两个张量。
 * - 不同形状但有相同通道数的张量：对两个张量进行适当的广播，使它们的形状匹配。
 * @param tenor1 张量1
 * @param tensor2 张量2
 * @return 形状一致的张量
 */
std::tuple<sftensor, sftensor> TensorBroadcast(const sftensor& tensor1,
                                               const sftensor& tensor2);

/**
 * 对张量的填充
 * @param tensor 待填充的张量
 * @param pads 填充的大小
 * @param padding_value 填充的值
 * @return 填充之后的张量
 */
std::shared_ptr<Tensor<float>> TensorPadding(
    const std::shared_ptr<Tensor<float>>& tensor,
    const std::vector<uint32_t>& pads, float padding_value);

/**
 * 比较tensor的值是否相同
 * @param a 输入张量1
 * @param b 输入张量2
 * @param threshold 张量之间差距的阈值
 * @return 比较结果
 */
bool TensorIsSame(const std::shared_ptr<Tensor<float>>& a,
                  const std::shared_ptr<Tensor<float>>& b,
                  float threshold = 1e-5f);

/**
 * 张量相加
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 * @return 张量相加的结果
 */
std::shared_ptr<Tensor<float>> TensorElementAdd(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2);

/**
 * 张量相加
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 * @param output_tensor 输出张量
 */
void TensorElementAdd(const std::shared_ptr<Tensor<float>>& tensor1,
                      const std::shared_ptr<Tensor<float>>& tensor2,
                      const std::shared_ptr<Tensor<float>>& output_tensor);

/**
 * 矩阵点乘
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 * @param output_tensor 输出张量
 */
void TensorElementMultiply(const std::shared_ptr<Tensor<float>>& tensor1,
                           const std::shared_ptr<Tensor<float>>& tensor2,
                           const std::shared_ptr<Tensor<float>>& output_tensor);

/**
 * 张量相乘
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 * @return 张量相乘的结果
 */
std::shared_ptr<Tensor<float>> TensorElementMultiply(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2);

/**
 * 创建一个张量
 * @param channels 通道数量
 * @param rows 行数
 * @param cols 列数
 * @return 创建后的张量
 */
std::shared_ptr<Tensor<float>> TensorCreate(uint32_t channels, uint32_t rows,
                                            uint32_t cols);

std::shared_ptr<Tensor<float>> TensorCreate(uint32_t rows, uint32_t cols);

std::shared_ptr<Tensor<float>> TensorCreate(uint32_t size);

/**
 * 创建一个张量
 * @param shapes 张量的形状
 * @return 创建后的张量
 */
std::shared_ptr<Tensor<float>> TensorCreate(
    const std::vector<uint32_t>& shapes);

/**
 * 返回一个深拷贝后的张量
 * @param 待Clone的张量
 * @return 新的张量
 */
std::shared_ptr<Tensor<float>> TensorClone(
    std::shared_ptr<Tensor<float>> tensor);

}  // namespace no_infer

#endif  // NO_INFER_DATA_TENSOR_UTIL_HPP