#ifndef NO_INFER_RUNTIME_OPERAND_HPP
#define NO_INFER_RUNTIME_OPERAND_HPP

#include <memory>
#include <string>
#include <vector>

#include "data/tensor.hpp"
#include "runtime_datatype.hpp"
#include "status_code.hpp"

namespace no_infer {

/// 计算节点输入输出的操作数
struct RuntimeOperand {
  std::string name;                                   /// 操作数的名称
  std::vector<int32_t> shapes;                        /// 操作数的形状
  std::vector<std::shared_ptr<Tensor<float>>> datas;  /// 存储操作数，多个张量存在vector中，也就是batch。
  RuntimeDataType type =
      RuntimeDataType::kTypeUnknown;  /// 操作数的类型，一般是float
};

}  // namespace no_infer

#endif  // NO_INFER_RUNTIME_OPERAND_HPP