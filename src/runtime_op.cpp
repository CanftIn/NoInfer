#include "runtime/runtime_op.hpp"

#include "data/tensor_util.hpp"

namespace no_infer {

RuntimeOperator::~RuntimeOperator() {
  for (auto& [_, param] : this->params) {
    if (param != nullptr) {
      delete param;
      param = nullptr;
    }
  }
}

}  // namespace no_infer