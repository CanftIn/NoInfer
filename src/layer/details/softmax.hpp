#ifndef NO_INFER_SRC_LAYER_DETAILS_SOFTMAX_HPP
#define NO_INFER_SRC_LAYER_DETAILS_SOFTMAX_HPP

#include "layer/abstract/non_param_layer.hpp"

namespace no_infer {
class SoftmaxLayer : public NonParamLayer {
 public:
  explicit SoftmaxLayer(int dim = -1);

  InferStatus Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static ParseParameterAttrStatus CreateInstance(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& softmax_layer);

 private:
  int softmax_dim_ = -1;
};
}  // namespace no_infer

#endif  // NO_INFER_SRC_LAYER_DETAILS_SOFTMAX_HPP
