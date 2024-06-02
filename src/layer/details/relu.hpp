#ifndef NO_INFER_SRC_LAYER_DETAILS_RELU_HPP
#define NO_INFER_SRC_LAYER_DETAILS_RELU_HPP

#include "layer/abstract/non_param_layer.hpp"

namespace no_infer {

class ReluLayer : public NonParamLayer {
 public:
  ReluLayer() : NonParamLayer("Relu") {}
  InferStatus Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static ParseParameterAttrStatus GetInstance(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& relu_layer);
};

}  // namespace no_infer

#endif  // NO_INFER_SRC_LAYER_DETAILS_RELU_HPP
