#ifndef NO_INFER_SRC_LAYER_DETAILS_LINEAR_HPP
#define NO_INFER_SRC_LAYER_DETAILS_LINEAR_HPP

#include "layer/abstract/layer.hpp"
#include "layer/abstract/param_layer.hpp"

namespace no_infer {

class LinearLayer : public ParamLayer {
 public:
  //  explicit LinearLayer(uint32_t batch, uint32_t in_channel, uint32_t in_dim,
  //  uint32_t out_dim, bool use_bias = true);
  explicit LinearLayer(int32_t in_features, int32_t out_features,
                       bool use_bias);

  InferStatus Forward(
      const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
      std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

  static ParseParameterAttrStatus GetInstance(
      const std::shared_ptr<RuntimeOperator> &op,
      std::shared_ptr<Layer> &linear_layer);

 private:
  int32_t in_features_ = 0;
  int32_t out_features_ = 0;
  bool use_bias_ = false;
};

}  // namespace no_infer

#endif  // NO_INFER_SRC_LAYER_DETAILS_LINEAR_HPP
