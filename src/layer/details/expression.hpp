#ifndef NO_INFER_SRC_LAYER_DETAILS_EXPRESSION_HPP
#define NO_INFER_SRC_LAYER_DETAILS_EXPRESSION_HPP

#include "layer/abstract/non_param_layer.hpp"
#include "parser/parse_expression.hpp"

namespace no_infer {

class ExpressionLayer : public NonParamLayer {
 public:
  explicit ExpressionLayer(std::string statement);

  InferStatus Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static ParseParameterAttrStatus GetInstance(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& expression_layer);

 private:
  std::string statement_;
  std::unique_ptr<ExpressionParser> parser_;
};

}  // namespace no_infer

#endif  // NO_INFER_SRC_LAYER_DETAILS_EXPRESSION_HPP
