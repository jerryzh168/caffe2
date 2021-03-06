#include <cmath>

#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

struct ExpCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* device_context) {
    std::transform(x, x + n, y, exp);
  }
};

namespace {
REGISTER_CPU_OPERATOR(
    Exp,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, ExpCPUFunctor>);

OPERATOR_SCHEMA(Exp)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Calculates the exponential of the given input tensor, element-wise. This
operation can be done in an in-place fashion too, by providing the same input
and output blobs.
)DOC")
    .Input(0, "input", "Input tensor")
    .Output(
        0,
        "output",
        "The exponential of the input tensor computed "
        "element-wise");

class GetExpGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Mul",
        "",
        std::vector<string>{O(0), GO(0)},
        std::vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Exp, GetExpGradient);
} // namespace
} // namespace caffe2
