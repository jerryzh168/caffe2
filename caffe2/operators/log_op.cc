#include <cmath>

#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

struct LogCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* device_context) {
    std::transform(x, x + n, y, log);
  }
};

namespace {
REGISTER_CPU_OPERATOR(
    Log,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, LogCPUFunctor>);

OPERATOR_SCHEMA(Log)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Calculates the natural log of the given input tensor, element-wise. This
operation can be done in an in-place fashion too, by providing the same input
and output blobs.
)DOC")
    .Input(0, "input", "Input tensor")
    .Output(
        0,
        "output",
        "The natural log of the input tensor computed "
        "element-wise");

class GetLogGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Div",
        "",
        std::vector<string>{GO(0), I(0)},
        std::vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Log, GetLogGradient);
} // namespace
} // namespace caffe2
