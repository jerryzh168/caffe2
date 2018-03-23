#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/Nodes.h"
#include "caffe2/mobile/contrib/arm-compute/core/context.h"
#include "caffe2/mobile/contrib/arm-compute/core/operator.h"

#include "caffe2/operators/conv_op.h"

namespace caffe2 {

template <typename T>
class GLConvOp final : public ConvPoolOpBase<GLContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(GLContext);
  GLConvOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<GLContext>(operator_def, ws) {
    // Since this is the default convolution implementation, we will
    // use CAFFE_ENFORCE instead of OPERATOR_NEEDS_FEATURE.
    CAFFE_ENFORCE(
        group_ == 1 || order_ == StorageOrder::NCHW,
        "Group convolution only supports NCHW order right now.");
  }
  ~GLConvOp() {}

  bool RunOnDevice() override;
private:
  arm_compute::GCDirectConvolutionLayer conv_;
  bool first_run_ = true, second_run_ = true;
  GLContext::deleted_unique_ptr<const GLTensor<T>> X_, filter_, bias_;
};

template <typename T>
bool GLConvOp<T>::RunOnDevice() {
  auto *Xblob = OperatorBase::Inputs()[0];
  auto *filterblob = OperatorBase::Inputs()[1];
  auto *biasblob = OperatorBase::Inputs()[2];
  X_ = GLContext::getGLTensor<T>(Xblob, X_.release());
  if (first_run_) {
    filter_ = GLContext::getGLTensor<T>(filterblob);
    bias_ = GLContext::getGLTensor<T>(biasblob);
  }

  GLTensor<T> *Y =
    OperatorBase::Outputs()[0]->template GetMutable<GLTensor<T>>();

  const int N = X_->dim32(0), H = X_->dim32(2), W = X_->dim32(3), C = X_->dim32(1);

  CAFFE_ENFORCE_EQ(kernel_.size(), 2,
                   "Only 2d convolution is supported with ARM compute backend");

  CAFFE_ENFORCE(X_->ndim(), filter_->ndim());
  const int M = filter_->dim32(0);
  CAFFE_ENFORCE(filter_->dim32(2) == kernel_h());
  CAFFE_ENFORCE(filter_->dim32(3) == kernel_w());
  CAFFE_ENFORCE(filter_->dim32(1) == C);

  if (first_run_) {
    first_run_ = false;

    // resize output accordingly
    TensorCPU fakeX;
    fakeX.Resize(X_->dims());
    TensorCPU fakeY;
    ConvPoolOpBase<GLContext>::SetOutputSize(fakeX, &fakeY, filter_->dim32(0));
    Y->ResizeLike(fakeY);
    LOG(INFO) << "[C2DEBUG] dims of X " << X_->dims();
    LOG(INFO) << "[C2DEBUG] dims of X(gctensor) "
      << X_->get_underlying()->info()->dimension(3) << " "
      << X_->get_underlying()->info()->dimension(2) << " "
      << X_->get_underlying()->info()->dimension(1) << " "
      << X_->get_underlying()->info()->dimension(0) << " "
    ;
    LOG(INFO) << "[C2DEBUG] dims of Y " << Y->dims();
    LOG(INFO) << "[C2DEBUG] dims of Y(gctensor) "
      << Y->get_underlying()->info()->dimension(3) << " "
      << Y->get_underlying()->info()->dimension(2) << " "
      << Y->get_underlying()->info()->dimension(1) << " "
      << Y->get_underlying()->info()->dimension(0) << " "
    ;

    conv_.configure(
        X_->get_underlying(), filter_->get_underlying(), bias_->get_underlying(),
        Y->get_underlying(),
        arm_compute::PadStrideInfo(stride_[0], stride_[1], pads_[0], pads_[1]));

  } else if (second_run_) {
    // Always attempt to copy the CPU to GPU on input
    LOG(ERROR) << "[C2DEBUG] second pass+ X_:" << X_->dims();
    X_->lazy_allocate(Xblob, second_run_, true);
    filter_->lazy_allocate(filterblob, second_run_, second_run_);
    bias_->lazy_allocate(biasblob, second_run_, second_run_);
    second_run_ = false;
    Y->allocate();
    conv_.run();
  } else {
    LOG(ERROR) << "[C2DEBUG] third pass+ X_:" << X_->dims();
    // hack
    X_->lazy_allocate(Xblob, second_run_, true);
    // LOG(ERROR) << "[C2DEBUG] 1";
    // arm_compute::TensorShape shape;
    // for (int i = 0; i < X_->dims().size(); i++) {
    //   shape.set(X_->dims().size() - i - 1, X_->dims()[i]);
    // }
    // LOG(ERROR) << "[C2DEBUG] 2";
    // X_->get_underlying()->info()->set_tensor_shape(shape);
    TensorCPU fakeX;
    LOG(ERROR) << "[C2DEBUG] Conv 3";
    fakeX.Resize(X_->dims());
    TensorCPU fakeY;
    LOG(ERROR) << "[C2DEBUG] Conv 4";
    ConvPoolOpBase<GLContext>::SetOutputSize(fakeX, &fakeY, filter_->dim32(0));
    LOG(ERROR) << "[C2DEBUG] before resize";
    Y->ResizeLike(fakeY);
    LOG(ERROR) << "[C2DEBUG] X dims " << X_->dims();
    LOG(ERROR) << "[C2DEBUG] Y dims " << Y->dims();
    LOG(ERROR) << "[C2DEBUG] conv reconfigure N:" << X_->dims()[0];
    conv_.configure(
                    X_->get_underlying(), filter_->get_underlying(), bias_->get_underlying(),
                    Y->get_underlying(),
                    arm_compute::PadStrideInfo(stride_[0], stride_[1], pads_[0], pads_[1]));
    LOG(ERROR) << "[C2DEBUG] after re-configure";
    conv_.run();
    LOG(ERROR) << "[C2DEBUG] after run";
 }

  return true;
}

REGISTER_GL_OPERATOR(Conv, GLConvOp<DataType>);

} // namespace caffe2
