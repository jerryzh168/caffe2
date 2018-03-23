#include "caffe2/mobile/contrib/arm-compute/core/context.h"
#include "caffe2/mobile/contrib/arm-compute/core/operator.h"

#include "caffe2/mobile/contrib/arm-compute/operators/activation_ops.h"
#include "caffe2/operators/relu_op.h"

namespace caffe2 {

template <typename T>
bool GLReluOp<T>::RunOnDevice() {

  auto *Xblob = OperatorBase::Inputs()[0];
  if (first_run_) {
    X_ = GLContext::getGLTensor<T>(Xblob);
  }

  GLTensor<T> *Y =
      OperatorBase::Outputs()[0]->template GetMutable<GLTensor<T>>();

  if (first_run_) {
    first_run_ = false;
    if (Y->get_underlying() != X_->get_underlying())
    {
        Y->ResizeLike(*X_);
    }
    relu_layer_.configure(
        X_->get_underlying(), Y->get_underlying(),
        arm_compute::ActivationLayerInfo(
          arm_compute::ActivationLayerInfo::ActivationFunction::RELU));

  } else if (second_run_) {
    X_->lazy_allocate(Xblob, second_run_, true);
    second_run_ = false;
    // in place activation, do not need to allocate new memory
    if (Y->get_underlying() != X_->get_underlying()) {
        Y->allocate();
    }
    relu_layer_.run();
  } else {
    X_->lazy_allocate(Xblob, second_run_, true);
    arm_compute::TensorShape shape;
    LOG(ERROR) << "[C2DEBUG] relu Xdims: " << X_->dims();
    for (int i = 0; i < X_->dims().size(); i++) {
      shape.set(X_->dims().size() - i - 1, X_->dims()[i]);
    }
    X_->get_underlying()->info()->set_tensor_shape(shape);
    if (Y->get_underlying() != X_->get_underlying())
    {
        Y->ResizeLike(*X_);
    }
    relu_layer_.configure(
        X_->get_underlying(), Y->get_underlying(),
        arm_compute::ActivationLayerInfo(
          arm_compute::ActivationLayerInfo::ActivationFunction::RELU));
    relu_layer_.run();
  }

  return true;
}

REGISTER_GL_OPERATOR(Relu, GLReluOp<half>);

template <typename T>
bool GLSigmoidOp<T>::RunOnDevice() {

  auto *Xblob = OperatorBase::Inputs()[0];
  if (first_run_) {
    X_ = GLContext::getGLTensor<T>(Xblob);
  }

  GLTensor<T> *Y =
      OperatorBase::Outputs()[0]->template GetMutable<GLTensor<T>>();
  if (first_run_) {
    first_run_ = false;

    if (Y->get_underlying() != X_->get_underlying())
    {
        Y->ResizeLike(*X_);
    }

    sigmoid_layer_.configure(
      X_->get_underlying(), Y->get_underlying(),
      arm_compute::ActivationLayerInfo(
          arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC));
  } else if (second_run_) {
    X_->lazy_allocate(Xblob, second_run_, true);
    second_run_ = false;
    // in place activation, do not need to allocate new memory
    if (Y->get_underlying() != X_->get_underlying()) {
      Y->allocate();
    }
    sigmoid_layer_.run();
  } else {
    X_->lazy_allocate(Xblob, second_run_, true);
    if (Y->get_underlying() != X_->get_underlying())
    {
        Y->ResizeLike(*X_);
    }
    sigmoid_layer_.configure(
      X_->get_underlying(), Y->get_underlying(),
      arm_compute::ActivationLayerInfo(
          arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC));
    sigmoid_layer_.run();
  }

  return true;
}

REGISTER_GL_OPERATOR(Sigmoid, GLSigmoidOp<DataType>);

} // namespace caffe2
