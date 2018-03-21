#include "caffe2/mobile/contrib/arm-compute/core/context.h"
#include "caffe2/mobile/contrib/arm-compute/core/operator.h"
#include "caffe2/core/timer.h"

namespace caffe2 {

template <typename T> class CopyFromGLOp final : public Operator<GLContext> {
public:
  CopyFromGLOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<GLContext>(operator_def, ws) {}
  virtual ~CopyFromGLOp() noexcept {}
  USE_OPERATOR_FUNCTIONS(GLContext);
  bool RunOnDevice() override;
private:
  bool first_run_ = true, second_run_ = true;
  std::vector<GLContext::deleted_unique_ptr<const GLTensor<T>>> inputs_;
};

template <typename T>
bool CopyFromGLOp<T>::RunOnDevice() {

  auto *X0blob = OperatorBase::Inputs()[0];
  auto X0 = GLContext::getGLTensor<T>(X0blob);

  std::vector<const Blob*> inputsBlob;
  inputsBlob.push_back(X0blob);

  for (int i = 1; i < Inputs().size(); ++i) {
    auto *Xblob = OperatorBase::Inputs()[i];
    inputsBlob.push_back(Xblob);
  }

  if (first_run_) {
    LOG(ERROR) << "[C2DEBUG] -- first run CopyFromGL";
    for (int i = 0; i < Inputs().size(); ++i) {
      auto *Xblob = inputsBlob[i];
      auto X = GLContext::getGLTensor<T>(Xblob);
      inputs_.push_back(std::move(X));
    }
  }

  if (first_run_) {
    first_run_ = false;
    for (int i = 0; i < Inputs().size(); ++i) {
      auto* Y = OperatorBase::Outputs()[i]->template GetMutable<TensorCPU>();
      Y->Resize(inputs_[i]->dims());
      Y->template mutable_data<float>();
    }
  } else {
    for (int i = 0; i < inputs_.size(); ++i) {
      auto* X = inputs_[i].get();
      auto* Xblob = inputsBlob[i];
      X->lazy_allocate(Xblob, second_run_, second_run_);
    }
    if (second_run_) {
      // Don't need to allocate output
      second_run_ = false;
    }
    for (auto i = 0; i < Inputs().size(); ++i) {
      // Blob
      auto* Xblob = inputsBlob[i];
      // GLTensor
      auto* X = inputs_[i].get();
      Output(i)->Resize(X->dims());
      Output(i)->template mutable_data<float>();
      // hardcoding CPU Tensors to be float
      CAFFE_ENFORCE_EQ(Output(i)->nbytes(), X->size() * sizeof(float));
      Timer timer;
      timer.Start();
      getTensorCPU(*X, *(OperatorBase::Outputs()[i]->template GetMutable<TensorCPU>()));
      auto millis = timer.MilliSeconds();
      //VLOG(2) << "[C2DEBUG] copy_op takes " << millis << " milliseconds";
    }
  }

  return true;
}

REGISTER_GL_OPERATOR(CopyFromGL, CopyFromGLOp<DataType>);

} // namespace caffe2
