#include "caffe2/mobile/contrib/arm-compute/core/context.h"
#include "caffe2/mobile/contrib/arm-compute/core/operator.h"

namespace caffe2 {

template<typename T>
class GLRoIAlignOp final : public Operator<GLContext> {
public:
  GLRoIAlignOp(const OperatorDef &operator_def, Workspace *ws)
    : Operator<GLContext>(operator_def, ws), spatial_scale_(1.0), pooled_h_(1), pooled_w_(1), sampling_ratio_(-1) {
    if (HasArgument("spatial_scale")) {
      spatial_scale_ = static_cast<float>(
          OperatorBase::GetSingleArgument<float>("spatial_scale", 1.0));
    }
    if (HasArgument("pooled_h")) {
      pooled_h_ = static_cast<int>(
          OperatorBase::GetSingleArgument<int>("pooled_h", 1));
    }
    if (HasArgument("pooled_w")) {
      pooled_w_ = static_cast<int>(
          OperatorBase::GetSingleArgument<int>("pooled_w", 1));
    }
    if (HasArgument("sampling_ratio")) {
      sampling_ratio_ = static_cast<int>(
          OperatorBase::GetSingleArgument<int>("sampling_ratio", -1));
    }

  }
  virtual ~GLRoIAlignOp() noexcept {}
  USE_OPERATOR_FUNCTIONS(GLContext);
  bool RunOnDevice() override;
private:
  float spatial_scale_;
  int pooled_h_;
  int pooled_w_;
  int sampling_ratio_;
  arm_compute::GCROIAlign roi_align_;
  bool first_run_ = true, second_run_ = true;
  GLContext::deleted_unique_ptr<const GLTensor<T>> X_, rois_;
};

template <typename T>
bool GLRoIAlignOp<T>::RunOnDevice() {

  auto* Xblob = OperatorBase::Inputs()[0];
  auto* ROIsblob = OperatorBase::Inputs()[1];

  if (first_run_) {
    X_ = GLContext::getGLTensor<T>(Xblob);
    rois_ = GLContext::getGLTensor<T>(ROIsblob);
  }

  CAFFE_ENFORCE(X_->dim32(0), 1);
  auto C = X_->dim32(1);
  auto H = X_->dim32(2);
  auto W = X_->dim32(3);

  auto N = rois_->dim32(0);
  auto col = rois_->dim32(1);
  CAFFE_ENFORCE_EQ(col, 5);

  GLTensor<T> *Y =
      OperatorBase::Outputs()[0]->template GetMutable<GLTensor<T>>();
  vector<TIndex> output_dims = {N, C, pooled_h_, pooled_w_};

  if (first_run_) {
    Y->Resize(output_dims);
    first_run_ = false;
    roi_align_.configure(X_->get_underlying(), Y->get_underlying(), rois_->get_underlying(), spatial_scale_, pooled_h_, pooled_w_, sampling_ratio_);
  } else if (second_run_) {
    X_->lazy_allocate(Xblob, second_run_, true);
    rois_->lazy_allocate(ROIsblob, second_run_, true);
    second_run_ = false;
    Y->Resize(output_dims);
    Y->allocate();
    roi_align_.run();
  } else {
    X_->lazy_allocate(Xblob, second_run_, true);
    rois_->lazy_allocate(ROIsblob, second_run_, true);
    bool need_allocation = Y->Resize(output_dims);
    roi_align_.configure(X_->get_underlying(), Y->get_underlying(), rois_->get_underlying(), spatial_scale_, pooled_h_, pooled_w_, sampling_ratio_);
    if (need_allocation) {
      Y->allocate();
    }
  }

  return true;
}

REGISTER_GL_OPERATOR(RoIAlign, GLRoIAlignOp<DataType>);

} // namespace caffe2
