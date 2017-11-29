#ifndef CAFFE_DICE_LOSS_LAYER_HPP_
#define CAFFE_DICE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class DiceLossLayer : public LossLayer<Dtype> {
 public:
  explicit DiceLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {
    union_ = NULL;
    intersection_ = NULL;
    class_exist_ = NULL;
  }

  virtual inline const char* type() const { return "DiceLoss"; }

 protected:
  /// @copydoc DiceLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  double* union_;
  double* intersection_;
  bool* class_exist_;
};

}  // namespace caffe

#endif  // CAFFE_DICE_LOSS_LAYER_HPP_
