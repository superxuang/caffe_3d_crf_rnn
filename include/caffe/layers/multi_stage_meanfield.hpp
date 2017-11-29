#ifndef CAFFE_MULTI_STAGE_MEANFIELD_LAYER_HPP_
#define CAFFE_MULTI_STAGE_MEANFIELD_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/modified_permutohedral.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/split_layer.hpp"
#include "caffe/layers/meanfield_iteration.hpp"
#include <boost/shared_array.hpp>

namespace caffe {

/*!
 *  \brief     The Caffe layer that implements the CRF-RNN described in the paper:
 *             Conditional Random Fields as Recurrent Neural Networks. IEEE ICCV 2015.
 *
 *  \authors   Sadeep Jayasumana, Bernardino Romera-Paredes, Shuai Zheng, Zhizhong Su.
 *  \version   1.0
 *  \date      2015
 *  \copyright Torr Vision Group, University of Oxford.
 *  \details   If you use this code, please consider citing the paper:
 *             Shuai Zheng, Sadeep Jayasumana, Bernardino Romera-Paredes, Vibhav Vineet, Zhizhong Su, Dalong Du,
 *             Chang Huang, Philip H. S. Torr. Conditional Random Fields as Recurrent Neural Networks. IEEE ICCV 2015.
 *
 *             For more information about CRF-RNN, please visit the project website http://crfasrnn.torr.vision.
 */
template <typename Dtype>
class MultiStageMeanfieldLayer : public Layer<Dtype> {

 public:
  explicit MultiStageMeanfieldLayer(const LayerParameter& param) : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {
    return "MultiStageMeanfield";
  }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void compute_spatial_kernel(float* const output_kernel);
  virtual void compute_bilateral_kernel(const Blob<Dtype>* const image_data_start, const int n, float* const output_kernel);

  int count_;
  int num_;
  int channels_;
  int length_;
  int height_;
  int width_;
  int num_pixels_;

  Dtype theta_alpha_;
  Dtype theta_beta_;
  Dtype theta_gamma_;
  int num_iterations_;

  boost::shared_array<Dtype> norm_feed_;
  Blob<Dtype> spatial_norm_;
  Blob<Dtype> bilateral_norms_;

  vector<Blob<Dtype>*> split_layer_bottom_vec_;
  vector<Blob<Dtype>*> split_layer_top_vec_;
  vector<shared_ptr<Blob<Dtype> > > split_layer_out_blobs_;
  vector<shared_ptr<Blob<Dtype> > > iteration_output_blobs_;
  vector<shared_ptr<MeanfieldIteration<Dtype> > > meanfield_iterations_;

  shared_ptr<SplitLayer<Dtype> > split_layer_;

  shared_ptr<ModifiedPermutohedral> spatial_lattice_;
  boost::shared_array<float> bilateral_kernel_buffer_;
  vector<shared_ptr<ModifiedPermutohedral> > bilateral_lattices_;
};

}  // namespace caffe

#endif  // CAFFE_MULTI_STAGE_MEANFIELD_LAYER_HPP_
