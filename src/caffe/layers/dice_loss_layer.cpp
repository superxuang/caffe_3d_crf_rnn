#include <algorithm>
#include <vector>

#include "caffe/layers/dice_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <itkImage.h>
#include <itkSmartPointer.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

using std::floor;
using std::max;
using std::min;
using std::pow;

namespace caffe {

template <typename Dtype>
void DiceLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();

  const int num = bottom[0]->num();
  const int channel = bottom[0]->channels();
  const int count = bottom[0]->count();
  const int cls_num = channel - 1;
  const int pixel_num = count / num / channel;

  Dtype* result_buffer = new Dtype[num * channel * pixel_num];
  for (int i = 0; i < num; ++i) {
    for (int k = 0; k < pixel_num; ++k) {
      double max_score = -1;
      int max_score_cls = 0;
      for (int m = 0; m < channel; ++m) {
        if (max_score < bottom_data[i * channel * pixel_num + m * pixel_num + k])
        {
          max_score = bottom_data[i * channel * pixel_num + m * pixel_num + k];
          max_score_cls = m;
        }
      }
      for (int j = 0; j < channel; ++j) {
        result_buffer[i * channel * pixel_num + j * pixel_num + k] = (max_score_cls == j) ? 1.0 : 0.0;
      }
    }
  }
  delete[]union_;
  delete[]intersection_;
  delete[]class_exist_;
  union_ = new double[num * channel];
  intersection_ = new double[num * channel];
  class_exist_ = new bool[num * channel];
  int exist_num = 0;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < channel; ++j) {
      double result_sum = 0;
      double label_sum = 0;
      union_[i * channel + j] = 0;
      intersection_[i * channel + j] = 0;
      class_exist_[i * channel + j] = false;
      for (int k = 0; k < pixel_num; ++k) {
        double result_value = result_buffer[i * channel * pixel_num + j * pixel_num + k];
        double label_value = (label[i * pixel_num + k] == j) ? 1 : 0;
        
        union_[i * channel + j] += pow(result_value, 2.0) + pow(label_value, 2.0);
        
        intersection_[i * channel + j] += result_value * label_value;
		

        if (j > 0) {
          result_sum += result_value;
          label_sum += label_value;
        }
      }
      union_[i * channel + j] += 0.00001;
      if (label_sum > 0) {
        class_exist_[i * channel + j] = true;
        exist_num++;
      }
    }
  }
  
  Dtype* loss = top[0]->mutable_cpu_data();
  loss[0] = 0;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < channel; ++j) {
      if (class_exist_[i * channel + j]) {
        loss[0] += 2 * intersection_[i * channel + j] / union_[i * channel + j];
      }
    }
  }
  if (exist_num > 0) {
    loss[0] = loss[0] / exist_num;
  }
  LOG(INFO) << "Average dice = " << loss[0];

  delete[]result_buffer;
}

template <typename Dtype>
void DiceLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* label = bottom[1]->cpu_data();
    const int num = bottom[0]->num();
    const int channel = bottom[0]->channels();
    const int count = bottom[0]->count();
    const int cls_num = channel - 1;
    const int pixel_num = count / num / channel;

    memset(bottom_diff, 0, sizeof(Dtype) * num * channel * pixel_num);
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < pixel_num; ++j) {
        for (int k = 1; k < channel; ++k) {
          if (class_exist_[i * channel + k]) {
            double result_value = bottom_data[i * channel * pixel_num + k * pixel_num + j];
            double label_value = (label[i * pixel_num + j] == k) ? 1 : 0;
            double union_value = union_[i * channel + k];
            double intersection_value = intersection_[i * channel + k];
            double diff = 
              2 * (label_value * union_value / (union_value * union_value) -
              2 * result_value * intersection_value / (union_value * union_value));
            bottom_diff[i * channel * pixel_num + k * pixel_num + j] -= diff;              
            bottom_diff[i * channel * pixel_num + 0 * pixel_num + j] += diff;
          }
        }
      }
    }
  }
}

INSTANTIATE_CLASS(DiceLossLayer);
REGISTER_LAYER_CLASS(DiceLoss);

}  // namespace caffe
