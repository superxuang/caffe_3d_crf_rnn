#ifndef CAFFE_SEGMENT_OUTPUT_LAYER_HPP_
#define CAFFE_SEGMENT_OUTPUT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <itkImage.h>
#include <itkSmartPointer.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkIdentityTransform.h>
#include <itkResampleImageFilter.h>
#include <itkNearestNeighborInterpolateImageFunction.h>

namespace caffe {

template <typename Dtype>
class SegmentOutputLayer : public Layer<Dtype> {
 public:
  typedef itk::Image<Dtype, 3> ImageType;
  typedef itk::Image<char, 3> LabelType;
  typedef itk::ImageFileReader<ImageType> ImageReaderType;
  typedef itk::ImageFileReader<LabelType> LabelReaderType;
  typedef itk::ImageFileWriter<ImageType> ImageWriterType;
  typedef itk::ImageFileWriter<LabelType> LabelWriterType;
  typedef itk::IdentityTransform<double, 3> IdentityTransformType;
  typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
  typedef itk::ResampleImageFilter<LabelType, LabelType> ResampleLabelFilterType;
  typedef itk::NearestNeighborInterpolateImageFunction<LabelType, double> InterpolatorType;

  explicit SegmentOutputLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
	  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 1;
  }

  virtual inline const char* type() const { return "SegmentOutput"; }

 protected:
  /// @copydoc SegmentOutputLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  vector<std::pair<std::string, std::pair<itk::SmartPointer<ImageType>, itk::SmartPointer<LabelType>>>> lines_;
  int lines_id_;
  double* heat_intersection_;
  double* heat_union_;
  double* mask_intersection_;
  double* mask_union_;
  double* voe_a_;
  double* voe_b_;
  double* rvd_a_;
  double* rvd_b_;
  double* asd_;
  double* asd_num_;
  double* msd_;
  double* msd_num_;
  double bin_intersection_;
  double bin_union_;
  LabelType::Pointer output_mask_;
  LabelType::Pointer gt_;
  std::vector<itk::SmartPointer<ImageType>> output_heatmap_;
};

}  // namespace caffe

#endif  // CAFFE_SEGMENT_OUTPUT_LAYER_HPP_
