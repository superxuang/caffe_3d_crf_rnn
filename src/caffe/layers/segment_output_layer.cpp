#include <algorithm>
#include <vector>

#include "caffe/layers/segment_output_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <itkMetaImageIOFactory.h>
#include <itkConnectedComponentImageFilter.h>
#include <itkContourDirectedMeanDistanceImageFilter.h>
#include <itkDirectedHausdorffDistanceImageFilter.h>

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
void SegmentOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const SegmentOutputParameter& segment_output_param = this->layer_param_.segment_output_param();
  const ContourNameList& contour_name_list = segment_output_param.contour_name_list();
  const int contour_num = contour_name_list.name_size();
  const string& source = segment_output_param.source();
  const string& root_folder = segment_output_param.root_folder();
  itk::ObjectFactoryBase::RegisterFactory(itk::MetaImageIOFactory::New());
  ImageType::SizeType image_size;

  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  size_t pos1, pos2;
  while (std::getline(infile, line)) {
    pos1 = line.find_first_of(' ');
    pos2 = line.find_last_of(' ');
    string image_file_name = line.substr(0, pos1);
    string label_file_name = line.substr(pos1 + 1, pos2 - pos1 - 1);
    string info_file_name = line.substr(pos2 + 1);
    std::ifstream infile_info(root_folder + info_file_name);
    std::vector<int> contour_labels;
	std::vector<int> exist_contours;
	int label_range[3][2];
	label_range[0][0] = INT_MAX;
	label_range[0][1] = INT_MIN;
	label_range[1][0] = INT_MAX;
	label_range[1][1] = INT_MIN;
	label_range[2][0] = INT_MAX;
	label_range[2][1] = INT_MIN;
	while (std::getline(infile_info, line)) {
      pos1 = string::npos;
      for (int i = 0; i < contour_name_list.name_size(); ++i) {
        pos1 = line.find(contour_name_list.name(i));
		if (pos1 != string::npos) {
          exist_contours.push_back(i + 1);
          break;
        }
      }
      if (pos1 == string::npos)
        continue;

      pos1 = line.find_first_of(' ', pos1);
      pos2 = line.find_first_of(' ', pos1 + 1);
      int label_value = atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str());

	  pos1 = pos2;
	  pos2 = line.find_first_of(' ', pos1 + 1);
	  label_range[0][0] = min(label_range[0][0], atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str()));

	  pos1 = pos2;
	  pos2 = line.find_first_of(' ', pos1 + 1);
	  label_range[0][1] = max(label_range[0][1], atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str()));

	  pos1 = pos2;
	  pos2 = line.find_first_of(' ', pos1 + 1);
	  label_range[1][0] = min(label_range[1][0], atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str()));

	  pos1 = pos2;
	  pos2 = line.find_first_of(' ', pos1 + 1);
	  label_range[1][1] = max(label_range[1][1], atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str()));

	  pos1 = pos2;
	  pos2 = line.find_first_of(' ', pos1 + 1);
	  label_range[2][0] = min(label_range[2][0], atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str()));

	  pos1 = pos2;
	  label_range[2][1] = max(label_range[2][1], atoi(line.substr(pos1 + 1).c_str()));

      contour_labels.push_back(label_value);
    }
    if (!contour_labels.empty()) {
      ImageType::DirectionType direct_src;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          direct_src[i][j] = (i == j) ? 1 : 0;
        }
      }

      ImageReaderType::Pointer reader_image = ImageReaderType::New();
      reader_image->SetFileName(root_folder + image_file_name);
      reader_image->Update();
      ImageType::Pointer image = reader_image->GetOutput();
      image->SetDirection(direct_src);

      LabelReaderType::Pointer reader_label = LabelReaderType::New();
      reader_label->SetFileName(root_folder + label_file_name);
      reader_label->Update();
      LabelType::Pointer label = reader_label->GetOutput();
      label->SetDirection(direct_src);

      int buffer_length = label->GetBufferedRegion().GetNumberOfPixels();
      char* label_buffer = label->GetBufferPointer();
      char* label_final_output_buffer = new char[buffer_length];
      memset(label_final_output_buffer, 0, sizeof(char) * buffer_length);
      for (int i = 0; i < buffer_length; ++i) {
        for (int j = 0; j < contour_labels.size(); ++j) {
          if (label_buffer[i] == contour_labels[j]) {
            label_final_output_buffer[i] = exist_contours[j];
            break;
          }
        }
      }
      memcpy(label_buffer, label_final_output_buffer, sizeof(char) * buffer_length);
      delete[]label_final_output_buffer;

      // resample
	  if (segment_output_param.resample_volume()) {
        ImageType::SizeType size_src = image->GetBufferedRegion().GetSize();
        ImageType::SpacingType spacing_src = image->GetSpacing();
        ImageType::PointType origin_src = image->GetOrigin();

        ImageType::SizeType size_resample;
		size_resample[0] = segment_output_param.resample_volume_width();
		size_resample[1] = segment_output_param.resample_volume_height();
		size_resample[2] = segment_output_param.resample_volume_length();
        ImageType::SpacingType spacing_resample;
		spacing_resample[0] = segment_output_param.resample_spacing_x();
		spacing_resample[1] = segment_output_param.resample_spacing_y();
        spacing_resample[2] = segment_output_param.resample_spacing_z();
        ImageType::PointType origin_resample;
		if (segment_output_param.center_align_label()) {
		  origin_resample[0] = origin_src[0] + ((label_range[0][0] + label_range[0][1]) * spacing_src[0] - size_resample[0] * spacing_resample[0]) * 0.5;
		  origin_resample[1] = origin_src[1] + ((label_range[1][0] + label_range[1][1]) * spacing_src[1] - size_resample[1] * spacing_resample[1]) * 0.5;
		  origin_resample[2] = origin_src[2] + ((label_range[2][0] + label_range[2][1]) * spacing_src[2] - size_resample[2] * spacing_resample[2]) * 0.5;
		}
		else {
		  origin_resample[0] = origin_src[0] + 0.5 * (size_src[0] * spacing_src[0] - size_resample[0] * spacing_resample[0]);
		  origin_resample[1] = origin_src[1] + 0.5 * (size_src[1] * spacing_src[1] - size_resample[1] * spacing_resample[1]);
		  origin_resample[2] = origin_src[2] + 0.5 * (size_src[2] * spacing_src[2] - size_resample[2] * spacing_resample[2]);
		}

        ResampleImageFilterType::Pointer resampler_image = ResampleImageFilterType::New();
        resampler_image->SetInput(image);
        resampler_image->SetSize(size_resample);
        resampler_image->SetOutputSpacing(spacing_resample);
        resampler_image->SetOutputOrigin(origin_resample);
        resampler_image->SetTransform(IdentityTransformType::New());
        resampler_image->Update();
        image = resampler_image->GetOutput();

        ResampleLabelFilterType::Pointer resampler_label = ResampleLabelFilterType::New();
        resampler_label->SetInput(label);
        resampler_label->SetSize(size_resample);
        resampler_label->SetOutputSpacing(spacing_resample);
        resampler_label->SetOutputOrigin(origin_resample);
        resampler_label->SetTransform(IdentityTransformType::New());
        resampler_label->SetInterpolator(InterpolatorType::New());
        resampler_label->Update();
        label = resampler_label->GetOutput();
      }

      lines_.push_back(
        std::make_pair(image_file_name,
        std::make_pair(image, label)));
    }
  }

  CHECK(!lines_.empty()) << "File is empty";

  LOG(INFO) << "A total of " << lines_.size() << " slices.";

  lines_id_ = 0;
  bin_intersection_ = 0;
  bin_union_ = 0;
  heat_intersection_ = new double[contour_num + 1];
  heat_union_ = new double[contour_num + 1];
  mask_intersection_ = new double[contour_num + 1];
  mask_union_ = new double[contour_num + 1];
  voe_a_ = new double[contour_num + 1];
  voe_b_ = new double[contour_num + 1];
  rvd_a_ = new double[contour_num + 1];
  rvd_b_ = new double[contour_num + 1];
  asd_ = new double[contour_num + 1];
  asd_num_ = new double[contour_num + 1];
  msd_ = new double[contour_num + 1];
  msd_num_ = new double[contour_num + 1];
  output_heatmap_ = std::vector<itk::SmartPointer<ImageType>>(contour_num + 1);
  for (int i = 0; i < contour_num + 1; ++i) {
    heat_intersection_[i] = 0;
    heat_union_[i] = 0;
    mask_intersection_[i] = 0;
    mask_union_[i] = 0;
	voe_a_[i] = 0;
	voe_b_[i] = 0;
	rvd_a_[i] = 0;
	rvd_b_[i] = 0;
	asd_[i] = 0;
	asd_num_[i] = 0;
	msd_[i] = 0;
	msd_num_[i] = 0;
    output_heatmap_[i] = 0;
  }

  itk::ObjectFactoryBase::RegisterFactory(itk::MetaImageIOFactory::New());
}

template <typename Dtype>
void SegmentOutputLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(0);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void SegmentOutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* score = bottom[0]->mutable_cpu_data();
  Dtype* label = bottom[1]->mutable_cpu_data();

  const int num = bottom[0]->num();
  const int channel = bottom[0]->channels();
  const int count = bottom[0]->count();
  const int dim = count / num;
  const int pixel_num = dim / channel;
  const SegmentOutputParameter& segment_output_param = this->layer_param_.segment_output_param();
  const ContourNameList& contour_name_list = segment_output_param.contour_name_list();
  const int contour_num = contour_name_list.name_size();
  const string& root_folder = segment_output_param.root_folder();
  const string& output_folder = segment_output_param.output_folder();

  Dtype* loss = top[0]->mutable_cpu_data();
  loss[0] = 0;

  typedef itk::IdentityTransform<double, 3> IdentityTransformType;
  typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
  ImageType::SizeType size_btm, size_out;
  ImageType::SpacingType spacing_btm, spacing_out;
  ImageType::PointType origin_btm, origin_out;
  ImageType::DirectionType direct_btm;
  ImageType::IndexType index_btm;
  ImageType::RegionType region_btm;

  for (int i = 0; i < num; ++i) {
    ImageType::Pointer image_src = lines_[lines_id_].second.first;
    ImageType::SizeType size_src = image_src->GetBufferedRegion().GetSize();
    ImageType::RegionType region_src = image_src->GetBufferedRegion();
    ImageType::SpacingType spacing_src = image_src->GetSpacing();
    ImageType::PointType origin_src = image_src->GetOrigin();
    ImageType::DirectionType direct_src = image_src->GetDirection();
	int src_buffer_length = image_src->GetBufferedRegion().GetNumberOfPixels();

    for (int j = 0; j < contour_num + 1; ++j) {
      ImageType::Pointer heatmap = ImageType::New();
      heatmap->SetDirection(direct_src);
      heatmap->SetOrigin(origin_src);
      heatmap->SetSpacing(spacing_src);
      heatmap->SetRegions(region_src);
      heatmap->Allocate();
      Dtype* heatmap_buffer = heatmap->GetBufferPointer();
	  memset(heatmap_buffer, 0, sizeof(Dtype) * src_buffer_length);
	
	  output_heatmap_[j] = heatmap;
    }

    LabelType::Pointer mask = LabelType::New();
    mask->SetDirection(direct_src);
    mask->SetOrigin(origin_src);
    mask->SetSpacing(spacing_src);
    mask->SetRegions(region_src);
    mask->Allocate();

    char* mask_buffer = mask->GetBufferPointer();
	memset(mask_buffer, 0, sizeof(char) * src_buffer_length);

    output_mask_ = mask;

    gt_ = lines_[lines_id_].second.second;

	Dtype* mask_max_score = new Dtype[src_buffer_length];
	memset(mask_max_score, 0, sizeof(Dtype) * src_buffer_length);

    LabelType::Pointer gt = gt_;
    char* gt_buffer = gt->GetBufferPointer();

    for (int j = 0; j < contour_num + 1; ++j) {
      ImageType::Pointer heatmap = output_heatmap_[j];
      Dtype* heatmap_buffer = heatmap->GetBufferPointer();
	  memcpy(heatmap_buffer, score + i * dim + j * src_buffer_length, sizeof(Dtype) * src_buffer_length);

	  for (int k = 0; k < src_buffer_length; ++k) {
        double heat_value = heatmap_buffer[k];
        if (mask_max_score[k] < heat_value) {
          mask_buffer[k] = j;
          mask_max_score[k] = heat_value;
        }
      }
    }
    for (int j = 0; j < contour_num + 1; ++j) {
      ImageType::Pointer heatmap = output_heatmap_[j];
      Dtype* heatmap_buffer = heatmap->GetBufferPointer();
      if (j > 0) {
        double a = 0, b = 0, a1 = 0, b1 = 0;
		double voe_a = 0, voe_b = 0;
		double rvd_a = 0, rvd_b = 0;
		ImageType::Pointer asd_input1 = ImageType::New();
		ImageType::Pointer asd_input2 = ImageType::New();
		asd_input1->SetDirection(direct_src);
		asd_input1->SetOrigin(origin_src);
		asd_input1->SetSpacing(spacing_src);
		asd_input1->SetRegions(region_src);
		asd_input1->Allocate();
		asd_input2->SetDirection(direct_src);
		asd_input2->SetOrigin(origin_src);
		asd_input2->SetSpacing(spacing_src);
		asd_input2->SetRegions(region_src);
		asd_input2->Allocate();
		Dtype* asd_input1_buffer = asd_input1->GetBufferPointer();
		Dtype* asd_input2_buffer = asd_input2->GetBufferPointer();
		memset(asd_input1_buffer, 0, sizeof(Dtype) * src_buffer_length);
		memset(asd_input2_buffer, 0, sizeof(Dtype) * src_buffer_length);

		for (int k = 0; k < src_buffer_length; ++k) {
          double label_value = (gt_buffer[k] == j) ? 1.0 : 0.0;
          double heat_value = heatmap_buffer[k];
          a += heat_value * label_value;
          b += heat_value * heat_value + label_value * label_value;
		  double mask_value = (mask_buffer[k] == j);
		  a1 += mask_value * label_value;
		  b1 += mask_value * mask_value + label_value * label_value;

		  voe_a += mask_value * label_value;
		  voe_b += (mask_value > 0 || label_value > 0) ? 1.0 : 0.0;

		  rvd_a += mask_value;
		  rvd_b += label_value;

		  asd_input1_buffer[k] = mask_value;
		  asd_input2_buffer[k] = label_value;
        }

        heat_intersection_[j] += a;
        heat_union_[j] += b;
		mask_intersection_[j] += a1;
		mask_union_[j] += b1;
		voe_a_[j] += voe_a;
		voe_b_[j] += voe_b;
		rvd_a_[j] += rvd_a;
		rvd_b_[j] += rvd_b;
		typedef itk::ContourDirectedMeanDistanceImageFilter<ImageType, ImageType> ASDImageFilterType;
		ASDImageFilterType::Pointer asd_filter = ASDImageFilterType::New();
		asd_filter->SetInput1(asd_input1);
		asd_filter->SetInput2(asd_input2);
		asd_filter->SetUseImageSpacing(true);
		asd_filter->Update();
		double asd = asd_filter->GetContourDirectedMeanDistance();
		asd_[j] += asd;
		asd_num_[j] += 1;
 		typedef itk::DirectedHausdorffDistanceImageFilter<ImageType, ImageType> MSDImageFilterType;
		MSDImageFilterType::Pointer msd_filter = MSDImageFilterType::New();
		msd_filter->SetInput1(asd_input1);
		msd_filter->SetInput2(asd_input2);
		msd_filter->SetUseImageSpacing(true);
		msd_filter->Update();
		double msd = msd_filter->GetDirectedHausdorffDistance();
		if (msd_[j] < msd)
		  msd_[j] = msd;
		msd_num_[j] += 1;
		double heatmap_dice = 2 * a / b;
		double mask_dice = 2 * a1 / b1;
        std::ofstream dice_file;
        dice_file.open("F:/dice_list.txt", ios::out | ios::app | ios::ate);
        if (dice_file.is_open()) {
          dice_file.fill('0');
		  dice_file.precision(7);
          dice_file << lines_[lines_id_].first << "\t" << heatmap_dice << "\t" << mask_dice << std::endl;
          dice_file.close();
        }

		if (segment_output_param.output_heatmap()) {
		  // resample to origin size
          ImageReaderType::Pointer reader_origin = ImageReaderType::New();
          reader_origin->SetFileName(root_folder + lines_[lines_id_].first);
          reader_origin->Update();
          ImageType::Pointer image_origin = reader_origin->GetOutput();
          ImageType::SizeType size_origin = image_origin->GetBufferedRegion().GetSize();
          ImageType::RegionType region_origin = image_origin->GetBufferedRegion();
          ImageType::SpacingType spacing_origin = image_origin->GetSpacing();
          ImageType::PointType origin_origin = image_origin->GetOrigin();
          ImageType::DirectionType direct_origin = image_origin->GetDirection();
		  
          ResampleImageFilterType::Pointer resampler_heatmap = ResampleImageFilterType::New();
          resampler_heatmap->SetInput(heatmap);
          resampler_heatmap->SetSize(size_origin);
          resampler_heatmap->SetOutputSpacing(spacing_origin);
          resampler_heatmap->SetOutputOrigin(origin_origin);
          resampler_heatmap->SetTransform(IdentityTransformType::New());
          //resampler_heatmap->SetInterpolator(InterpolatorType::New());
          resampler_heatmap->Update();
          heatmap = resampler_heatmap->GetOutput();
          heatmap->SetDirection(direct_origin);
		  
          const string heatmap_filename = output_folder + contour_name_list.name(j - 1) + "_heatmap_" + lines_[lines_id_].first;
          ImageWriterType::Pointer writer_heatmap = ImageWriterType::New();
          writer_heatmap->SetInput(heatmap);
          writer_heatmap->SetFileName(heatmap_filename);
          writer_heatmap->Update();
		}
		LOG(INFO) << "Total DSC of " << contour_name_list.name(j - 1) << " = " << 2 * mask_intersection_[j] / mask_union_[j];
		LOG(INFO) << "Total VOE of " << contour_name_list.name(j - 1) << " = " << 1 - voe_a_[j] / voe_b_[j];
		LOG(INFO) << "Total RVD of " << contour_name_list.name(j - 1) << " = " << (rvd_a_[j] - rvd_b_[j]) / rvd_b_[j];
		LOG(INFO) << "Total ASD of " << contour_name_list.name(j - 1) << " = " << asd_[j] / asd_num_[j];
		LOG(INFO) << "Total MSD of " << contour_name_list.name(j - 1) << " = " << msd_[j];
	  }
    }
	delete[]mask_max_score;

    double a = 0, b = 0;
	for (int j = 0; j < src_buffer_length; ++j) {
      double label_value = 0.0;
      double mask_value = 0.0;
      if (gt_buffer[j] != 0 && mask_buffer[j] != 0) {
        if (gt_buffer[j] == mask_buffer[j]) {
          label_value = 1.0;
          mask_value = 1.0;
        }
        else {
          label_value = 1.0;
          mask_value = 0.0;
        }
      }
      else if (gt_buffer[j] == 0 && mask_buffer[j] != 0) {
        label_value = 0.0;
        mask_value = 1.0;
      }
      else if (gt_buffer[j] != 0 && mask_buffer[j] == 0) {
        label_value = 1.0;
        mask_value = 0.0;
      }
      a += mask_value * label_value;
      b += mask_value * mask_value + label_value * label_value;
	}
    bin_intersection_ += a;
    bin_union_ += b;
    double mask_dice = 2 * a / b;

	if (segment_output_param.output_mask()) {
	  // resample to origin size
      ImageReaderType::Pointer reader_origin = ImageReaderType::New();
      reader_origin->SetFileName(root_folder + lines_[lines_id_].first);
      reader_origin->Update();
      ImageType::Pointer image_origin = reader_origin->GetOutput();
      ImageType::SizeType size_origin = image_origin->GetBufferedRegion().GetSize();
      ImageType::RegionType region_origin = image_origin->GetBufferedRegion();
      ImageType::SpacingType spacing_origin = image_origin->GetSpacing();
      ImageType::PointType origin_origin = image_origin->GetOrigin();
      ImageType::DirectionType direct_origin = image_origin->GetDirection();
	  
      ResampleLabelFilterType::Pointer resampler_mask = ResampleLabelFilterType::New();
      resampler_mask->SetInput(mask);
      resampler_mask->SetSize(size_origin);
      resampler_mask->SetOutputSpacing(spacing_origin);
      resampler_mask->SetOutputOrigin(origin_origin);
      resampler_mask->SetTransform(IdentityTransformType::New());
      resampler_mask->SetInterpolator(InterpolatorType::New());
      resampler_mask->Update();
      mask = resampler_mask->GetOutput();
      mask->SetDirection(direct_origin);
      mask_buffer = mask->GetBufferPointer();

      const string mask_filename = output_folder + "mask_" + lines_[lines_id_].first;
	  LabelWriterType::Pointer writer_mask = LabelWriterType::New();
	  writer_mask->SetInput(mask);
	  writer_mask->SetFileName(mask_filename);
	  writer_mask->Update();
	}
    LOG(INFO) << "Total mask dice = " << 2 * bin_intersection_ / bin_union_;

    if (segment_output_param.output_dice_file() && lines_id_ == lines_.size() - 1) {
      const string& dice_file_name = segment_output_param.dice_file_name();
      std::ofstream dice_file;
      dice_file.open(dice_file_name, ios::out | ios::app | ios::ate);
      if (dice_file.is_open()) {
        dice_file.fill('0');
		dice_file.precision(7);
        for (int j = 1; j < contour_num + 1; ++j) {
		  dice_file 
			  << 2 * heat_intersection_[j] / heat_union_[j] << "\t" 
			  << 2 * mask_intersection_[j] / mask_union_[j] << "\t" 
			  << 1 - voe_a_[j] / voe_b_[j] << "\t" 
			  << (rvd_a_[j] - rvd_b_[j]) / rvd_b_[j] << "\t"
			  << asd_[j] / asd_num_[j] << "\t" 
			  << msd_[j] << "\t";
        }
        dice_file << 2 * bin_intersection_ / bin_union_ << std::endl;
        dice_file.close();
      }
    }

    lines_id_ = (lines_id_ + 1) % lines_.size();
  }
}

template <typename Dtype>
void SegmentOutputLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_CLASS(SegmentOutputLayer);
REGISTER_LAYER_CLASS(SegmentOutput);

}  // namespace caffe
