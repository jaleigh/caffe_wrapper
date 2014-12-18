
#include "CaffeWrapper.h"
#include "caffe/caffe.hpp"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace CNN;
using namespace caffe;

namespace caffe{
	LayerRegisterer<float> get_g_creator_f_CONVOLUTION();
}

CaffeWrapper::CaffeWrapper(const char *model_path, const char *trained_params_path)
	:
CNNEngine()
{
	Caffe::set_mode(Caffe::CPU);
	Caffe::set_phase(Caffe::TEST);

	model = std::shared_ptr<caffe::Net<double>>(new Net<double>(model_path));

	model->CopyTrainedLayersFrom(trained_params_path);
}

CNNEngine::CNNResult CaffeWrapper::PredictImage(const char *image_filepath)
{
	return PredictImage(image_filepath, UPRIGHT);
}

CNNEngine::CNNResult CaffeWrapper::PredictImage(const char *image_filepath, IMAGE_ORIENTATION orientation)
{
	CNNResult result;

	cv::Mat src = cv::imread(image_filepath);

	if(src.cols == 0 || src.rows == 0) {
		return result;
	}

	if(orientation != UPRIGHT)
	{
		cv::Mat dest;
		cv::Mat rot_mat;
		if(orientation == UPSIDE_DOWN)
		{
			rot_mat = cv::getRotationMatrix2D(cv::Point2f(src.cols * 0.5f, src.rows * 0.5f), -180.f, 1.f);
		}
		else if(orientation == ANTI_CLOCKWISE_90)
		{
			rot_mat = cv::getRotationMatrix2D(cv::Point2f(src.cols * 0.5f, src.rows * 0.5f), 90.f, 1.f);
		}
		else if(orientation == CLOCKWISE_90)
		{
			rot_mat = cv::getRotationMatrix2D(cv::Point2f(src.cols * 0.5f, src.rows * 0.5f), -90.f, 1.f);
		}
		cv::warpAffine(src, dest, rot_mat, src.size());
		src = dest;
	}

	cv::Mat dest;

	const boost::shared_ptr<MemoryDataLayer<double> > memory_layer = 
		boost::dynamic_pointer_cast<MemoryDataLayer<double> >(model->layer_by_name("data"));

	// resize the image to 256x256 as image needs to be the same size as the mean image used by the memory layer
	// presumably this should be handled by the transformer.

	// following ImageNet, the image is resized so the shortest side if 256, we then crop out a 256x256 from the
	// centre of the remaining image.
	cv::Size new_size;
	cv::Rect rect;
	if(src.rows > src.cols)
	{
		new_size = cv::Size(256, src.rows * (256.f / (float)src.cols));
		rect.x = 0;
		rect.width = 256;
		rect.height = 256;
		rect.y = (int)(0.5 + (new_size.height - 256) * 0.5f);
	}
	else
	{
		new_size = cv::Size(src.cols * (256.f / (float)src.rows), 256);
		rect.y = 0;
		rect.width = 256;
		rect.height = 256;
		rect.x = (int)(0.5 + (new_size.width - 256) * 0.5f);

	}
	cv::resize(src, dest, new_size);
	dest = dest(rect);

	std::vector<cv::Mat> input;
	input.push_back(dest);
	std::vector<int> label;
	label.push_back(0);

	memory_layer->AddMatVector(input, label);

	// classify the image
	double loss = 0.f;
	const vector<Blob<double>*>& output = model->ForwardPrefilled(&loss);

	// populate the results, store the top N results, store the scores and the string
	for(int n=0; n < CNNResult::NUM_RESULTS; n++)
	{
		result.class_id[n] = -1;
		result.scores[n] = -9.9e9;
	}
	
	for (int i = 0; i < output[1]->count(); ++i) 
	{
		float value = output[1]->cpu_data()[i];
		for(size_t n=0; n < CNNResult::NUM_RESULTS; n++)
		{
			if(value > result.scores[n])
			{
				// move all results down 1
				for(int j=CNNResult::NUM_RESULTS-1; j > n; j--)
				{
					result.class_id[j] = result.class_id[j-1];
					result.scores[j] = result.scores[j-1];
				}

				result.scores[n] = value;
				result.class_id[n] = i;
				break;
			}
		}
	} 

	return result;
}