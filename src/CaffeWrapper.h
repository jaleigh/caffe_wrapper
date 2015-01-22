#ifndef CAFFE_WRAPPER_H
#define CAFFE_WRAPPER_H

#include "CNNEngine.h"
#include "caffe/caffe.hpp"
#include <memory>

namespace CNN
{

class CaffeWrapper : public CNNEngine
{

public:

	CaffeWrapper(const char *model_path, const char *trained_params_path);

    ~CaffeWrapper();
    
	virtual CNNResult PredictImage(const char *image_filepath);

	virtual CNNResult PredictImage(const char *image_filepath, IMAGE_ORIENTATION orientation);

	bool ExtractFeatures(const char *image_filepath, IMAGE_ORIENTATION orientation, CNNEngine::CNNFeature& output);

private:

	bool PreprocessAndLoadImage(const char *image_filepath, IMAGE_ORIENTATION orientation);

// Attributes
private:

	caffe::Net<double> *model;

};

} // namespace

#endif