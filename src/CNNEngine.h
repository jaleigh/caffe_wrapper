#ifndef CNN_ENGINE_H
#define CNN_ENGINE_H

#include <string>
#include <vector>
#include "host.h"

namespace CNN
{

class CNN_EXPORT CNNEngine
{

public:

	// stores all data for a classification result
	struct CNNResult
	{
		enum {NUM_RESULTS=5};
		double scores[NUM_RESULTS];
		int class_id[NUM_RESULTS];

		CNNResult()
		{
			for(int i=0; i < NUM_RESULTS; i++)
			{
				scores[i] = 0;
				class_id[i] = -1;
			}
		}
	};

	struct CNNFeature
	{
		enum {MAX_COUNT=4096};
		double values[MAX_COUNT];
		int count;

		CNNFeature() : count(0) {}
	};

	enum IMAGE_ORIENTATION {UPRIGHT, CLOCKWISE_90, ANTI_CLOCKWISE_90, UPSIDE_DOWN, ORIENTATION_UNKNOWN};

	/// urrgh to match caffe
	static void GlobalInit(int argc, char** argv);

	// predict the tags of the image, e.g. scene type etc
	virtual CNNResult PredictImage(const char *image_filepath) = 0;

	// predict the tags of the image with known image orientation, e.g. scene type etc
	virtual CNNResult PredictImage(const char *image_filepath, IMAGE_ORIENTATION orientation) = 0;

	// extract the top layer feature vector from the image, with known image orientation
	virtual bool ExtractFeatures(const char *image_filepath, IMAGE_ORIENTATION orientation, CNNEngine::CNNFeature& output) = 0;

	// model_path = filepath to the caffe model to use for classification.
	// CNNEngines can only be created through this, so all memory allocation happens
	// inside the dll
	static CNNEngine* Create(const char *model_path, const char *params_path);

	// deletes a previously allocated CNNEngine
	static void Delete(CNNEngine *engine);

protected:

	CNNEngine();
	virtual ~CNNEngine() = 0;

};

} // namespace

#endif