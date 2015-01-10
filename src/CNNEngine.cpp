
#include "CNNEngine.h"
#include "CaffeWrapper.h"
#include "caffe/caffe.hpp"

using namespace CNN;

CNNEngine* CNNEngine::Create(const char *model_path, const char *params_path)
{
	return new CaffeWrapper(model_path, params_path);
}

void CNNEngine::Delete(CNNEngine *engine)
{
	delete engine;
}
	
CNNEngine::CNNEngine()
{

}

CNNEngine::~CNNEngine()
{

}

void CNNEngine::GlobalInit(int argc, char** argv)
{
	caffe::GlobalInit(&argc, &argv);
#if defined(_MSC_VER) || defined (__MACH__)
	caffe::InitLayerFactory();
#endif
}