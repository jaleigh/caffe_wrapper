// CNNWrapperExe.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "CNNEngine.h"
#include <iostream>

int _tmain(int argc, _TCHAR* argv[])
{
	std::string model_path;
	std::string params_path;
	std::string images_path;

	for(int i=0; i < argc; i++)
	{
		if(std::string(argv[i]) == "-help")
		{
			std::cout << "-model \"filepath\"   This is the caffe .prototxt file" << std::endl;
			std::cout << "-params \"filepath\"   This is the caffe .caffemodel file" << std::endl;
			std::cout << "-images \"filepath\"   This is a text file containing the full filepath to an image to be processed. Each image should be on a new line" << std::endl;
		}
		else if(std::string(argv[i]) == "-model")
		{
			if(i+1 < argc) {
				model_path = argv[i+1];
			}
		}
		else if(std::string(argv[i]) == "-params")
		{
			if(i+1 < argc) {
				params_path = argv[i+1];
			}
		}
		else if(std::string(argv[i]) == "-images")
		{
			if(i+1 < argc) {
				images_path = argv[i+1];
			}
		}
	}

	CNN::CNNEngine::GlobalInit(argc, argv);

	CNN::CNNEngine *engine = CNN::CNNEngine::Create(model_path.c_str(), params_path.c_str());

	FILE *fp = fopen(images_path.c_str(), "r");
	if(fp)
	{
		const int MAX_LINE_LENGTH = 255;
		char line[MAX_LINE_LENGTH];
		while(fgets(line, MAX_LINE_LENGTH, fp) != NULL)
		{
			printf("Run %s \n", line);

			CNN::CNNEngine::CNNResult res = engine->PredictImage(line);

			for(int i=0; i < res.NUM_RESULTS; i++)
			{
				printf("%d - %f\n", res.class_id[i], res.scores[i]);
			}

			printf("\n");
		}
	}
	else
	{
		std::cout << "Could not open image file";
	}

	CNN::CNNEngine::Delete(engine);

	char c;
	std::cin >> c;

	return 0;
}

