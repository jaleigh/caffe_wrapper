#ifndef CNN_HOST_H
#define CNN_HOST_H

#if defined (_MSC_VER)
	#ifdef CNN_BUILD_DLL
		#define CNN_EXPORT __declspec(dllexport)
	#else
		#define CNN_EXPORT __declspec(dllimport)
	#endif
#else
    #ifdef CNN_BUILD_DLL
        #define CNN_EXPORT __attribute__((visibility("default")))
    #else
        #define CNN_EXPORT
    #endif
#endif

#endif