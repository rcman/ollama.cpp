#include <jni.h>
#include <string>
#include "ollama.h" // Ollama.cpp header

static struct ollama_model model;

extern "C" JNIEXPORT jboolean JNICALL
Java_OllamaNativeInterface_initialize(JNIEnv *env, jobject) {
    // Initialize Ollama library
    return ollama_init() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_OllamaNativeInterface_loadModel(JNIEnv *env, jobject, jstring modelPath) {
    const char *path = env->GetStringUTFChars(modelPath, 0);
    bool success = ollama_load_model(path, &model);
    env->ReleaseStringUTFChars(modelPath, path);
    return success ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_OllamaNativeInterface_unloadModel(JNIEnv *, jobject) {
    ollama_unload_model(&model);
}

extern "C" JNIEXPORT jstring JNICALL
Java_OllamaNativeInterface_generateText(JNIEnv *env, jobject, jstring prompt) {
    const char *input = env->GetStringUTFChars(prompt, 0);
    char *response = ollama_generate_text(&model, input);
    env->ReleaseStringUTFChars(prompt, input);
    jstring result = env->NewStringUTF(response);
    free(response);
    return result;
}