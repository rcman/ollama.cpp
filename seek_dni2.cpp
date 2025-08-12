#include <jni.h>
#include <string>
#include "ollama.h"

static struct ollama_model model;
static struct ollama_params params;

extern "C" JNIEXPORT jboolean JNICALL
Java_OllamaNativeInterface_initialize(JNIEnv *, jobject) {
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
Java_OllamaNativeInterface_setParams(
    JNIEnv *, jobject, 
    jint ctx, jfloat temp, jint top_k, 
    jfloat top_p, jfloat repeat_penalty, jint batch_size
) {
    params = {
        .n_ctx = ctx,
        .temp = temp,
        .top_k = top_k,
        .top_p = top_p,
        .repeat_penalty = repeat_penalty,
        .n_batch = batch_size,
        // Other parameters set to defaults
    };
}

extern "C" JNIEXPORT jstring JNICALL
Java_OllamaNativeInterface_generateText(JNIEnv *env, jobject, jstring prompt) {
    const char *input = env->GetStringUTFChars(prompt, 0);
    char *response = ollama_generate_text_with_params(&model, &params, input);
    env->ReleaseStringUTFChars(prompt, input);
    jstring result = env->NewStringUTF(response);
    free(response);
    return result;
}