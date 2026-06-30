#include "vieneu/vieneu_tts.h"
#include <stdio.h>

static void progress_cb(const struct vieneu_progress * progress, void * user_data) {
    (void)user_data;
    if (progress) {
        printf("[ABI Test] Progress callback type OK: %s %.2f\n",
               progress->stage ? progress->stage : "",
               progress->progress);
    }
}

int main(void) {
    printf("[ABI Test] VieNeu TTS Public ABI Contract: OK\n");
    printf("[ABI Test] Runtime version: %s\n", vieneu_version());
    
    // Smoke check structures and functions exist
    struct vieneu_init_params params;
    vieneu_init_default_params(&params);
    
    struct vieneu_tts_params tts_params;
    vieneu_tts_default_params(&tts_params);

    struct vieneu_progress progress;
    progress.abi_version = 1;
    progress.stage = "test";
    progress.current = 1;
    progress.total = 1;
    progress.progress = 1.0f;
    progress.message = "ok";
    progress_cb(&progress, NULL);
    vieneu_set_progress_callback(NULL, progress_cb, NULL);
    
    return 0;
}
