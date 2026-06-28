#include "vieneu/vieneu_tts.h"
#include <stdio.h>

int main(void) {
    printf("[ABI Test] VieNeu TTS Public ABI Contract: OK\n");
    printf("[ABI Test] Runtime version: %s\n", vieneu_version());
    
    // Smoke check structures and functions exist
    struct vieneu_init_params params;
    vieneu_init_default_params(&params);
    
    struct vieneu_tts_params tts_params;
    vieneu_tts_default_params(&tts_params);
    
    return 0;
}
