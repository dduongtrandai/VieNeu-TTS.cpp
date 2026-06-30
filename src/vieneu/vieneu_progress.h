#ifndef VIENEU_PROGRESS_H
#define VIENEU_PROGRESS_H

#include <functional>
#include <string>

struct VieneuProgressEvent {
    const char* stage = "";
    int current = 0;
    int total = 0;
    float progress = 0.0f;
    std::string message;
};

using VieneuProgressFn = std::function<void(const VieneuProgressEvent&)>;

inline void vieneu_report_progress(const VieneuProgressFn& fn,
                                   const char* stage,
                                   int current,
                                   int total,
                                   float progress,
                                   const std::string& message = std::string()) {
    if (!fn) {
        return;
    }
    VieneuProgressEvent event;
    event.stage = stage ? stage : "";
    event.current = current;
    event.total = total;
    event.progress = progress;
    event.message = message;
    fn(event);
}

#endif // VIENEU_PROGRESS_H
