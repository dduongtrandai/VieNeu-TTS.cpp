#include "../vieneu_v3_onnx.h"
#include "vieneu_v3_onnx_internal.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <nlohmann/json.hpp>

// --- VieneuV3OnnxEngine Voice Member Functions ---

bool VieneuV3OnnxEngine::load_voices(const std::string& voices_path, std::string& error) {
    voices_json_.clear();
    default_voice_id_.clear();
    voice_presets_.clear();
    if (voices_path.empty()) {
        return true;
    }
    if (!read_text_file(voices_path, voices_json_)) {
        error = "Failed to read VieNeu v3 voices JSON: " + voices_path;
        return false;
    }
    try {
        const auto root = nlohmann::json::parse(voices_json_);
        if (root.contains("default_voice") && root.at("default_voice").is_string()) {
            default_voice_id_ = root.at("default_voice").get<std::string>();
        }
        if (!root.contains("presets") || !root.at("presets").is_object()) {
            return true;
        }

        const auto& presets = root.at("presets");
        for (auto it = presets.begin(); it != presets.end(); ++it) {
            const std::string id = it.key();
            const auto& item = it.value();
            VoicePreset preset;
            preset.found = true;
            if (item.contains("reserved_id") && !item.at("reserved_id").is_null()) {
                preset.has_reserved_id = true;
                preset.reserved_id = item.at("reserved_id").get<int>();
            }
            if (item.contains("codes") && item.at("codes").is_array()) {
                const auto& codes = item.at("codes");
                for (const auto& row : codes) {
                    if (!row.is_array() || static_cast<int>(row.size()) != config_.n_vq) {
                        error = "VieNeu v3 preset voice has invalid codes shape: " + id;
                        return false;
                    }
                    for (const auto& v : row) {
                        preset.codes.push_back(v.get<int64_t>());
                    }
                }
            }
            voice_presets_[id] = std::move(preset);
        }
    } catch (const std::exception& e) {
        error = std::string("Failed to parse VieNeu v3 voices JSON: ") + e.what();
        return false;
    }
    return true;
}

bool VieneuV3OnnxEngine::parse_voice_reserved_id(const std::string& voice_id, int& reserved_id) const {
    static const std::unordered_map<std::string, int> fallback = {
        {"Ngọc Lan", 13}, {"Ngọc Linh", 14}, {"Trúc Ly", 15}, {"Mỹ Duyên", 16},
        {"Xuân Vĩnh", 17}, {"Thái Sơn", 18}, {"Gia Bảo", 19}, {"Đức Trí", 20},
        {"Trọng Hữu", 21}, {"Bình An", 22}
    };
    if (!voice_id.empty()) {
        auto it = fallback.find(voice_id);
        if (it != fallback.end()) {
            reserved_id = it->second;
            return true;
        }
    }
    if (voice_id.empty()) {
        return false;
    }
    auto it = voice_presets_.find(voice_id);
    if (it != voice_presets_.end() && it->second.has_reserved_id) {
        reserved_id = it->second.reserved_id;
        return true;
    }
    return false;
}

bool VieneuV3OnnxEngine::resolve_voice_preset(
    const std::string& voice_id,
    VoicePreset& preset,
    std::string& error) const {
    preset = VoicePreset{};
    if (voice_presets_.empty()) {
        return true;
    }

    const std::string selected = voice_id.empty() ? default_voice_id_ : voice_id;
    if (selected.empty()) {
        return true;
    }

    auto it = voice_presets_.find(selected);
    if (it == voice_presets_.end()) {
        error = "VieNeu v3 voice preset not found: " + selected;
        return false;
    }
    preset = it->second;
    return true;
}
