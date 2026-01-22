#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "legged_rl_deploy/deploy_config.h"

namespace legged_rl_deploy {

/**
 * ObservationAssembler
 * - Input: per-term observation vectors at each tick
 * - Maintains: per-term ring buffers for history_length
 * - Output: flattened observation vector in config order
 *
 * Layout rule:
 *   For each term in cfg.observations.terms order:
 *     append [term(t), term(t-1), ..., term(t-H+1)]
 *   where each term(.) is a dim-D vector.
 */
class ObservationAssembler {
public:
  // If expected_obs_dim >= 0, constructor validates total obs dim.
  explicit ObservationAssembler(const ObservationsConfig& cfg, int expected_obs_dim = -1)
      : cfg_(cfg) {
    BuildIndex();
    BuildBuffers();
    BuildLayout();
    if (expected_obs_dim >= 0) {
      ValidateExpectedDim(expected_obs_dim);
    }
  }

  // ---------- Meta ----------
  int obs_dim() const { return obs_dim_; }
  int num_terms() const { return static_cast<int>(terms_.size()); }

  // ordered term names (same as YAML order)
  const std::vector<std::string>& term_names() const { return names_; }

  // per-term flattened dim = dim * H (same order as term_names())
  const std::vector<int>& per_term_flat_dims() const { return per_term_flat_dims_; }

  // term offsets in final obs (same order)
  const std::vector<int>& term_offsets() const { return term_offsets_; }

  // name -> index (for fast lookup)
  const std::unordered_map<std::string, int>& name_to_index() const { return name2idx_; }

  // Optional: name -> flat_dim map (useful for logging)
  std::unordered_map<std::string, int> PerTermFlatDimMap() const {
    std::unordered_map<std::string, int> m;
    m.reserve(static_cast<size_t>(num_terms()));
    for (int i = 0; i < num_terms(); ++i) {
      m.emplace(names_[i], per_term_flat_dims_[i]);
    }
    return m;
  }

  // Validate overall observation dim equals expected_dim.
  // Throws with per-term breakdown if mismatch.
  void ValidateExpectedDim(int expected_dim) const {
    if (obs_dim_ == expected_dim) return;

    std::ostringstream oss;
    oss << "Observation dim mismatch:\n"
        << "  expected total dim = " << expected_dim << "\n"
        << "  actual total dim   = " << obs_dim_ << "\n"
        << "  per-term breakdown (name : dim*H = flat_dim, offset):\n";

    for (int i = 0; i < num_terms(); ++i) {
      const int flat = per_term_flat_dims_[i];
      const int off  = term_offsets_[i];
      const int dim  = buffers_[static_cast<size_t>(i)].dim;
      const int H    = buffers_[static_cast<size_t>(i)].H;
      oss << "    - " << names_[i]
          << " : " << dim << "*" << H
          << " = " << flat
          << ", offset=" << off << "\n";
    }
    throw std::runtime_error(oss.str());
  }

  // ---------- State ----------
  void Reset() {
    for (auto& b : buffers_) {
      std::fill(b.ring.begin(), b.ring.end(), 0.0f);
      b.head = -1;
      b.count = 0;
      b.seen = false;
    }
  }

  // ---------- Push API (recommended: by index) ----------
  void PushRawByIndex(int term_idx, const std::vector<double>& raw) {
    CheckTermIndex(term_idx);
    TermBuffer& b = buffers_[static_cast<size_t>(term_idx)];
    RequireRawDim(term_idx, raw.size());

    // clip + scale
    tmp_processed_.assign(static_cast<size_t>(b.dim), 0.0f);
    for (int i = 0; i < b.dim; ++i) {
      float x = static_cast<float>(raw[static_cast<size_t>(i)]);
      x = std::clamp(x, b.clip_lo, b.clip_hi);
      x *= b.scale[static_cast<size_t>(i)];
      tmp_processed_[static_cast<size_t>(i)] = x;
    }
    PushProcessedByIndex(term_idx, tmp_processed_);
  }

  void PushProcessedByIndex(int term_idx, const std::vector<float>& vec) {
    CheckTermIndex(term_idx);
    TermBuffer& b = buffers_[static_cast<size_t>(term_idx)];
    if (static_cast<int>(vec.size()) != b.dim) {
      throw std::runtime_error("PushProcessedByIndex(): dim mismatch for term '" +
                               names_[term_idx] + "': expected " +
                               std::to_string(b.dim) + ", got " +
                               std::to_string(vec.size()));
    }

    // ring push
    b.head = (b.head + 1) % b.H;
    std::copy(vec.begin(), vec.end(), b.ring.begin() + b.head * b.dim);
    b.count = std::min(b.count + 1, b.H);
    b.seen = true;
  }

  // ---------- Push API (by name) ----------
  void PushRaw(const std::string& term_name, const std::vector<double>& raw) {
    auto it = name2idx_.find(term_name);
    if (it == name2idx_.end()) {
      throw std::runtime_error("PushRaw(): unknown term '" + term_name + "'.");
    }
    PushRawByIndex(it->second, raw);
  }

  void PushProcessed(const std::string& term_name, const std::vector<float>& vec) {
    auto it = name2idx_.find(term_name);
    if (it == name2idx_.end()) {
      throw std::runtime_error("PushProcessed(): unknown term '" + term_name + "'.");
    }
    PushProcessedByIndex(it->second, vec);
  }

  // ---------- Assemble ----------
  // require_all_terms=true: any term never pushed -> throw
  // false: unseen terms treated as zeros (including history)
  std::vector<float> Assemble(bool require_all_terms = true) const {
    if (require_all_terms) {
      for (int i = 0; i < num_terms(); ++i) {
        if (!buffers_[static_cast<size_t>(i)].seen) {
          throw std::runtime_error("Assemble(): term '" + names_[i] +
                                   "' has never been pushed.");
        }
      }
    }

    std::vector<float> out(static_cast<size_t>(obs_dim_), 0.0f);

    // fill each term block
    for (int ti = 0; ti < num_terms(); ++ti) {
      const TermBuffer& b = buffers_[static_cast<size_t>(ti)];
      const int base = term_offsets_[ti];

      if (!b.seen) continue;  // allow zero-fill

      // within block: [t, t-1, ..., t-H+1], each is dim
      for (int k = 0; k < b.H; ++k) {
        const int dst = base + k * b.dim;
        if (k >= b.count) {
          // not enough history => keep zeros
          continue;
        }
        int idx = b.head - k;
        if (idx < 0) idx += b.H;

        const float* src = b.ring.data() + idx * b.dim;
        std::copy(src, src + b.dim, out.begin() + dst);
      }
    }

    return out;
  }

private:
  struct TermInfo {
    std::string name;
    const ObsTerm* term = nullptr;
  };

  struct TermBuffer {
    int dim = 0;   // scale.size()
    int H = 1;     // history_length
    float clip_lo = -std::numeric_limits<float>::infinity();
    float clip_hi =  std::numeric_limits<float>::infinity();
    std::vector<float> scale;  // size dim

    // ring: H frames, each dim
    std::vector<float> ring;   // size H * dim
    int head = -1;
    int count = 0;
    bool seen = false;
  };

  void BuildIndex() {
    terms_.clear();
    names_.clear();
    name2idx_.clear();

    terms_.reserve(cfg_.terms.size());
    names_.reserve(cfg_.terms.size());
    name2idx_.reserve(cfg_.terms.size());

    for (size_t i = 0; i < cfg_.terms.size(); ++i) {
      const auto& kv = cfg_.terms[i];
      const std::string& name = kv.first;

      if (name2idx_.count(name)) {
        throw std::runtime_error("ObservationAssembler: duplicate term name '" + name + "'.");
      }

      TermInfo info;
      info.name = name;
      info.term = &kv.second;

      terms_.push_back(info);
      names_.push_back(name);
      name2idx_[name] = static_cast<int>(i);
    }
  }

  void BuildBuffers() {
    buffers_.clear();
    buffers_.reserve(terms_.size());

    for (size_t i = 0; i < terms_.size(); ++i) {
      const ObsTerm& t = *terms_[i].term;

      if (t.history_length <= 0) {
        throw std::runtime_error("ObservationAssembler: term '" + terms_[i].name +
                                 "' has invalid history_length=" +
                                 std::to_string(t.history_length));
      }
      if (t.clip.size() != 2) {
        throw std::runtime_error("ObservationAssembler: term '" + terms_[i].name +
                                 "' clip must have size 2.");
      }
      if (t.scale.empty()) {
        throw std::runtime_error("ObservationAssembler: term '" + terms_[i].name +
                                 "' scale is empty.");
      }

      TermBuffer b;
      b.dim = static_cast<int>(t.scale.size());
      b.H = t.history_length;
      b.clip_lo = static_cast<float>(t.clip[0]);
      b.clip_hi = static_cast<float>(t.clip[1]);

      b.scale.resize(static_cast<size_t>(b.dim));
      for (int k = 0; k < b.dim; ++k) {
        b.scale[static_cast<size_t>(k)] =
            static_cast<float>(t.scale[static_cast<size_t>(k)]);
      }

      b.ring.assign(static_cast<size_t>(b.H * b.dim), 0.0f);
      b.head = -1;
      b.count = 0;
      b.seen = false;

      buffers_.push_back(std::move(b));
    }
  }

  void BuildLayout() {
    obs_dim_ = 0;
    per_term_flat_dims_.clear();
    term_offsets_.clear();

    per_term_flat_dims_.reserve(terms_.size());
    term_offsets_.reserve(terms_.size());

    for (size_t i = 0; i < terms_.size(); ++i) {
      const TermBuffer& b = buffers_[i];
      term_offsets_.push_back(obs_dim_);
      const int flat_dim = b.dim * b.H;
      per_term_flat_dims_.push_back(flat_dim);
      obs_dim_ += flat_dim;
    }
  }

  void CheckTermIndex(int idx) const {
    if (idx < 0 || idx >= num_terms()) {
      throw std::runtime_error("Invalid term_idx=" + std::to_string(idx) +
                               ", num_terms=" + std::to_string(num_terms()));
    }
  }

  void RequireRawDim(int term_idx, size_t raw_size) const {
    const int expected = buffers_[static_cast<size_t>(term_idx)].dim;
    if (static_cast<int>(raw_size) != expected) {
      throw std::runtime_error("PushRaw(): dim mismatch for term '" + names_[term_idx] +
                               "': expected " + std::to_string(expected) +
                               ", got " + std::to_string(raw_size));
    }
  }

private:
  const ObservationsConfig& cfg_;

  // ordered terms (same order as cfg_.terms)
  std::vector<TermInfo> terms_;
  std::vector<std::string> names_;
  std::unordered_map<std::string, int> name2idx_;

  // buffers per term, same order
  std::vector<TermBuffer> buffers_;

  // layout
  int obs_dim_ = 0;
  std::vector<int> per_term_flat_dims_;
  std::vector<int> term_offsets_;

  // temp buffer (avoid alloc each PushRaw)
  mutable std::vector<float> tmp_processed_;
};

}  // namespace legged_rl_deploy
