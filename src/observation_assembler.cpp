#include "legged_rl_deploy/observation_assembler.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace legged_rl_deploy {

ObservationAssembler::ObservationAssembler(const ObservationsConfig& cfg,
                                           int expected_obs_dim)
    : cfg_(cfg) {
  BuildIndex();
  BuildBuffers();
  BuildLayout();
  if (expected_obs_dim >= 0) {
    ValidateExpectedDim(expected_obs_dim);
  }
}

std::unordered_map<std::string, int> ObservationAssembler::PerTermFlatDimMap() const {
  std::unordered_map<std::string, int> m;
  m.reserve(static_cast<size_t>(num_terms()));
  for (int i = 0; i < num_terms(); ++i) {
    m.emplace(names_[i], per_term_flat_dims_[i]);
  }
  return m;
}

void ObservationAssembler::ValidateExpectedDim(int expected_dim) const {
  if (obs_dim_ == expected_dim) return;

  std::ostringstream oss;
  oss << "Observation dim mismatch:\n"
      << "  expected total dim = " << expected_dim << "\n"
      << "  actual total dim   = " << obs_dim_ << "\n";

  if (cfg_.stack_method == ObsStackMethod::StackThenConcat) {
    oss << "  layout = StackThenConcat (term-major)\n"
        << "  per-term breakdown (name : dim*H_i = flat_dim, final_offset):\n";
    for (int i = 0; i < num_terms(); ++i) {
      const int dim  = buffers_[static_cast<size_t>(i)].dim;
      const int H    = buffers_[static_cast<size_t>(i)].H;
      const int flat = per_term_flat_dims_[i];
      const int off  = term_offsets_[i];
      oss << "    - " << names_[i]
          << " : " << dim << "*" << H
          << " = " << flat
          << ", offset=" << off << "\n";
    }
  } else {
    oss << "  layout = ConcatThenStack (time-major)\n"
        << "  NOTE: terms are interleaved across frames (not contiguous slices)\n"
        << "  frame_dim=" << frame_dim_ << ", H=" << global_H_ << "\n"
        << "  per-term breakdown (name : dim, frame_offset):\n";
    for (int i = 0; i < num_terms(); ++i) {
      const int dim = buffers_[static_cast<size_t>(i)].dim;
      const int off = term_in_frame_offsets_[i];
      oss << "    - " << names_[i]
          << " : dim=" << dim
          << ", frame_offset=" << off << "\n";
    }
  }

  throw std::runtime_error(oss.str());
}

void ObservationAssembler::Reset() {
  for (auto& b : buffers_) {
    std::fill(b.ring.begin(), b.ring.end(), 0.0f);
    b.head = -1;
    b.count = 0;
    b.seen = false;
  }
}

void ObservationAssembler::PushRawByIndex(int term_idx, const std::vector<double>& raw) {
  CheckTermIndex(term_idx);
  TermBuffer& b = buffers_[static_cast<size_t>(term_idx)];
  RequireRawDim(term_idx, raw.size());

  tmp_processed_.assign(static_cast<size_t>(b.dim), 0.0f);
  for (int i = 0; i < b.dim; ++i) {
    float x = static_cast<float>(raw[static_cast<size_t>(i)]);
    x = std::clamp(x, b.clip_lo, b.clip_hi);
    x *= b.scale[static_cast<size_t>(i)];
    tmp_processed_[static_cast<size_t>(i)] = x;
  }
  PushProcessedByIndex(term_idx, tmp_processed_);
}

void ObservationAssembler::PushProcessedByIndex(int term_idx, const std::vector<float>& vec) {
  CheckTermIndex(term_idx);
  TermBuffer& b = buffers_[static_cast<size_t>(term_idx)];
  if (static_cast<int>(vec.size()) != b.dim) {
    throw std::runtime_error("PushProcessedByIndex(): dim mismatch for term '" +
                             names_[term_idx] + "': expected " +
                             std::to_string(b.dim) + ", got " +
                             std::to_string(vec.size()));
  }

  b.head = (b.head + 1) % b.H;
  std::copy(vec.begin(), vec.end(), b.ring.begin() + b.head * b.dim);
  b.count = std::min(b.count + 1, b.H);
  b.seen = true;
}

void ObservationAssembler::PushRaw(const std::string& term_name, const std::vector<double>& raw) {
  auto it = name2idx_.find(term_name);
  if (it == name2idx_.end()) {
    throw std::runtime_error("PushRaw(): unknown term '" + term_name + "'.");
  }
  PushRawByIndex(it->second, raw);
}

void ObservationAssembler::PushProcessed(const std::string& term_name, const std::vector<float>& vec) {
  auto it = name2idx_.find(term_name);
  if (it == name2idx_.end()) {
    throw std::runtime_error("PushProcessed(): unknown term '" + term_name + "'.");
  }
  PushProcessedByIndex(it->second, vec);
}

std::vector<float> ObservationAssembler::Assemble(bool require_all_terms) const {
  if (require_all_terms) {
    for (int i = 0; i < num_terms(); ++i) {
      if (!buffers_[static_cast<size_t>(i)].seen) {
        throw std::runtime_error("Assemble(): term '" + names_[i] + "' has never been pushed.");
      }
    }
  }
  return (cfg_.stack_method == ObsStackMethod::StackThenConcat) ? AssembleTermMajor() : AssembleTimeMajor();
}

std::vector<float> ObservationAssembler::AssembleTermMajor() const {
  std::vector<float> out(static_cast<size_t>(obs_dim_), 0.0f);

  for (int ti = 0; ti < num_terms(); ++ti) {
    const TermBuffer& b = buffers_[static_cast<size_t>(ti)];
    const int base = term_offsets_[ti];
    if (!b.seen) continue;

    for (int k = 0; k < b.H; ++k) {
      const int dst = base + k * b.dim;
      if (k >= b.count) continue;

      int idx = b.head - k;
      if (idx < 0) idx += b.H;

      const float* src = b.ring.data() + idx * b.dim;
      std::copy(src, src + b.dim, out.begin() + dst);
    }
  }

  return out;
}

std::vector<float> ObservationAssembler::AssembleTimeMajor() const {
  std::vector<float> out(static_cast<size_t>(obs_dim_), 0.0f);

  for (int k = 0; k < global_H_; ++k) {
    const int frame_base = k * frame_dim_;

    for (int ti = 0; ti < num_terms(); ++ti) {
      const TermBuffer& b = buffers_[static_cast<size_t>(ti)];
      const int dst = frame_base + term_in_frame_offsets_[ti];
      if (!b.seen) continue;
      if (k >= b.count) continue;

      int idx = b.head - k;
      if (idx < 0) idx += b.H;

      const float* src = b.ring.data() + idx * b.dim;
      std::copy(src, src + b.dim, out.begin() + dst);
    }
  }

  return out;
}

void ObservationAssembler::BuildIndex() {
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

void ObservationAssembler::BuildBuffers() {
  buffers_.clear();
  buffers_.reserve(terms_.size());

  for (size_t i = 0; i < terms_.size(); ++i) {
    const ObsTerm& t = *terms_[i].term;

    if (t.history_length <= 0) {
      throw std::runtime_error("ObservationAssembler: term '" + terms_[i].name +
                               "' has invalid history_length=" + std::to_string(t.history_length));
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
      b.scale[static_cast<size_t>(k)] = static_cast<float>(t.scale[static_cast<size_t>(k)]);
    }

    b.ring.assign(static_cast<size_t>(b.H * b.dim), 0.0f);
    b.head = -1;
    b.count = 0;
    b.seen = false;

    buffers_.push_back(std::move(b));
  }
}

void ObservationAssembler::BuildLayout() {
  obs_dim_ = 0;
  per_term_flat_dims_.clear();
  term_offsets_.clear();
  term_in_frame_offsets_.clear();

  const int n = static_cast<int>(terms_.size());
  per_term_flat_dims_.reserve(n);
  term_offsets_.reserve(n);
  term_in_frame_offsets_.reserve(n);

  frame_dim_ = 0;
  global_H_ = -1;

  // compute frame offsets + enforce shared H for ConcatThenStack
  for (int i = 0; i < n; ++i) {
    const TermBuffer& b = buffers_[static_cast<size_t>(i)];

    if (cfg_.stack_method == ObsStackMethod::ConcatThenStack) {
      if (global_H_ < 0) {
        global_H_ = b.H;
      } else if (b.H != global_H_) {
        throw std::runtime_error(
            "ObservationAssembler: ConcatThenStack requires all terms to have the same "
            "history_length, but got mismatch at term '" + names_[i] +
            "': expected H=" + std::to_string(global_H_) +
            ", got H=" + std::to_string(b.H));
      }
    }

    term_in_frame_offsets_.push_back(frame_dim_);
    frame_dim_ += b.dim;
  }

  if (cfg_.stack_method == ObsStackMethod::StackThenConcat) {
    global_H_ = 0;  // not used

    for (int i = 0; i < n; ++i) {
      const TermBuffer& b = buffers_[static_cast<size_t>(i)];
      term_offsets_.push_back(obs_dim_);
      const int flat_dim = b.dim * b.H;
      per_term_flat_dims_.push_back(flat_dim);
      obs_dim_ += flat_dim;
    }
  } else {
    // time-major
    obs_dim_ = frame_dim_ * global_H_;
    term_offsets_.clear();  // final offsets not defined for time-major

    for (int i = 0; i < n; ++i) {
      const TermBuffer& b = buffers_[static_cast<size_t>(i)];
      per_term_flat_dims_.push_back(b.dim * global_H_);
    }
  }
}

void ObservationAssembler::CheckTermIndex(int idx) const {
  if (idx < 0 || idx >= num_terms()) {
    throw std::runtime_error("Invalid term_idx=" + std::to_string(idx) +
                             ", num_terms=" + std::to_string(num_terms()));
  }
}

void ObservationAssembler::RequireRawDim(int term_idx, size_t raw_size) const {
  const int expected = buffers_[static_cast<size_t>(term_idx)].dim;
  if (static_cast<int>(raw_size) != expected) {
    throw std::runtime_error("PushRaw(): dim mismatch for term '" + names_[term_idx] +
                             "': expected " + std::to_string(expected) +
                             ", got " + std::to_string(raw_size));
  }
}

}  // namespace legged_rl_deploy
