#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "legged_rl_deploy/deploy_config.h"

namespace legged_rl_deploy {

/**
 * ObservationAssembler
 *
 * You push each observation term every tick, it keeps per-term history (ring buffers),
 * and outputs one flattened float vector for the policy.
 *
 * Layouts:
 * 1) StackThenConcat (default, term-major; backward compatible)
 *    obs = [ term1(t..t-H1+1), term2(t..t-H2+1), ... ]  (cfg order)
 *    - Each term is a contiguous slice in the final obs.
 *    - history_length can be term-specific.
 *
 * 2) ConcatThenStack (time-major)
 *    frame(t) = [ term1(t), term2(t), ... ] (cfg order)
 *    obs      = [ frame(t), frame(t-1), ..., frame(t-H+1) ]
 *    - Requires ALL terms to have the SAME history_length H, otherwise throws.
 *    - Terms are interleaved across frames (NOT contiguous slices in final obs).
 */
class ObservationAssembler {
public:
  explicit ObservationAssembler(const ObservationsConfig& cfg,
                              int expected_obs_dim = -1);

  // ---- meta ----
  int obs_dim() const { return obs_dim_; }
  int num_terms() const { return static_cast<int>(terms_.size()); }

  const std::vector<std::string>& term_names() const { return names_; }

  // - StackThenConcat: dim_i * H_i
  // - ConcatThenStack: dim_i * H (shared)
  const std::vector<int>& per_term_flat_dims() const { return per_term_flat_dims_; }

  // StackThenConcat only: term offsets in FINAL observation (contiguous slices).
  // ConcatThenStack: returns empty (use term_offsets_in_frame()).
  const std::vector<int>& term_offsets() const { return term_offsets_; }

  // Offsets inside ONE frame (cfg order). Always available.
  const std::vector<int>& term_offsets_in_frame() const { return term_in_frame_offsets_; }

  // frame_dim = sum(dim_i)
  int frame_dim() const { return frame_dim_; }

  // ConcatThenStack only: shared history length H. StackThenConcat returns 0.
  int shared_history_length() const {
    return (cfg_.stack_method == ObsStackMethod::ConcatThenStack) ? global_H_ : 0;
  }

  const std::unordered_map<std::string, int>& name_to_index() const { return name2idx_; }

  std::unordered_map<std::string, int> PerTermFlatDimMap() const;

  // throws if obs_dim != expected_dim, prints a useful breakdown
  void ValidateExpectedDim(int expected_dim) const;

  // ---- state ----
  void Reset();

  // ---- push ----
  void PushRawByIndex(int term_idx, const std::vector<double>& raw);
  void PushProcessedByIndex(int term_idx, const std::vector<float>& vec);
  void PushRaw(const std::string& term_name, const std::vector<double>& raw);
  void PushProcessed(const std::string& term_name, const std::vector<float>& vec);

  // ---- assemble ----
  std::vector<float> Assemble(bool require_all_terms = true) const;

private:
  // NOTE: These MUST be fully defined in header because we store std::vector<TermInfo/TermBuffer>.
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

  void BuildIndex();
  void BuildBuffers();
  void BuildLayout();

  void CheckTermIndex(int idx) const;
  void RequireRawDim(int term_idx, size_t raw_size) const;

  std::vector<float> AssembleTermMajor() const;
  std::vector<float> AssembleTimeMajor() const;

private:
  const ObservationsConfig& cfg_;

  // ordered terms (same order as cfg_.terms)
  std::vector<TermInfo> terms_;
  std::vector<std::string> names_;
  std::unordered_map<std::string, int> name2idx_;

  // buffers per term, same order
  std::vector<TermBuffer> buffers_;

  // layout meta
  int obs_dim_ = 0;
  std::vector<int> per_term_flat_dims_;

  // StackThenConcat only (final contiguous offsets)
  std::vector<int> term_offsets_;

  // frame info (always computed)
  int frame_dim_ = 0;
  std::vector<int> term_in_frame_offsets_;

  // ConcatThenStack only
  int global_H_ = 0;

  // temp buffer (avoid alloc each PushRaw)
  mutable std::vector<float> tmp_processed_;
};

}  // namespace legged_rl_deploy
