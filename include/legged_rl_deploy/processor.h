#pragma once
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace legged_rl_deploy {

class Processor {
public:
  enum class Op { Clip, Scale, Offset };

  // ---------------- load ----------------
  static std::optional<Processor> TryLoad(const YAML::Node& node) {
    if (!Present(node) || !node.IsMap()) return std::nullopt;

    const YAML::Node po = node["process_order"];
    if (!Present(po)) return std::nullopt;            // e.g. last_action
    if (!po.IsSequence())
      throw std::runtime_error("process_order must be a sequence");

    Processor p;
    for (auto it : po) p.order_.push_back(ParseOp(it.as<std::string>()));

    std::unordered_set<Op> seen;
    for (auto op : p.order_) {
      if (seen.count(op)) continue;
      seen.insert(op);

      if (op == Op::Clip) {
        p.min_ = Load(node["min"]);
        p.max_ = Load(node["max"]);
        if (p.min_.empty() || p.max_.empty() || p.min_.size() != p.max_.size())
          throw std::runtime_error("clip requires min/max with same dim");
      } else if (op == Op::Scale) {
        p.scale_ = Load(node["scale"]);
        if (p.scale_.empty())
          throw std::runtime_error("scale requires scale");
      } else { // Offset
        p.offset_ = Load(node["offset"]);
        if (p.offset_.empty())
          throw std::runtime_error("offset requires offset");
      }
    }
    return p;
  }

  // ---------------- process ----------------
  void process(std::vector<float>& x) const {
    const std::size_t dim = x.size();

    for (auto op : order_) {
      if (op == Op::Offset) {
        for (std::size_t i = 0; i < dim; ++i)
          x[i] += Pick(offset_, i, dim, "offset");
      } 
      else if (op == Op::Clip) {
        for (std::size_t i = 0; i < dim; ++i)
          x[i] = Clamp(x[i],
                       Pick(min_, i, dim, "min"),
                       Pick(max_, i, dim, "max"));
      } 
      else { // Scale
        for (std::size_t i = 0; i < dim; ++i)
          x[i] *= Pick(scale_, i, dim, "scale");
      }
    }
  }

private:
  // data
  std::vector<Op> order_;
  std::vector<float> min_, max_, scale_, offset_;

  // helpers (minimal)
  static bool Present(const YAML::Node& n) {
    return n && n.IsDefined() && !n.IsNull();
  }

  static Op ParseOp(const std::string& s) {
    if (s == "clip")   return Op::Clip;
    if (s == "scale")  return Op::Scale;
    if (s == "offset") return Op::Offset;
    throw std::runtime_error("unknown preprocess op: " + s);
  }

  static std::vector<float> Load(const YAML::Node& n) {
    if (!Present(n)) return {};
    if (n.IsScalar())  return {n.as<float>()};
    if (n.IsSequence()) return n.as<std::vector<float>>();
    throw std::runtime_error("expected scalar or sequence");
  }

  static float Pick(const std::vector<float>& v,
                    std::size_t i,
                    std::size_t dim,
                    const char* name) {
    if (v.size() == 1)   return v[0];
    if (v.size() == dim) return v[i];
    throw std::runtime_error(
        std::string(name) + " dim mismatch (got " +
        std::to_string(v.size()) + ", expected 1 or " +
        std::to_string(dim) + ")");
  }

  static float Clamp(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
  }
};

} // namespace legged_rl_deploy
