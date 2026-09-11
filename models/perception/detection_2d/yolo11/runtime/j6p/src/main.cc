#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"

namespace {

constexpr int kInputSize = 640;
constexpr int kClassCount = 80;
constexpr int kRegMax = 16;
constexpr int kRaw6OutputCount = 6;
constexpr int kMaxNmsCandidates = 30000;

struct Options {
  std::string model_path;
  std::string image_path;
  std::string output_path;
  std::string dump_dir;
  float confidence = 0.25F;
  float iou_threshold = 0.7F;
  int max_detections = 300;
  int warmup = 0;
  int iterations = 1;
};

struct LetterboxInfo {
  cv::Mat image;
  int original_width = 0;
  int original_height = 0;
  float ratio = 1.0F;
  int pad_left = 0;
  int pad_top = 0;
};

struct Detection {
  float x1 = 0.0F;
  float y1 = 0.0F;
  float x2 = 0.0F;
  float y2 = 0.0F;
  float score = 0.0F;
  int class_id = -1;
};

struct TensorView {
  const hbDNNTensor* tensor = nullptr;
  int channel_axis = -1;
  int height_axis = -1;
  int width_axis = -1;
  int channels = 0;
  int height = 0;
  int width = 0;
};

struct RuntimeMetrics {
  int warmup = 0;
  int iterations = 0;
  double inference_ms = 0.0;
  double postprocess_ms = 0.0;
  double total_ms = 0.0;
};

/** 检查 Horizon API 返回值并在失败时抛出异常. */
void CheckStatus(int32_t status, const std::string& operation) {
  if (status != 0) {
    throw std::runtime_error(operation + "失败, 错误码=" + std::to_string(status));
  }
}

/** 将字符串解析为严格的浮点数. */
float ParseFloat(const std::string& value, const std::string& name) {
  std::size_t parsed = 0;
  const float result = std::stof(value, &parsed);
  if (parsed != value.size() || !std::isfinite(result)) {
    throw std::invalid_argument(name + "不是有效浮点数: " + value);
  }
  return result;
}

/** 将字符串解析为严格的整数. */
int ParseInt(const std::string& value, const std::string& name) {
  std::size_t parsed = 0;
  const int result = std::stoi(value, &parsed);
  if (parsed != value.size()) {
    throw std::invalid_argument(name + "不是有效整数: " + value);
  }
  return result;
}

/** 解析命令行参数. */
Options ParseArgs(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto require_value = [&](const std::string& name) -> std::string {
      if (++index >= argc) {
        throw std::invalid_argument("参数缺少值: " + name);
      }
      return argv[index];
    };
    if (argument == "--model") {
      options.model_path = require_value(argument);
    } else if (argument == "--image") {
      options.image_path = require_value(argument);
    } else if (argument == "--output") {
      options.output_path = require_value(argument);
    } else if (argument == "--dump-dir") {
      options.dump_dir = require_value(argument);
    } else if (argument == "--conf") {
      options.confidence = ParseFloat(require_value(argument), argument);
    } else if (argument == "--iou") {
      options.iou_threshold = ParseFloat(require_value(argument), argument);
    } else if (argument == "--max-det") {
      options.max_detections = ParseInt(require_value(argument), argument);
    } else if (argument == "--warmup") {
      options.warmup = ParseInt(require_value(argument), argument);
    } else if (argument == "--iterations") {
      options.iterations = ParseInt(require_value(argument), argument);
    } else if (argument == "--help" || argument == "-h") {
      std::cout << "用法: yolo11_j6p --model MODEL.hbm --image IMAGE "
                   "--output detections.json [--conf 0.25] [--iou 0.7] "
                   "[--max-det 300] [--warmup 0] [--iterations 1] "
                   "[--dump-dir DIR]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("未知参数: " + argument);
    }
  }
  if (options.model_path.empty() || options.image_path.empty() ||
      options.output_path.empty()) {
    throw std::invalid_argument("--model、--image 和 --output 均为必需参数.");
  }
  if (options.confidence < 0.0F || options.confidence > 1.0F ||
      options.iou_threshold < 0.0F || options.iou_threshold > 1.0F ||
      options.max_detections <= 0 || options.warmup < 0 ||
      options.iterations <= 0) {
    throw std::invalid_argument(
        "阈值必须位于 [0, 1], max-det/iterations 必须大于 0, warmup 不能小于 0.");
  }
  return options;
}

/** 使用 Python round 的 ties-to-even 语义进行尺寸取整. */
int PythonRound(double value) {
  return static_cast<int>(std::nearbyint(value));
}

/** 按当前 Python 评估链路执行 LetterBox. */
LetterboxInfo PrepareImage(const std::string& image_path) {
  cv::Mat source = cv::imread(image_path, cv::IMREAD_COLOR);
  if (source.empty()) {
    throw std::runtime_error("图片读取失败: " + image_path);
  }

  LetterboxInfo info;
  info.original_width = source.cols;
  info.original_height = source.rows;
  info.ratio = std::min(static_cast<float>(kInputSize) / source.rows,
                        static_cast<float>(kInputSize) / source.cols);
  const int resized_width = PythonRound(source.cols * info.ratio);
  const int resized_height = PythonRound(source.rows * info.ratio);

  cv::Mat resized;
  if (resized_width != source.cols || resized_height != source.rows) {
    cv::resize(source, resized, cv::Size(resized_width, resized_height), 0.0,
               0.0, cv::INTER_LINEAR);
  } else {
    resized = source;
  }

  const int padding_width = kInputSize - resized_width;
  const int padding_height = kInputSize - resized_height;
  info.pad_left = PythonRound(padding_width / 2.0 - 0.1);
  const int pad_right = PythonRound(padding_width / 2.0 + 0.1);
  info.pad_top = PythonRound(padding_height / 2.0 - 0.1);
  const int pad_bottom = PythonRound(padding_height / 2.0 + 0.1);
  cv::copyMakeBorder(resized, info.image, info.pad_top, pad_bottom,
                     info.pad_left, pad_right, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));
  return info;
}

/** 返回张量维度数并检查范围. */
int TensorRank(const hbDNNTensor& tensor) {
  const int rank = tensor.properties.validShape.numDimensions;
  if (rank <= 0 || rank > HB_DNN_TENSOR_MAX_DIMENSIONS) {
    throw std::runtime_error("张量维度数非法: " + std::to_string(rank));
  }
  return rank;
}

/** 按字节 stride 计算四维张量元素地址. */
float* TensorAddress(hbDNNTensor* tensor,
                     const std::array<int, 4>& coordinates) {
  if (TensorRank(*tensor) != 4 ||
      tensor->properties.tensorType != HB_DNN_TENSOR_TYPE_F32) {
    throw std::runtime_error("当前 Runtime 仅支持四维 float32 张量.");
  }
  int64_t offset = 0;
  for (int axis = 0; axis < 4; ++axis) {
    const int dimension = tensor->properties.validShape.dimensionSize[axis];
    if (coordinates[axis] < 0 || coordinates[axis] >= dimension) {
      throw std::out_of_range("张量坐标越界.");
    }
    offset += static_cast<int64_t>(coordinates[axis]) *
              tensor->properties.stride[axis];
  }
  if (offset < 0 || offset + static_cast<int64_t>(sizeof(float)) >
                        tensor->properties.alignedByteSize) {
    throw std::out_of_range("张量 stride 地址越界.");
  }
  auto* bytes = static_cast<std::uint8_t*>(tensor->sysMem.virAddr);
  return reinterpret_cast<float*>(bytes + offset);
}

/** 将 RGB NCHW float32 输入写入 UCP 张量并保留物理 padding. */
void FillInputTensor(const LetterboxInfo& image, hbDNNTensor* tensor) {
  if (TensorRank(*tensor) != 4 ||
      tensor->properties.tensorType != HB_DNN_TENSOR_TYPE_F32) {
    throw std::runtime_error("模型输入必须为四维 float32 张量.");
  }
  const auto& shape = tensor->properties.validShape.dimensionSize;
  if (shape[0] != 1 || shape[1] != 3 || shape[2] != kInputSize ||
      shape[3] != kInputSize) {
    throw std::runtime_error("模型输入必须为 1x3x640x640 NCHW.");
  }
  std::memset(tensor->sysMem.virAddr, 0,
              static_cast<std::size_t>(tensor->properties.alignedByteSize));
  for (int channel = 0; channel < 3; ++channel) {
    const int bgr_channel = 2 - channel;
    for (int row = 0; row < kInputSize; ++row) {
      const auto* pixels = image.image.ptr<cv::Vec3b>(row);
      for (int column = 0; column < kInputSize; ++column) {
        *TensorAddress(tensor, {0, channel, row, column}) =
            static_cast<float>(pixels[column][bgr_channel]) / 255.0F;
      }
    }
  }
}

/** 管理一个 HBM 模型及其 UCP 输入输出内存. */
class J6pModel {
 public:
  /** 加载 HBM 并为全部张量分配缓存内存. */
  explicit J6pModel(const std::string& model_path) {
    const char* model_file = model_path.c_str();
    CheckStatus(hbDNNInitializeFromFiles(&packed_handle_, &model_file, 1),
                "加载 HBM");
    try {
      const char** names = nullptr;
      int model_count = 0;
      CheckStatus(hbDNNGetModelNameList(&names, &model_count, packed_handle_),
                  "读取模型名称");
      if (model_count != 1) {
        throw std::runtime_error("预期 HBM 仅包含一个模型, 实际=" +
                                 std::to_string(model_count));
      }
      model_name_ = names[0];
      CheckStatus(hbDNNGetModelHandle(&model_handle_, packed_handle_, names[0]),
                  "获取模型句柄");
      AllocateTensors();
    } catch (...) {
      FreeTensorMemory();
      hbDNNRelease(packed_handle_);
      packed_handle_ = nullptr;
      throw;
    }
  }

  J6pModel(const J6pModel&) = delete;
  J6pModel& operator=(const J6pModel&) = delete;

  /** 释放张量内存和 HBM 句柄. */
  ~J6pModel() {
    FreeTensorMemory();
    if (packed_handle_ != nullptr) {
      hbDNNRelease(packed_handle_);
    }
  }

  /** 返回唯一输入张量. */
  hbDNNTensor* input() {
    if (inputs_.size() != 1) {
      throw std::runtime_error("YOLO11 Runtime 预期一个输入张量.");
    }
    return &inputs_[0];
  }

  /** 返回六路 Raw6 输出. */
  const std::vector<hbDNNTensor>& outputs() const { return outputs_; }

  /** 提交一次同步推理并完成 Cache 同步. */
  void Infer() {
    for (auto& tensor : inputs_) {
      CheckStatus(hbUCPMemFlush(&tensor.sysMem, HB_SYS_MEM_CACHE_CLEAN),
                  "刷新输入 Cache");
    }

    hbUCPTaskHandle_t task_handle = nullptr;
    CheckStatus(hbDNNInferV2(&task_handle, outputs_.data(), inputs_.data(),
                             model_handle_),
                "创建推理任务");
    try {
      hbUCPSchedParam schedule;
      HB_UCP_INITIALIZE_SCHED_PARAM(&schedule);
      schedule.backend = HB_UCP_BPU_CORE_ANY;
      CheckStatus(hbUCPSubmitTask(task_handle, &schedule), "提交推理任务");
      CheckStatus(hbUCPWaitTaskDone(task_handle, 0), "等待推理任务");
      for (auto& tensor : outputs_) {
        CheckStatus(
            hbUCPMemFlush(&tensor.sysMem, HB_SYS_MEM_CACHE_INVALIDATE),
            "失效输出 Cache");
      }
    } catch (...) {
      hbUCPReleaseTask(task_handle);
      throw;
    }
    CheckStatus(hbUCPReleaseTask(task_handle), "释放推理任务");
  }

  /** 返回 HBM 内模型名称. */
  const std::string& model_name() const { return model_name_; }

 private:
  /** 释放已经成功申请的输入输出内存. */
  void FreeTensorMemory() noexcept {
    for (auto& tensor : inputs_) {
      if (tensor.sysMem.virAddr != nullptr) {
        hbUCPFree(&tensor.sysMem);
        tensor.sysMem.virAddr = nullptr;
      }
    }
    for (auto& tensor : outputs_) {
      if (tensor.sysMem.virAddr != nullptr) {
        hbUCPFree(&tensor.sysMem);
        tensor.sysMem.virAddr = nullptr;
      }
    }
  }

  /** 查询张量属性并分配 UCP 缓存内存. */
  void AllocateTensors() {
    int input_count = 0;
    int output_count = 0;
    CheckStatus(hbDNNGetInputCount(&input_count, model_handle_),
                "读取输入数量");
    CheckStatus(hbDNNGetOutputCount(&output_count, model_handle_),
                "读取输出数量");
    if (input_count != 1 || output_count != kRaw6OutputCount) {
      throw std::runtime_error("模型必须包含一个输入和六个 Raw6 输出.");
    }
    inputs_.resize(input_count);
    outputs_.resize(output_count);
    for (int index = 0; index < input_count; ++index) {
      CheckStatus(hbDNNGetInputTensorProperties(&inputs_[index].properties,
                                                 model_handle_, index),
                  "读取输入属性");
      CheckStatus(hbUCPMallocCached(
                      &inputs_[index].sysMem,
                      inputs_[index].properties.alignedByteSize, 0),
                  "分配输入内存");
    }
    for (int index = 0; index < output_count; ++index) {
      CheckStatus(hbDNNGetOutputTensorProperties(&outputs_[index].properties,
                                                  model_handle_, index),
                  "读取输出属性");
      CheckStatus(hbUCPMallocCached(
                      &outputs_[index].sysMem,
                      outputs_[index].properties.alignedByteSize, 0),
                  "分配输出内存");
    }
  }

  hbDNNPackedHandle_t packed_handle_ = nullptr;
  hbDNNHandle_t model_handle_ = nullptr;
  std::string model_name_;
  std::vector<hbDNNTensor> inputs_;
  std::vector<hbDNNTensor> outputs_;
};

/** 从四维输出属性建立 stride-aware 张量视图. */
TensorView BuildTensorView(const hbDNNTensor& tensor) {
  if (TensorRank(tensor) != 4 ||
      tensor.properties.tensorType != HB_DNN_TENSOR_TYPE_F32) {
    throw std::runtime_error("Raw6 输出必须为四维 float32 张量.");
  }
  TensorView view;
  view.tensor = &tensor;
  const auto& dimensions = tensor.properties.validShape.dimensionSize;
  if (dimensions[1] == 64 || dimensions[1] == 80) {
    view.channel_axis = 1;
  } else if (dimensions[3] == 64 || dimensions[3] == 80) {
    view.channel_axis = 3;
  }
  if (view.channel_axis == -1) {
    throw std::runtime_error("Raw6 输出缺少 64/80 通道轴.");
  }
  view.channels = dimensions[view.channel_axis];
  std::vector<int> spatial_axes;
  for (int axis = 1; axis < 4; ++axis) {
    if (axis != view.channel_axis) {
      spatial_axes.push_back(axis);
    }
  }
  view.height_axis = spatial_axes[0];
  view.width_axis = spatial_axes[1];
  view.height = dimensions[view.height_axis];
  view.width = dimensions[view.width_axis];
  if (dimensions[0] != 1 || view.height != view.width ||
      kInputSize % view.height != 0) {
    throw std::runtime_error("Raw6 输出 Batch 或特征图尺寸非法.");
  }
  return view;
}

/** 从 Raw6 张量读取指定通道与坐标的 float32 值. */
float ReadFeature(const TensorView& view, int channel, int row, int column) {
  std::array<int, 4> coordinates{0, 0, 0, 0};
  coordinates[view.channel_axis] = channel;
  coordinates[view.height_axis] = row;
  coordinates[view.width_axis] = column;
  return *TensorAddress(const_cast<hbDNNTensor*>(view.tensor), coordinates);
}

/** 将六路有效输出按连续 NCHW float32 保存, 用于跨后端逐元素核对. */
void DumpRawOutputs(const std::vector<hbDNNTensor>& outputs,
                    const std::string& dump_dir) {
  if (dump_dir.empty()) {
    return;
  }
  std::filesystem::create_directories(dump_dir);
  std::ofstream manifest(std::filesystem::path(dump_dir) / "manifest.json");
  if (!manifest.is_open()) {
    throw std::runtime_error("无法写入 Raw6 manifest: " + dump_dir);
  }
  manifest << "{\n  \"layout\": \"NCHW\",\n  \"dtype\": \"float32\",\n"
           << "  \"outputs\": [\n";
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    const TensorView view = BuildTensorView(outputs[index]);
    const std::string filename = "output_" + std::to_string(index) + ".bin";
    std::ofstream output(std::filesystem::path(dump_dir) / filename,
                         std::ios::binary);
    if (!output.is_open()) {
      throw std::runtime_error("无法写入 Raw6 输出: " + filename);
    }
    for (int channel = 0; channel < view.channels; ++channel) {
      for (int row = 0; row < view.height; ++row) {
        for (int column = 0; column < view.width; ++column) {
          const float value = ReadFeature(view, channel, row, column);
          output.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
      }
    }
    if (!output.good()) {
      throw std::runtime_error("Raw6 输出写入失败: " + filename);
    }
    manifest << "    {\"index\": " << index << ", \"file\": \""
             << filename << "\", \"shape\": [1, " << view.channels << ", "
             << view.height << ", " << view.width << "]}";
    manifest << (index + 1 == outputs.size() ? "\n" : ",\n");
  }
  manifest << "  ]\n}\n";
  if (!manifest.good()) {
    throw std::runtime_error("Raw6 manifest 写入失败: " + dump_dir);
  }
}

/** 稳定计算 Sigmoid. */
float Sigmoid(float value) {
  if (value >= 0.0F) {
    return 1.0F / (1.0F + std::exp(-value));
  }
  const float exponent = std::exp(value);
  return exponent / (1.0F + exponent);
}

/** 计算两个 XYXY 检测框的 IoU. */
float IntersectionOverUnion(const Detection& lhs, const Detection& rhs) {
  const float left = std::max(lhs.x1, rhs.x1);
  const float top = std::max(lhs.y1, rhs.y1);
  const float right = std::min(lhs.x2, rhs.x2);
  const float bottom = std::min(lhs.y2, rhs.y2);
  const float intersection =
      std::max(0.0F, right - left) * std::max(0.0F, bottom - top);
  const float lhs_area =
      std::max(0.0F, lhs.x2 - lhs.x1) * std::max(0.0F, lhs.y2 - lhs.y1);
  const float rhs_area =
      std::max(0.0F, rhs.x2 - rhs.x1) * std::max(0.0F, rhs.y2 - rhs.y1);
  const float union_area = lhs_area + rhs_area - intersection;
  return union_area > 0.0F ? intersection / union_area : 0.0F;
}

/** 对一个尺度执行 DFL、Sigmoid 和候选框解码. */
void DecodeScale(const TensorView& regression, const TensorView& classification,
                 float confidence, std::vector<Detection>* candidates) {
  if (regression.channels != 64 || classification.channels != 80 ||
      regression.height != classification.height ||
      regression.width != classification.width) {
    throw std::runtime_error("Raw6 同尺度回归与分类张量不匹配.");
  }
  const int feature_size = regression.height;
  const float stride = static_cast<float>(kInputSize / feature_size);
  for (int row = 0; row < feature_size; ++row) {
    for (int column = 0; column < feature_size; ++column) {
      std::array<float, 4> distances{};
      for (int side = 0; side < 4; ++side) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (int bin = 0; bin < kRegMax; ++bin) {
          maximum = std::max(
              maximum,
              ReadFeature(regression, side * kRegMax + bin, row, column));
        }
        float denominator = 0.0F;
        float weighted_sum = 0.0F;
        for (int bin = 0; bin < kRegMax; ++bin) {
          const float weight = std::exp(
              ReadFeature(regression, side * kRegMax + bin, row, column) -
              maximum);
          denominator += weight;
          weighted_sum += weight * static_cast<float>(bin);
        }
        distances[side] = weighted_sum / denominator;
      }

      const float anchor_x = static_cast<float>(column) + 0.5F;
      const float anchor_y = static_cast<float>(row) + 0.5F;
      const float x1 = (anchor_x - distances[0]) * stride;
      const float y1 = (anchor_y - distances[1]) * stride;
      const float x2 = (anchor_x + distances[2]) * stride;
      const float y2 = (anchor_y + distances[3]) * stride;
      for (int class_id = 0; class_id < kClassCount; ++class_id) {
        const float score =
            Sigmoid(ReadFeature(classification, class_id, row, column));
        if (score > confidence) {
          candidates->push_back({x1, y1, x2, y2, score, class_id});
        }
      }
    }
  }
}

/** 将六路 Raw6 输出解码为经过按类别 NMS 的检测框. */
std::vector<Detection> DecodeRaw6(const std::vector<hbDNNTensor>& outputs,
                                  const LetterboxInfo& image,
                                  const Options& options) {
  if (outputs.size() != kRaw6OutputCount) {
    throw std::runtime_error("Raw6 输出数量必须为 6.");
  }
  std::vector<TensorView> views;
  views.reserve(outputs.size());
  for (const auto& output : outputs) {
    views.push_back(BuildTensorView(output));
  }

  std::vector<Detection> candidates;
  for (const int feature_size : {80, 40, 20}) {
    const TensorView* regression = nullptr;
    const TensorView* classification = nullptr;
    for (const auto& view : views) {
      if (view.height != feature_size) {
        continue;
      }
      if (view.channels == 64) {
        regression = &view;
      } else if (view.channels == 80) {
        classification = &view;
      }
    }
    if (regression == nullptr || classification == nullptr) {
      throw std::runtime_error("Raw6 缺少尺度 " +
                               std::to_string(feature_size) + " 的输出.");
    }
    DecodeScale(*regression, *classification, options.confidence,
                &candidates);
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const Detection& lhs, const Detection& rhs) {
              return lhs.score > rhs.score;
            });
  if (candidates.size() > kMaxNmsCandidates) {
    candidates.resize(kMaxNmsCandidates);
  }

  std::vector<Detection> selected;
  selected.reserve(static_cast<std::size_t>(options.max_detections));
  for (const auto& candidate : candidates) {
    bool suppressed = false;
    for (const auto& accepted : selected) {
      if (candidate.class_id == accepted.class_id &&
          IntersectionOverUnion(candidate, accepted) > options.iou_threshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      selected.push_back(candidate);
      if (selected.size() >=
          static_cast<std::size_t>(options.max_detections)) {
        break;
      }
    }
  }
  for (auto& detection : selected) {
    detection.x1 = std::clamp(
        (detection.x1 - image.pad_left) / image.ratio, 0.0F,
        static_cast<float>(image.original_width));
    detection.x2 = std::clamp(
        (detection.x2 - image.pad_left) / image.ratio, 0.0F,
        static_cast<float>(image.original_width));
    detection.y1 = std::clamp(
        (detection.y1 - image.pad_top) / image.ratio, 0.0F,
        static_cast<float>(image.original_height));
    detection.y2 = std::clamp(
        (detection.y2 - image.pad_top) / image.ratio, 0.0F,
        static_cast<float>(image.original_height));
  }
  return selected;
}

/** 对 JSON 字符串进行最小必要转义. */
std::string EscapeJson(const std::string& value) {
  std::ostringstream stream;
  for (const char character : value) {
    if (character == '\\' || character == '"') {
      stream << '\\' << character;
    } else if (character == '\n') {
      stream << "\\n";
    } else if (character == '\r') {
      stream << "\\r";
    } else if (character == '\t') {
      stream << "\\t";
    } else {
      stream << character;
    }
  }
  return stream.str();
}

/** 将检测结果写为稳定、便于比较的 JSON. */
void WriteDetections(const Options& options, const std::string& model_name,
                     const LetterboxInfo& image,
                     const std::vector<Detection>& detections,
                     const RuntimeMetrics& metrics) {
  std::ofstream output(options.output_path);
  if (!output.is_open()) {
    throw std::runtime_error("无法写入结果文件: " + options.output_path);
  }
  output << std::fixed << std::setprecision(7);
  output << "{\n"
         << "  \"model_name\": \"" << EscapeJson(model_name) << "\",\n"
         << "  \"image\": \"" << EscapeJson(options.image_path) << "\",\n"
         << "  \"original_shape\": [" << image.original_height << ", "
         << image.original_width << "],\n"
         << "  \"confidence_threshold\": " << options.confidence << ",\n"
         << "  \"iou_threshold\": " << options.iou_threshold << ",\n"
         << "  \"benchmark\": {\"warmup\": " << metrics.warmup
         << ", \"iterations\": " << metrics.iterations
         << ", \"inference_ms\": " << metrics.inference_ms
         << ", \"postprocess_ms\": " << metrics.postprocess_ms
         << ", \"total_ms\": " << metrics.total_ms << "},\n"
         << "  \"detections\": [\n";
  for (std::size_t index = 0; index < detections.size(); ++index) {
    const auto& detection = detections[index];
    output << "    {\"class_id\": " << detection.class_id
           << ", \"score\": " << detection.score << ", \"xyxy\": ["
           << detection.x1 << ", " << detection.y1 << ", " << detection.x2
           << ", " << detection.y2 << "]}";
    output << (index + 1 == detections.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
  if (!output.good()) {
    throw std::runtime_error("检测结果写入失败: " + options.output_path);
  }
}

/** 执行板端 YOLO11s Raw6 推理完整链路. */
int Run(const Options& options) {
  using Clock = std::chrono::steady_clock;
  std::cout << "[1/6] 读取并预处理图片." << std::endl;
  const LetterboxInfo image = PrepareImage(options.image_path);
  std::cout << "[2/6] 加载 J6P HBM." << std::endl;
  J6pModel model(options.model_path);
  FillInputTensor(image, model.input());
  std::cout << "[3/6] 执行预热, 次数=" << options.warmup << "." << std::endl;
  for (int index = 0; index < options.warmup; ++index) {
    model.Infer();
  }

  std::cout << "[4/6] 执行推理与 Raw6 后处理, 次数=" << options.iterations
            << "." << std::endl;
  RuntimeMetrics metrics;
  metrics.warmup = options.warmup;
  metrics.iterations = options.iterations;
  std::vector<Detection> detections;
  double inference_total_ms = 0.0;
  double postprocess_total_ms = 0.0;
  const auto total_start = Clock::now();
  const int progress_interval = std::max(1, options.iterations / 10);
  for (int index = 0; index < options.iterations; ++index) {
    const auto inference_start = Clock::now();
    model.Infer();
    const auto inference_end = Clock::now();
    detections = DecodeRaw6(model.outputs(), image, options);
    const auto postprocess_end = Clock::now();
    inference_total_ms +=
        std::chrono::duration<double, std::milli>(inference_end -
                                                  inference_start)
            .count();
    postprocess_total_ms +=
        std::chrono::duration<double, std::milli>(postprocess_end -
                                                  inference_end)
            .count();
    if ((index + 1) % progress_interval == 0 ||
        index + 1 == options.iterations) {
      std::cout << "进度: " << index + 1 << "/" << options.iterations
                << std::endl;
    }
  }
  const auto total_end = Clock::now();
  metrics.inference_ms = inference_total_ms / options.iterations;
  metrics.postprocess_ms = postprocess_total_ms / options.iterations;
  metrics.total_ms =
      std::chrono::duration<double, std::milli>(total_end - total_start)
          .count() /
      options.iterations;

  std::cout << "[5/6] 保存可选 Raw6 输出." << std::endl;
  DumpRawOutputs(model.outputs(), options.dump_dir);
  std::cout << "[6/6] 写入检测结果, 数量=" << detections.size() << "."
            << std::endl;
  WriteDetections(options, model.model_name(), image, detections, metrics);
  return 0;
}

}  // namespace

/** 程序入口. */
int main(int argc, char** argv) {
  try {
    return Run(ParseArgs(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "错误: " << error.what() << std::endl;
    return 1;
  }
}
