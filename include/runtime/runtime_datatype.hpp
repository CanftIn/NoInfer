#ifndef NO_INFER_RUNTIME_RUNTIME_DATATYPE_HPP
#define NO_INFER_RUNTIME_RUNTIME_DATATYPE_HPP

// 计算节点属性中的权重类型
enum class RuntimeDataType {
  kTypeUnknown = 0,
  kTypeFloat32 = 1,
  kTypeFloat64 = 2,
  kTypeFloat16 = 3,
  kTypeInt32 = 4,
  kTypeInt64 = 5,
  kTypeInt16 = 6,
  kTypeInt8 = 7,
  kTypeUInt8 = 8,
};

#endif  // NO_INFER_RUNTIME_RUNTIME_DATATYPE_HPP