
#include "register/tilingdata_base.h"

#include <cstdint>

namespace optiling {

constexpr uint32_t GREATER_MAX_DIMS = 8;

BEGIN_TILING_DATA_DEF(GreaterTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, totalSize);
  TILING_DATA_FIELD_DEF(uint32_t, rank);
  TILING_DATA_FIELD_DEF(uint32_t, blockDim);
  TILING_DATA_FIELD_DEF_ARR(int64_t, 8, outShape);
  TILING_DATA_FIELD_DEF_ARR(int64_t, 8, selfStride);
  TILING_DATA_FIELD_DEF_ARR(int64_t, 8, otherStride);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Greater, GreaterTilingData)
}
