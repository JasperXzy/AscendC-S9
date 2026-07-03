
#include "greater_tiling.h"
#include "register/op_def_registry.h"

#include <algorithm>
#include <cstdint>

namespace {
constexpr uint32_t kDefaultBlockDim = 8;

int64_t GetRightAlignedDim(const gert::Shape& shape, size_t out_rank, size_t out_axis)
{
    const size_t rank = shape.GetDimNum();
    const size_t leading_ones = out_rank - rank;
    if (out_axis < leading_ones) {
        return 1;
    }
    return shape.GetDim(out_axis - leading_ones);
}

ge::graphStatus BroadcastShape(const gert::Shape& self_shape, const gert::Shape& other_shape,
                               int64_t out_shape[optiling::GREATER_MAX_DIMS], uint32_t& rank,
                               uint64_t& total_size)
{
    const size_t self_rank = self_shape.GetDimNum();
    const size_t other_rank = other_shape.GetDimNum();
    const size_t out_rank = std::max(self_rank, other_rank);
    if (out_rank > optiling::GREATER_MAX_DIMS) {
        return ge::GRAPH_FAILED;
    }

    rank = static_cast<uint32_t>(out_rank);
    total_size = 1;
    for (size_t axis = 0; axis < out_rank; ++axis) {
        const int64_t self_dim = GetRightAlignedDim(self_shape, out_rank, axis);
        const int64_t other_dim = GetRightAlignedDim(other_shape, out_rank, axis);
        if (self_dim < 0 || other_dim < 0) {
            return ge::GRAPH_FAILED;
        }

        int64_t dim = 0;
        if (self_dim == other_dim) {
            dim = self_dim;
        } else if (self_dim == 1) {
            dim = other_dim;
        } else if (other_dim == 1) {
            dim = self_dim;
        } else {
            return ge::GRAPH_FAILED;
        }

        out_shape[axis] = dim;
        total_size *= static_cast<uint64_t>(dim);
    }
    return ge::GRAPH_SUCCESS;
}

void BuildBroadcastStride(const gert::Shape& shape, const int64_t out_shape[optiling::GREATER_MAX_DIMS],
                          uint32_t rank, int64_t stride[optiling::GREATER_MAX_DIMS])
{
    int64_t compact_stride[optiling::GREATER_MAX_DIMS] = {};
    const size_t input_rank = shape.GetDimNum();
    int64_t running_stride = 1;
    for (size_t i = input_rank; i > 0; --i) {
        const size_t axis = i - 1;
        compact_stride[axis] = running_stride;
        running_stride *= shape.GetDim(axis);
    }

    const size_t leading_ones = static_cast<size_t>(rank) - input_rank;
    for (size_t axis = 0; axis < rank; ++axis) {
        if (axis < leading_ones) {
            stride[axis] = 0;
            continue;
        }

        const size_t input_axis = axis - leading_ones;
        const int64_t input_dim = shape.GetDim(input_axis);
        stride[axis] = (input_dim == 1 && out_shape[axis] != 1) ? 0 : compact_stride[input_axis];
    }
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* self_storage_shape = context->GetInputShape(0);
    const gert::StorageShape* other_storage_shape = context->GetInputShape(1);
    auto raw_tiling_data = context->GetRawTilingData();
    if (self_storage_shape == nullptr || other_storage_shape == nullptr || raw_tiling_data == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& self_shape = self_storage_shape->GetStorageShape();
    const gert::Shape& other_shape = other_storage_shape->GetStorageShape();

    int64_t out_shape[GREATER_MAX_DIMS] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t self_stride[GREATER_MAX_DIMS] = {};
    int64_t other_stride[GREATER_MAX_DIMS] = {};
    uint32_t rank = 0;
    uint64_t total_size = 1;
    auto ret = BroadcastShape(self_shape, other_shape, out_shape, rank, total_size);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    BuildBroadcastStride(self_shape, out_shape, rank, self_stride);
    BuildBroadcastStride(other_shape, out_shape, rank, other_stride);

    uint32_t block_dim = kDefaultBlockDim;
    if (total_size == 0) {
        block_dim = 1;
    } else if (total_size < kDefaultBlockDim) {
        block_dim = static_cast<uint32_t>(total_size);
    }

    ret = context->SetBlockDim(block_dim);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    GreaterTilingData tiling;
    tiling.set_totalSize(total_size);
    tiling.set_rank(rank);
    tiling.set_blockDim(block_dim);
    tiling.set_outShape(out_shape);
    tiling.set_selfStride(self_stride);
    tiling.set_otherStride(other_stride);
    tiling.SaveToBuffer(raw_tiling_data->GetData(), raw_tiling_data->GetCapacity());
    raw_tiling_data->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    const gert::Shape* self_shape = context->GetInputShape(0);
    const gert::Shape* other_shape = context->GetInputShape(1);
    gert::Shape* output_shape = context->GetOutputShape(0);
    if (self_shape == nullptr || other_shape == nullptr || output_shape == nullptr) {
        return GRAPH_FAILED;
    }

    int64_t out_shape[optiling::GREATER_MAX_DIMS] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint32_t rank = 0;
    uint64_t total_size = 1;
    auto ret = BroadcastShape(*self_shape, *other_shape, out_shape, rank, total_size);
    if (ret != GRAPH_SUCCESS) {
        return ret;
    }

    output_shape->SetDimNum(rank);
    for (uint32_t axis = 0; axis < rank; ++axis) {
        output_shape->SetDim(axis, out_shape[axis]);
    }
    return GRAPH_SUCCESS;
}
}


namespace ops {
class Greater : public OpDef {
public:
    explicit Greater(const char* name) : OpDef(name)
    {
        this->Input("self")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("other")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Greater);
}
