#include "kernel_operator.h"

constexpr int32_t GREATER_MAX_DIMS = 8;
constexpr uint32_t GREATER_COMPARE_COUNT = 128;
constexpr uint32_t GREATER_COMPARE_PACKED_BYTES = GREATER_COMPARE_COUNT / 8;
constexpr uint64_t GREATER_OUTPUT_CACHE_LINE_BYTES = 64;

__aicore__ inline void GreaterSyncSToV()
{
    AscendC::TEventID eventId = GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(eventId);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventId);
}

__aicore__ inline void GreaterSyncVToS()
{
    AscendC::TEventID eventId = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(eventId);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventId);
}

__aicore__ inline void GreaterFlushOutput(AscendC::GlobalTensor<uint8_t>& outGm)
{
    AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                      AscendC::DcciDst::CACHELINE_OUT>(outGm);
}

template <typename TilingData>
__aicore__ inline void CalcCacheLineBlockRange(const TilingData& tilingData, uint64_t& start, uint64_t& end)
{
    const uint64_t totalSize = tilingData.totalSize;
    const uint64_t blockDim = tilingData.blockDim;
    const uint64_t blockIdx = AscendC::GetBlockIdx();
    const uint64_t totalCacheLines = (totalSize + GREATER_OUTPUT_CACHE_LINE_BYTES - 1) /
                                     GREATER_OUTPUT_CACHE_LINE_BYTES;
    const uint64_t startLine = totalCacheLines * blockIdx / blockDim;
    const uint64_t endLine = totalCacheLines * (blockIdx + 1) / blockDim;

    start = startLine * GREATER_OUTPUT_CACHE_LINE_BYTES;
    end = endLine * GREATER_OUTPUT_CACHE_LINE_BYTES;
    if (start > totalSize) {
        start = totalSize;
    }
    if (end > totalSize) {
        end = totalSize;
    }
}

template <typename TilingData>
__aicore__ inline void CalcBroadcastOffsets(const TilingData& tilingData, uint64_t outputIndex,
                                            uint64_t& selfOffset, uint64_t& otherOffset)
{
    selfOffset = 0;
    otherOffset = 0;
    uint64_t remain = outputIndex;
    const int32_t rank = tilingData.rank > GREATER_MAX_DIMS ? GREATER_MAX_DIMS : static_cast<int32_t>(tilingData.rank);
    for (int32_t axis = rank - 1; axis >= 0; --axis) {
        const uint64_t dim = static_cast<uint64_t>(tilingData.outShape[axis]);
        const uint64_t coord = remain % dim;
        remain /= dim;
        selfOffset += coord * static_cast<uint64_t>(tilingData.selfStride[axis]);
        otherOffset += coord * static_cast<uint64_t>(tilingData.otherStride[axis]);
    }
}

template <typename DataType>
class KernelGreater {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR other, GM_ADDR out, uint64_t totalSize)
    {
        selfGm.SetGlobalBuffer((__gm__ DataType*)self);
        otherGm.SetGlobalBuffer((__gm__ DataType*)other);
        outGm.SetGlobalBuffer((__gm__ uint8_t*)out, totalSize);
    }

    template <typename TilingData>
    __aicore__ inline void Process(const TilingData& tilingData)
    {
        const uint64_t totalSize = tilingData.totalSize;
        const uint64_t blockDim = tilingData.blockDim;
        const uint64_t blockIdx = AscendC::GetBlockIdx();
        if (blockDim == 0 || blockIdx >= blockDim) {
            return;
        }

        uint64_t start = 0;
        uint64_t end = 0;
        CalcCacheLineBlockRange(tilingData, start, end);
        for (uint64_t outputIndex = start; outputIndex < end; ++outputIndex) {
            uint64_t selfOffset = 0;
            uint64_t otherOffset = 0;
            CalcBroadcastOffsets(tilingData, outputIndex, selfOffset, otherOffset);
            const bool value = selfGm.GetValue(selfOffset) > otherGm.GetValue(otherOffset);
            outGm.SetValue(outputIndex, static_cast<uint8_t>(value ? 1 : 0));
        }
        GreaterFlushOutput(outGm);
    }

private:
    AscendC::GlobalTensor<DataType> selfGm;
    AscendC::GlobalTensor<DataType> otherGm;
    AscendC::GlobalTensor<uint8_t> outGm;
};

template <>
class KernelGreater<half> {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR other, GM_ADDR out, uint64_t totalSize)
    {
        selfGm.SetGlobalBuffer((__gm__ half*)self);
        otherGm.SetGlobalBuffer((__gm__ half*)other);
        outGm.SetGlobalBuffer((__gm__ uint8_t*)out, totalSize);
        pipe.InitBuffer(selfBuf, GREATER_COMPARE_COUNT * sizeof(half));
        pipe.InitBuffer(otherBuf, GREATER_COMPARE_COUNT * sizeof(half));
        pipe.InitBuffer(packedBuf, GREATER_COMPARE_PACKED_BYTES * sizeof(uint8_t));
    }

    template <typename TilingData>
    __aicore__ inline void Process(const TilingData& tilingData)
    {
        const uint64_t totalSize = tilingData.totalSize;
        const uint64_t blockDim = tilingData.blockDim;
        const uint64_t blockIdx = AscendC::GetBlockIdx();
        if (blockDim == 0 || blockIdx >= blockDim) {
            return;
        }

        uint64_t start = 0;
        uint64_t end = 0;
        CalcCacheLineBlockRange(tilingData, start, end);
        for (uint64_t outputBase = start; outputBase < end; outputBase += GREATER_COMPARE_COUNT) {
            const uint32_t validCount = static_cast<uint32_t>(
                (end - outputBase) < GREATER_COMPARE_COUNT ? (end - outputBase) : GREATER_COMPARE_COUNT);
            FillInputTiles(tilingData, outputBase, validCount);
            GreaterSyncSToV();
            AscendC::LocalTensor<uint8_t> packedLocal = packedBuf.Get<uint8_t>();
            AscendC::Compare(packedLocal, selfBuf.Get<half>(), otherBuf.Get<half>(), AscendC::CMPMODE::GT,
                             GREATER_COMPARE_COUNT);
            GreaterSyncVToS();
            WritePackedResult(outputBase, validCount, packedLocal);
        }
        GreaterFlushOutput(outGm);
    }

private:
    template <typename TilingData>
    __aicore__ inline void FillInputTiles(const TilingData& tilingData, uint64_t outputBase, uint32_t validCount)
    {
        AscendC::LocalTensor<half> selfLocal = selfBuf.Get<half>();
        AscendC::LocalTensor<half> otherLocal = otherBuf.Get<half>();
        for (uint32_t i = 0; i < GREATER_COMPARE_COUNT; ++i) {
            if (i < validCount) {
                uint64_t selfOffset = 0;
                uint64_t otherOffset = 0;
                CalcBroadcastOffsets(tilingData, outputBase + i, selfOffset, otherOffset);
                selfLocal.SetValue(i, selfGm.GetValue(selfOffset));
                otherLocal.SetValue(i, otherGm.GetValue(otherOffset));
            } else {
                selfLocal.SetValue(i, static_cast<half>(0.0));
                otherLocal.SetValue(i, static_cast<half>(0.0));
            }
        }
    }

    __aicore__ inline void WritePackedResult(uint64_t outputBase, uint32_t validCount,
                                             const AscendC::LocalTensor<uint8_t>& packedLocal)
    {
        for (uint32_t i = 0; i < validCount; ++i) {
            const uint8_t packed = packedLocal.GetValue(i / 8);
            const bool value = ((packed >> (i % 8)) & 1) != 0;
            outGm.SetValue(outputBase + i, static_cast<uint8_t>(value ? 1 : 0));
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> selfBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> otherBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> packedBuf;
    AscendC::GlobalTensor<half> selfGm;
    AscendC::GlobalTensor<half> otherGm;
    AscendC::GlobalTensor<uint8_t> outGm;
};

template <>
class KernelGreater<bfloat16_t> {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR other, GM_ADDR out, uint64_t totalSize)
    {
        selfGm.SetGlobalBuffer((__gm__ bfloat16_t*)self);
        otherGm.SetGlobalBuffer((__gm__ bfloat16_t*)other);
        outGm.SetGlobalBuffer((__gm__ uint8_t*)out, totalSize);
        pipe.InitBuffer(selfBuf, GREATER_COMPARE_COUNT * sizeof(bfloat16_t));
        pipe.InitBuffer(otherBuf, GREATER_COMPARE_COUNT * sizeof(bfloat16_t));
        pipe.InitBuffer(selfFloatBuf, GREATER_COMPARE_COUNT * sizeof(float));
        pipe.InitBuffer(otherFloatBuf, GREATER_COMPARE_COUNT * sizeof(float));
        pipe.InitBuffer(packedBuf, GREATER_COMPARE_PACKED_BYTES * sizeof(uint8_t));
    }

    template <typename TilingData>
    __aicore__ inline void Process(const TilingData& tilingData)
    {
        const uint64_t totalSize = tilingData.totalSize;
        const uint64_t blockDim = tilingData.blockDim;
        const uint64_t blockIdx = AscendC::GetBlockIdx();
        if (blockDim == 0 || blockIdx >= blockDim) {
            return;
        }

        uint64_t start = 0;
        uint64_t end = 0;
        CalcCacheLineBlockRange(tilingData, start, end);
        for (uint64_t outputBase = start; outputBase < end; outputBase += GREATER_COMPARE_COUNT) {
            const uint32_t validCount = static_cast<uint32_t>(
                (end - outputBase) < GREATER_COMPARE_COUNT ? (end - outputBase) : GREATER_COMPARE_COUNT);
            FillInputTiles(tilingData, outputBase, validCount);
            GreaterSyncSToV();
            AscendC::LocalTensor<float> selfFloatLocal = selfFloatBuf.Get<float>();
            AscendC::LocalTensor<float> otherFloatLocal = otherFloatBuf.Get<float>();
            AscendC::Cast(selfFloatLocal, selfBuf.Get<bfloat16_t>(), AscendC::RoundMode::CAST_NONE,
                          GREATER_COMPARE_COUNT);
            AscendC::Cast(otherFloatLocal, otherBuf.Get<bfloat16_t>(), AscendC::RoundMode::CAST_NONE,
                          GREATER_COMPARE_COUNT);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<uint8_t> packedLocal = packedBuf.Get<uint8_t>();
            AscendC::Compare(packedLocal, selfFloatLocal, otherFloatLocal, AscendC::CMPMODE::GT,
                             GREATER_COMPARE_COUNT);
            GreaterSyncVToS();
            WritePackedResult(outputBase, validCount, packedLocal);
        }
        GreaterFlushOutput(outGm);
    }

private:
    template <typename TilingData>
    __aicore__ inline void FillInputTiles(const TilingData& tilingData, uint64_t outputBase, uint32_t validCount)
    {
        AscendC::LocalTensor<bfloat16_t> selfLocal = selfBuf.Get<bfloat16_t>();
        AscendC::LocalTensor<bfloat16_t> otherLocal = otherBuf.Get<bfloat16_t>();
        for (uint32_t i = 0; i < GREATER_COMPARE_COUNT; ++i) {
            if (i < validCount) {
                uint64_t selfOffset = 0;
                uint64_t otherOffset = 0;
                CalcBroadcastOffsets(tilingData, outputBase + i, selfOffset, otherOffset);
                selfLocal.SetValue(i, selfGm.GetValue(selfOffset));
                otherLocal.SetValue(i, otherGm.GetValue(otherOffset));
            } else {
                selfLocal.SetValue(i, static_cast<bfloat16_t>(0.0));
                otherLocal.SetValue(i, static_cast<bfloat16_t>(0.0));
            }
        }
    }

    __aicore__ inline void WritePackedResult(uint64_t outputBase, uint32_t validCount,
                                             const AscendC::LocalTensor<uint8_t>& packedLocal)
    {
        for (uint32_t i = 0; i < validCount; ++i) {
            const uint8_t packed = packedLocal.GetValue(i / 8);
            const bool value = ((packed >> (i % 8)) & 1) != 0;
            outGm.SetValue(outputBase + i, static_cast<uint8_t>(value ? 1 : 0));
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> selfBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> otherBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> selfFloatBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> otherFloatBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> packedBuf;
    AscendC::GlobalTensor<bfloat16_t> selfGm;
    AscendC::GlobalTensor<bfloat16_t> otherGm;
    AscendC::GlobalTensor<uint8_t> outGm;
};

extern "C" __global__ __aicore__ void greater(GM_ADDR self, GM_ADDR other, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    using DataType = DTYPE_SELF;
    KernelGreater<DataType> op;
    op.Init(self, other, out, tiling_data.totalSize);
    op.Process(tiling_data);
}
