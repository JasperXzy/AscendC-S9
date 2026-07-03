# CANN 8.5 API Usage Notes for Greater

This implementation uses local CANN Community Edition 8.5 documentation under `docs/CANNCommunityEdition`.

## Host Registration

- `OP_ADD`: `API/ascendcopapi/atlasascendc_api_07_0945.md`
- `SetInferShape`: `API/ascendcopapi/atlasascendc_api_07_0950.md`
- `SetTiling`: `API/ascendcopapi/atlasascendc_api_07_0979.md`
- `AddConfig`: `API/ascendcopapi/atlasascendc_api_07_0986.md`

## Host Shape and Tiling

- `InferShapeContext::GetInputShape`: `API/basicdataapi/atlasopapi_07_00095.md`
- `InferShapeContext::GetOutputShape`: `API/basicdataapi/atlasopapi_07_00101.md`
- `Shape::GetDimNum`, `Shape::GetDim`, `Shape::SetDimNum`, `Shape::SetDim`: `API/basicdataapi/atlasopapi_07_00150.md`
- `TilingContext::GetInputShape`: `API/basicdataapi/atlasopapi_07_00224.md`
- `TilingContext::SetBlockDim`: `API/basicdataapi/atlasopapi_07_00236.md`
- `TilingContext::GetRawTilingData`: `API/basicdataapi/atlasopapi_07_00243.md`
- `BEGIN_TILING_DATA_DEF`, `TILING_DATA_FIELD_DEF`, `TILING_DATA_FIELD_DEF_ARR`, `END_TILING_DATA_DEF`: `API/ascendcopapi/atlasascendc_api_07_1005.md`
- `REGISTER_TILING_DATA_CLASS`: `API/ascendcopapi/atlasascendc_api_07_1006.md`
- `SaveToBuffer`, `SetDataSize`, `GetDataSize` host tiling flow: `opdevg/Ascendcopdevg/atlas_ascendc_10_00021.md`

## Kernel

- `GM_ADDR` kernel pointer convention: `opdevg/Ascendcopdevg/atlas_ascendc_10_0014.md`
- `GET_TILING_DATA`: `API/ascendcopapi/atlasascendc_api_07_0214.md`
- `AscendC::GetBlockIdx`: `API/ascendcopapi/atlasascendc_api_07_0185.md`
- `AscendC::GlobalTensor`, `SetGlobalBuffer`, `GetValue`, `SetValue`: `API/ascendcopapi/atlasascendc_api_07_00022.md`
- `AscendC::LocalTensor`, `GetValue`, `SetValue`: `API/ascendcopapi/atlasascendc_api_07_00100.md`
- `AscendC::TBuf` and `TPipe::InitBuffer`: `API/ascendcopapi/atlasascendc_api_07_0110.md`, `API/ascendcopapi/atlasascendc_api_07_0163.md`
- `AscendC::Compare`: `API/ascendcopapi/atlasascendc_api_07_0066.md`
- `AscendC::Cast`: `API/ascendcopapi/atlasascendc_api_07_0073.md`
- `AscendC::SetFlag`, `AscendC::WaitFlag`, `TPipe::FetchEventID`: `API/ascendcopapi/atlasascendc_api_07_0270.md`, `API/ascendcopapi/atlasascendc_api_07_0116.md`
- `AscendC::PipeBarrier`: `API/ascendcopapi/atlasascendc_api_07_0271.md`
- Scalar GM `GetValue`/`SetValue` DataCache behavior: `opdevg/Ascendcopdevg/atlas_ascendc_10_00031.md`
- `AscendC::DataCacheCleanAndInvalid`: `API/ascendcopapi/atlasascendc_api_07_0177.md`
- `bfloat16_t` vector conversion pattern: `opdevg/Ascendcopdevg/atlas_ascendc_10_10003.md`

The kernel uses a scalar GM path for `float`, `int32`, and `int8`. For `half`, it fills small UB tiles and uses `Compare` because direct scalar `half > half` is rejected by the CANN compiler. For `bfloat16_t`, it casts the UB tile to `float`, uses `Compare`, then unpacks the documented `uint8_t` bit-packed result into the bool output tensor. The vector paths add explicit `S_V` and `V_S` synchronization around scalar-filled UB data and scalar unpacking, and use `PipeBarrier<PIPE_V>()` between `Cast` and `Compare` in the bfloat16 path. Because scalar writes to GM go through DataCache, output ranges are split on 64-byte cache-line boundaries and each core calls `DataCacheCleanAndInvalid` after writing its result bytes.
