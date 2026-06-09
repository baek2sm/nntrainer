# L9 Class Declaration Inventory

> Mechanical index generated from maintained source files. It intentionally excludes vendored/generated snapshots such as `subprojects/`, local worktrees, duplicated `json.hpp`, and `third_party/` trees; those are summarized in the global class map as external dependency boundaries.

---

## 1. Coverage

Scanned maintained source files: `1010`

Files with class/struct/enum declarations: `460`

Files without class-like declarations: `550`

Total indexed declarations: `1469`

Declaration-free files are not repeated below because they do not add class
relationships. Vendored/generated dependency snapshots are summarized in
[`09-class-map.md`](09-class-map.md) instead of indexed as nntrainer-owned
classes. The full scanned-file list is in
[`09-source-file-coverage.md`](09-source-file-coverage.md).

| Area | Declarations |
|---|---:|
| `nntrainer` | 715 |
| `Applications` | 551 |
| `test` | 136 |
| `api` | 49 |
| `nntrainer-windows-resource` | 8 |
| `nnstreamer` | 7 |
| `benchmarks` | 3 |

---

## 2. File-level declarations

| File | Line | Symbol | Declaration |
|---|---:|---|---|
| `api/capi/include/nntrainer_internal.h` | 111 | `(anonymous typedef struct)` | `typedef struct {` |
| `api/capi/include/nntrainer_internal.h` | 123 | `(anonymous typedef struct)` | `typedef struct {` |
| `api/capi/include/nntrainer_internal.h` | 135 | `(anonymous typedef struct)` | `typedef struct {` |
| `api/capi/include/nntrainer_internal.h` | 148 | `(anonymous typedef struct)` | `typedef struct {` |
| `api/capi/include/nntrainer_internal.h` | 160 | `(anonymous typedef struct)` | `typedef struct {` |
| `api/capi/src/nntrainer-capi-tizen-feature-check.cpp` | 69 | `_feature_info_s` | `typedef struct _feature_info_s {` |
| `api/ccapi/include/common.h` | 30 | `ExportMethods` | `enum class ExportMethods {` |
| `api/ccapi/include/common.h` | 40 | `ExecutionMode` | `enum class ExecutionMode {` |
| `api/ccapi/include/common.h` | 49 | `LayerComputeEngine` | `enum LayerComputeEngine {` |
| `api/ccapi/include/common.h` | 62 | `ISA` | `enum class ISA {` |
| `api/ccapi/include/dataset.h` | 40 | `DatasetType` | `enum class DatasetType {` |
| `api/ccapi/include/dataset.h` | 50 | `DatasetModeType` | `enum class DatasetModeType {` |
| `api/ccapi/include/dataset.h` | 61 | `Dataset` | `class Dataset {` |
| `api/ccapi/include/functions.h` | 29 | `Function` | `class Function;` |
| `api/ccapi/include/functions.h` | 34 | `TensorNode` | `class TensorNode {` |
| `api/ccapi/include/functions.h` | 55 | `Tensor` | `class Tensor {` |
| `api/ccapi/include/functions.h` | 119 | `Function` | `class Function {` |
| `api/ccapi/include/functions.h` | 171 | `Add` | `class Add : public Function {` |
| `api/ccapi/include/functions.h` | 179 | `Sub` | `class Sub : public Function {` |
| `api/ccapi/include/functions.h` | 187 | `Mul` | `class Mul : public Function {` |
| `api/ccapi/include/functions.h` | 195 | `Div` | `class Div : public Function {` |
| `api/ccapi/include/functions.h` | 203 | `Pow` | `class Pow : public Function {` |
| `api/ccapi/include/layer.h` | 37 | `LayerType` | `enum LayerType {` |
| `api/ccapi/include/layer.h` | 144 | `Layer` | `class Layer {` |
| `api/ccapi/include/model.h` | 34 | `RunLayerContext` | `class RunLayerContext;` |
| `api/ccapi/include/model.h` | 35 | `Tensor` | `class Tensor;` |
| `api/ccapi/include/model.h` | 41 | `Tensor` | `class Tensor; // Forward declaration for graph-based compile` |
| `api/ccapi/include/model.h` | 46 | `RunStats` | `struct RunStats {` |
| `api/ccapi/include/model.h` | 70 | `ModelType` | `enum class ModelType {` |
| `api/ccapi/include/model.h` | 79 | `ReferenceLayersType` | `enum class ReferenceLayersType {` |
| `api/ccapi/include/model.h` | 88 | `ModelFormat` | `enum class ModelFormat {` |
| `api/ccapi/include/model.h` | 110 | `Model` | `class Model {` |
| `api/ccapi/include/optimizer.h` | 30 | `LearningRateScheduler` | `class LearningRateScheduler;` |
| `api/ccapi/include/optimizer.h` | 35 | `OptimizerType` | `enum OptimizerType {` |
| `api/ccapi/include/optimizer.h` | 47 | `Optimizer` | `class Optimizer {` |
| `api/ccapi/include/optimizer.h` | 161 | `LearningRateSchedulerType` | `enum LearningRateSchedulerType {` |
| `api/ccapi/include/optimizer.h` | 173 | `LearningRateScheduler` | `class LearningRateScheduler {` |
| `api/ccapi/include/tensor_api.h` | 40 | `Tensor` | `class Tensor {` |
| `api/ccapi/include/tensor_api.h` | 567 | `Impl` | `struct Impl;` |
| `api/ccapi/include/tensor_api.h` | 591 | `LayerHandle` | `class LayerHandle {` |
| `api/ccapi/include/tensor_dim.h` | 40 | `TensorDim` | `class TensorDim {` |
| `api/ccapi/include/tensor_dim.h` | 48 | `Format` | `enum class Format { NCHW, NHWC };` |
| `api/ccapi/include/tensor_dim.h` | 55 | `DataType` | `enum class DataType {` |
| `api/ccapi/include/tensor_dim.h` | 76 | `StorageOrder` | `enum class StorageOrder { ROW_MAJOR, COL_MAJOR };` |
| `api/ccapi/include/tensor_dim.h` | 82 | `TensorType` | `struct TensorType {` |
| `api/ccapi/src/tensor_api_graph.cpp` | 249 | `LayerInfo` | `struct LayerInfo {` |
| `api/ccapi/src/tensor_api_graph.cpp` | 269 | `LeafInfo` | `struct LeafInfo {` |
| `api/ccapi/src/tensor_api_impl.h` | 39 | `SymbolicGraphNode` | `struct SymbolicGraphNode {` |
| `api/ccapi/src/tensor_api_impl.h` | 50 | `Tensor` | `struct Tensor::Impl {` |
| `Applications/Android/kotlin/app/src/main/java/com/samsung/sr/nntr/nntrainer/MainActivity.kt` | 13 | `MainActivity` | `class MainActivity : ComponentActivity() {` |
| `Applications/Android/kotlin/app/src/main/java/com/samsung/sr/nntr/nntrainer/Trainer.kt` | 6 | `NNTrainerNative` | `class NNTrainerNative {` |
| `Applications/Android/kotlin/app/src/main/java/com/samsung/sr/nntr/nntrainer/Trainer.kt` | 17 | `Trainer` | `class Trainer(private val context: Context ) {` |
| `Applications/Android/NNDetector/app/src/androidTest/java/com/samsung/android/nndetector/ExampleInstrumentedTest.kt` | 17 | `ExampleInstrumentedTest` | `class ExampleInstrumentedTest {` |
| `Applications/Android/NNDetector/app/src/main/java/com/samsung/android/nndetector/activities/CropViewActivity.kt` | 26 | `CropViewActivity` | `class CropViewActivity: ComponentActivity() {` |
| `Applications/Android/NNDetector/app/src/main/java/com/samsung/android/nndetector/activities/InferenceActivity.kt` | 25 | `InferenceActivity` | `class InferenceActivity: AppCompatActivity() {` |
| `Applications/Android/NNDetector/app/src/main/java/com/samsung/android/nndetector/activities/MainActivity.kt` | 42 | `MainActivity` | `class MainActivity : AppCompatActivity() {` |
| `Applications/Android/NNDetector/app/src/main/java/com/samsung/android/nndetector/activities/SplashActivity.kt` | 27 | `SplashActivity` | `class SplashActivity : AppCompatActivity() {` |
| `Applications/Android/NNDetector/app/src/main/java/com/samsung/android/nndetector/activities/TestActivity.kt` | 51 | `TestActivity` | `class TestActivity: AppCompatActivity(){` |
| `Applications/Android/NNDetector/app/src/main/java/com/samsung/android/nndetector/activities/TrainActivity.kt` | 26 | `TrainActivity` | `class TrainActivity: ComponentActivity(){` |
| `Applications/Android/NNDetector/app/src/main/java/com/samsung/android/nndetector/actors/NNImageAnalyzer.kt` | 33 | `NNImageAnalyzer` | `class NNImageAnalyzer(detModelPointer: Long, recModelPointer: Long,` |
| `Applications/Android/NNDetector/app/src/main/java/com/samsung/android/nndetector/actors/NNImageAnalyzer.kt` | 225 | `Result` | `public class Result{` |
| `Applications/Android/NNDetector/app/src/main/java/com/samsung/android/nndetector/actors/NNImageTester.kt` | 16 | `NNImageTester` | `class NNImageTester(` |
| `Applications/Android/NNDetector/app/src/main/java/com/samsung/android/nndetector/actors/NNImageTester.kt` | 85 | `TesterState` | `class TesterState{` |
| `Applications/Android/NNDetector/app/src/main/java/com/samsung/android/nndetector/actors/NNImageTrainer.kt` | 16 | `NNImageTrainer` | `class NNImageTrainer(` |
| `Applications/Android/NNDetector/app/src/main/java/com/samsung/android/nndetector/actors/NNImageTrainer.kt` | 71 | `TrainerState` | `class TrainerState{` |
| `Applications/Android/NNDetector/app/src/main/java/com/samsung/android/nndetector/items/DataShared.kt` | 19 | `DataShared` | `class DataShared {` |
| `Applications/Android/NNDetector/app/src/main/java/com/samsung/android/nndetector/items/InferenceResultDialog.kt` | 18 | `InferenceResultDialog` | `class InferenceResultDialog(context: Context): Dialog(context)  {` |
| `Applications/Android/NNDetector/app/src/main/java/com/samsung/android/nndetector/items/ItemDrawable.kt` | 21 | `ItemDrawable` | `class ItemDrawable(rect: Rect, lbl: String, scre: String, iColor:Int = Color.YELLOW) : Drawable() {` |
| `Applications/Android/NNDetector/app/src/main/java/com/samsung/android/nndetector/items/UserImageDialog.kt` | 19 | `UserImageDialog` | `class UserImageDialog(context: Context, imageUri: Uri?): Dialog(context) {` |
| `Applications/Android/NNDetector/app/src/main/jni/dataloader.h` | 41 | `boundingBoxInfo` | `struct boundingBoxInfo {` |
| `Applications/Android/NNDetector/app/src/main/jni/dataloader.h` | 87 | `DirDataLoader` | `class DirDataLoader {` |
| `Applications/Android/NNDetector/app/src/main/jni/image.h` | 26 | `Image` | `class Image {` |
| `Applications/Android/NNDetector/app/src/main/jni/image.h` | 104 | `ImageFactory` | `class ImageFactory {` |
| `Applications/Android/NNDetector/app/src/test/java/com/samsung/android/nndetector/ExampleUnitTest.kt` | 12 | `ExampleUnitTest` | `class ExampleUnitTest {` |
| `Applications/Android/PicoGPTJNI/app/src/main/java/com/applications/picogptjni/MainActivity.java` | 41 | `MainActivity` | `public class MainActivity extends AppCompatActivity {` |
| `Applications/Android/ResnetJNI/app/src/androidTest/java/com/applications/resnetjni/ExampleInstrumentedTest.java` | 19 | `ExampleInstrumentedTest` | `public class ExampleInstrumentedTest {` |
| `Applications/Android/ResnetJNI/app/src/main/java/com/applications/resnetjni/MainActivity.java` | 40 | `MainActivity` | `public class MainActivity extends AppCompatActivity {` |
| `Applications/Android/ResnetJNI/app/src/main/jni/dataloader.cpp` | 37 | `bmpimage` | `struct bmpimage {` |
| `Applications/Android/ResnetJNI/app/src/main/jni/dataloader.h` | 38 | `DataLoader` | `class DataLoader {` |
| `Applications/Android/ResnetJNI/app/src/main/jni/dataloader.h` | 62 | `RandomDataLoader` | `class RandomDataLoader final : public DataLoader {` |
| `Applications/Android/ResnetJNI/app/src/main/jni/dataloader.h` | 99 | `DirDataLoader` | `class DirDataLoader final : public DataLoader {` |
| `Applications/Android/ResnetJNI/app/src/main/jni/image.h` | 23 | `Image` | `class Image {` |
| `Applications/Android/ResnetJNI/app/src/main/jni/image.h` | 101 | `ImageFactory` | `class ImageFactory {` |
| `Applications/Android/ResnetJNI/app/src/test/java/com/applications/resnetjni/ExampleUnitTest.java` | 12 | `ExampleUnitTest` | `public class ExampleUnitTest {` |
| `Applications/CausalLM/api/causal_lm_api.cpp` | 65 | `RegisteredModel` | `struct RegisteredModel {` |
| `Applications/CausalLM/api/causal_lm_api.cpp` | 220 | `stat` | `struct stat buffer;` |
| `Applications/CausalLM/api/causal_lm_api.h` | 30 | `(anonymous typedef struct)` | `typedef struct {` |
| `Applications/CausalLM/api/causal_lm_api.h` | 73 | `(anonymous typedef struct)` | `typedef struct {` |
| `Applications/CausalLM/api/model_config_internal.h` | 25 | `(anonymous typedef struct)` | `typedef struct {` |
| `Applications/CausalLM/api/model_config_internal.h` | 53 | `(anonymous typedef struct)` | `typedef struct {` |
| `Applications/CausalLM/benchmarks/benchmark_android.py` | 73 | `ConfigModifier` | `class ConfigModifier:` |
| `Applications/CausalLM/chat_template.cpp` | 158 | `ChatTemplate` | `struct ChatTemplate::Impl {` |
| `Applications/CausalLM/chat_template.h` | 28 | `ChatTemplate` | `class ChatTemplate {` |
| `Applications/CausalLM/chat_template.h` | 33 | `Options` | `struct Options {` |
| `Applications/CausalLM/chat_template.h` | 37 | `GenerationPromptMode` | `enum class GenerationPromptMode { Auto, Always, Never };` |
| `Applications/CausalLM/chat_template.h` | 42 | `DeveloperRolePolicy` | `enum class DeveloperRolePolicy { Auto, Preserve, MergeIntoSystem };` |
| `Applications/CausalLM/chat_template.h` | 66 | `Impl` | `struct Impl;` |
| `Applications/CausalLM/factory.h` | 26 | `Factory` | `class Factory {` |
| `Applications/CausalLM/huggingface_tokenizer.cpp` | 16 | `HFTokenizer` | `class HFTokenizer : public Tokenizer {` |
| `Applications/CausalLM/kv_cache_manager.h` | 46 | `KVCacheManager` | `class KVCacheManager {` |
| `Applications/CausalLM/kv_cache_manager.h` | 207 | `LayerCache` | `struct LayerCache {` |
| `Applications/CausalLM/layers/causallm_common_properties.h` | 44 | `MoEActivation` | `class MoEActivation final` |
| `Applications/CausalLM/layers/causallm_common_properties.h` | 53 | `NumExperts` | `class NumExperts : public nntrainer::PositiveIntegerProperty {` |
| `Applications/CausalLM/layers/causallm_common_properties.h` | 62 | `NumExpertsPerToken` | `class NumExpertsPerToken : public nntrainer::PositiveIntegerProperty {` |
| `Applications/CausalLM/layers/causallm_common_properties.h` | 73 | `FeatureSize` | `class FeatureSize : public nntrainer::PositiveIntegerProperty {` |
| `Applications/CausalLM/layers/deberta_attention_layer.cpp` | 70 | `RelativeIndexKey` | `struct RelativeIndexKey {` |
| `Applications/CausalLM/layers/deberta_attention_layer.cpp` | 89 | `RelativeIndexValue` | `struct RelativeIndexValue {` |
| `Applications/CausalLM/layers/deberta_attention_layer.h` | 41 | `MaxRelativePositions` | `class MaxRelativePositions : public nntrainer::Property<unsigned int> {` |
| `Applications/CausalLM/layers/deberta_attention_layer.h` | 51 | `C2P` | `class C2P : public nntrainer::Property<bool> {` |
| `Applications/CausalLM/layers/deberta_attention_layer.h` | 61 | `P2C` | `class P2C : public nntrainer::Property<bool> {` |
| `Applications/CausalLM/layers/deberta_attention_layer.h` | 71 | `MaxPositionEmbeddings` | `class MaxPositionEmbeddings : public nntrainer::PositiveIntegerProperty {` |
| `Applications/CausalLM/layers/deberta_attention_layer.h` | 81 | `ShareAttKey` | `class ShareAttKey : public nntrainer::Property<bool> {` |
| `Applications/CausalLM/layers/deberta_attention_layer.h` | 91 | `RelativeAttention` | `class RelativeAttention : public nntrainer::Property<bool> {` |
| `Applications/CausalLM/layers/deberta_attention_layer.h` | 101 | `PositionBuckets` | `class PositionBuckets : public nntrainer::Property<int> {` |
| `Applications/CausalLM/layers/deberta_attention_layer.h` | 111 | `InputLen` | `class InputLen : public nntrainer::Property<unsigned int> {` |
| `Applications/CausalLM/layers/deberta_attention_layer.h` | 124 | `WIN_EXPORT` | `class WIN_EXPORT DebertaAttentionLayer : public nntrainer::LayerImpl {` |
| `Applications/CausalLM/layers/deberta_attention_layer.h` | 226 | `InputIndex` | `enum InputIndex {` |
| `Applications/CausalLM/layers/deberta_attention_layer.h` | 232 | `OutputIndex` | `enum OutputIndex {` |
| `Applications/CausalLM/layers/deberta_attention_layer.h` | 236 | `AttentionParams` | `enum AttentionParams { cache_key = 0, cache_value = 1, max_params };` |
| `Applications/CausalLM/layers/embedding_layer.cpp` | 26 | `EmbeddingParams` | `enum EmbeddingParams { weight };` |
| `Applications/CausalLM/layers/embedding_normalize_layer.h` | 34 | `WIN_EXPORT` | `class WIN_EXPORT EmbeddingNormalizeLayer : public nntrainer::LayerImpl {` |
| `Applications/CausalLM/layers/embedding_pooling_layer.h` | 36 | `WordEmbeddingDimension` | `class WordEmbeddingDimension : public nntrainer::Property<unsigned int> {` |
| `Applications/CausalLM/layers/embedding_pooling_layer.h` | 46 | `PoolingModeClsToken` | `class PoolingModeClsToken : public nntrainer::Property<bool> {` |
| `Applications/CausalLM/layers/embedding_pooling_layer.h` | 57 | `PoolingModeMeanTokens` | `class PoolingModeMeanTokens : public nntrainer::Property<bool> {` |
| `Applications/CausalLM/layers/embedding_pooling_layer.h` | 68 | `PoolingModeMaxTokens` | `class PoolingModeMaxTokens : public nntrainer::Property<bool> {` |
| `Applications/CausalLM/layers/embedding_pooling_layer.h` | 78 | `PoolingModeMeanSqrtLenTokens` | `class PoolingModeMeanSqrtLenTokens : public nntrainer::Property<bool> {` |
| `Applications/CausalLM/layers/embedding_pooling_layer.h` | 89 | `PoolingModeWeightedMeanTokens` | `class PoolingModeWeightedMeanTokens : public nntrainer::Property<bool> {` |
| `Applications/CausalLM/layers/embedding_pooling_layer.h` | 100 | `PoolingModeLastToken` | `class PoolingModeLastToken : public nntrainer::Property<bool> {` |
| `Applications/CausalLM/layers/embedding_pooling_layer.h` | 111 | `IncludePrompt` | `class IncludePrompt : public nntrainer::Property<bool> {` |
| `Applications/CausalLM/layers/embedding_pooling_layer.h` | 126 | `WIN_EXPORT` | `class WIN_EXPORT EmbeddingPoolingLayer : public nntrainer::LayerImpl {` |
| `Applications/CausalLM/layers/lm_head.cpp` | 28 | `LmHeadParams` | `enum LmHeadParams {` |
| `Applications/CausalLM/layers/mha_core.h` | 55 | `NumHeads_KV` | `class NumHeads_KV : public nntrainer::PositiveIntegerProperty {` |
| `Applications/CausalLM/layers/mha_core.h` | 69 | `SlidingWindow` | `class SlidingWindow : public nntrainer::Property<unsigned int> {` |
| `Applications/CausalLM/layers/mha_core.h` | 80 | `MaxNewTokens` | `class MaxNewTokens : public nntrainer::Property<unsigned int> {` |
| `Applications/CausalLM/layers/mha_core.h` | 91 | `MaxPositionEmbeddings` | `class MaxPositionEmbeddings : public nntrainer::Property<unsigned int> {` |
| `Applications/CausalLM/layers/mha_core.h` | 102 | `RopeTheta` | `class RopeTheta : public nntrainer::Property<unsigned int> {` |
| `Applications/CausalLM/layers/mha_core.h` | 112 | `UseSink` | `class UseSink : public nntrainer::Property<bool> {` |
| `Applications/CausalLM/layers/mha_core.h` | 122 | `AttnLogitSoftcapping` | `class AttnLogitSoftcapping : public nntrainer::Property<float> {` |
| `Applications/CausalLM/layers/mha_core.h` | 133 | `IsCausal` | `class IsCausal : public nntrainer::Property<bool> {` |
| `Applications/CausalLM/layers/mha_core.h` | 145 | `RopeScalingType` | `class RopeScalingType : public nntrainer::Property<std::string> {` |
| `Applications/CausalLM/layers/mha_core.h` | 155 | `RopeScalingFactor` | `class RopeScalingFactor : public nntrainer::Property<float> {` |
| `Applications/CausalLM/layers/mha_core.h` | 166 | `RopeScalingMaxPositionEmbeddings` | `class RopeScalingMaxPositionEmbeddings` |
| `Applications/CausalLM/layers/mha_core.h` | 353 | `INOUT_INDEX` | `enum INOUT_INDEX {` |
| `Applications/CausalLM/layers/mha_core.h` | 366 | `AttentionParams` | `enum AttentionParams {` |
| `Applications/CausalLM/layers/mha_core.h` | 398 | `RopeFreqCache` | `struct RopeFreqCache {` |
| `Applications/CausalLM/layers/qkv_layer.cpp` | 38 | `QKVParams` | `enum QKVParams { Q, K, V };` |
| `Applications/CausalLM/layers/qkv_layer.h` | 33 | `QUnit` | `class QUnit : public nntrainer::PositiveIntegerProperty {` |
| `Applications/CausalLM/layers/qkv_layer.h` | 39 | `KUnit` | `class KUnit : public nntrainer::PositiveIntegerProperty {` |
| `Applications/CausalLM/layers/qkv_layer.h` | 45 | `VUnit` | `class VUnit : public nntrainer::PositiveIntegerProperty {` |
| `Applications/CausalLM/layers/shared_fully_connected_layer.cpp` | 33 | `FCParams` | `enum FCParams { weight, bias };` |
| `Applications/CausalLM/layers/shared_fully_connected_layer.h` | 44 | `SharedMode` | `class SharedMode : public nntrainer::Property<bool> {` |
| `Applications/CausalLM/layers/shared_fully_connected_layer.h` | 54 | `FullInputRange` | `class FullInputRange : public nntrainer::Property<bool> {` |
| `Applications/CausalLM/layers/shared_fully_connected_layer.h` | 68 | `WIN_EXPORT` | `class WIN_EXPORT SharedFullyConnectedLayer : public nntrainer::LayerImpl {` |
| `Applications/CausalLM/layers/tie_word_embedding.cpp` | 31 | `TieWordEmbeddingParams` | `enum TieWordEmbeddingParams {` |
| `Applications/CausalLM/layers/tie_word_embedding.h` | 158 | `mode` | `enum mode { embedding, lm_head };` |
| `Applications/CausalLM/layers/tie_word_embedding.h` | 159 | `mode` | `enum mode mode_;` |
| `Applications/CausalLM/main.cpp` | 78 | `rusage` | `struct rusage usage;` |
| `Applications/CausalLM/models/bert/bert_transformer.h` | 41 | `BertTransformer` | `class BertTransformer : virtual public Transformer {` |
| `Applications/CausalLM/models/bert/multilingual_tinybert_16mb.h` | 33 | `MultilingualTinyBert` | `class MultilingualTinyBert : public BertTransformer {` |
| `Applications/CausalLM/models/deberta_v2/deberta_v2.h` | 25 | `DebertaV2` | `class DebertaV2 : public SentenceTransformer {` |
| `Applications/CausalLM/models/gemma3/embedding_gemma.h` | 25 | `EmbeddingGemma` | `class EmbeddingGemma : public SentenceTransformer, public Gemma3Transformer {` |
| `Applications/CausalLM/models/gemma3/gemma3_causallm.h` | 24 | `Gemma3Transformer` | `class Gemma3Transformer : virtual public Transformer {` |
| `Applications/CausalLM/models/gemma3/gemma3_causallm.h` | 66 | `Gemma3CausalLM` | `class Gemma3CausalLM : public CausalLM, public Gemma3Transformer {` |
| `Applications/CausalLM/models/gpt_oss/gpt_oss_moe_layer.h` | 38 | `WIN_EXPORT` | `class WIN_EXPORT GptOssMoELayer : public nntrainer::LayerImpl {` |
| `Applications/CausalLM/models/gpt_oss/gptoss_causallm.h` | 24 | `GptOssForCausalLM` | `class GptOssForCausalLM : public CausalLM {` |
| `Applications/CausalLM/models/gpt_oss_cached_slim/gpt_oss_moe_layer_cached.h` | 39 | `WIN_EXPORT` | `class WIN_EXPORT CachedSlimGptOssMoELayer : public nntrainer::LayerImpl {` |
| `Applications/CausalLM/models/gpt_oss_cached_slim/gptoss_cached_slim_causallm.h` | 24 | `GptOssCachedSlimCausalLM` | `class GptOssCachedSlimCausalLM : public CausalLM {` |
| `Applications/CausalLM/models/performance_metrics.h` | 25 | `(anonymous typedef struct)` | `typedef struct {` |
| `Applications/CausalLM/models/performance_metrics.h` | 63 | `rusage` | `struct rusage rusage;` |
| `Applications/CausalLM/models/qwen2/qwen2_causallm.h` | 24 | `Qwen2Transformer` | `class Qwen2Transformer : virtual public Transformer {` |
| `Applications/CausalLM/models/qwen2/qwen2_causallm.h` | 42 | `Qwen2CausalLM` | `class Qwen2CausalLM : public CausalLM, public Qwen2Transformer {` |
| `Applications/CausalLM/models/qwen2/qwen2_embedding.h` | 24 | `Qwen2Embedding` | `class Qwen2Embedding : public SentenceTransformer, public Qwen2Transformer {` |
| `Applications/CausalLM/models/qwen3/qwen3_causallm.h` | 24 | `Qwen3Transformer` | `class Qwen3Transformer : virtual public Transformer {` |
| `Applications/CausalLM/models/qwen3/qwen3_causallm.h` | 43 | `Qwen3CausalLM` | `class Qwen3CausalLM : public CausalLM, public Qwen3Transformer {` |
| `Applications/CausalLM/models/qwen3/qwen3_embedding.h` | 25 | `Qwen3Embedding` | `class Qwen3Embedding : public SentenceTransformer, public Qwen3Transformer {` |
| `Applications/CausalLM/models/qwen3_cached_slim_moe/qwen_moe_layer_cached.h` | 44 | `WIN_EXPORT` | `class WIN_EXPORT CachedSlimMoELayer : public nntrainer::LayerImpl {` |
| `Applications/CausalLM/models/qwen3_cached_slim_moe/qwen3_cached_slim_moe_causallm.h` | 25 | `Qwen3CachedSlimMoECausalLM` | `class Qwen3CachedSlimMoECausalLM : public Qwen3CausalLM {` |
| `Applications/CausalLM/models/qwen3_moe/qwen_moe_layer.h` | 43 | `WIN_EXPORT` | `class WIN_EXPORT MoELayer : public nntrainer::LayerImpl {` |
| `Applications/CausalLM/models/qwen3_moe/qwen3_moe_causallm.h` | 25 | `Qwen3MoECausalLM` | `class Qwen3MoECausalLM : public Qwen3CausalLM {` |
| `Applications/CausalLM/models/qwen3_slim_moe/qwen_moe_layer_fsu.h` | 43 | `WIN_EXPORT` | `class WIN_EXPORT SlimMoELayer : public nntrainer::LayerImpl {` |
| `Applications/CausalLM/models/qwen3_slim_moe/qwen3_slim_moe_causallm.h` | 25 | `Qwen3SlimMoECausalLM` | `class Qwen3SlimMoECausalLM : public Qwen3CausalLM {` |
| `Applications/CausalLM/models/timm_vit/timm_vit_transformer.h` | 24 | `TimmViTTransformer` | `class TimmViTTransformer : virtual public Transformer {` |
| `Applications/CausalLM/models/transformer.h` | 64 | `ModelType` | `enum class ModelType { MODEL, CAUSALLM, EMBEDDING, UNKNOWN };` |
| `Applications/CausalLM/tokenizers_c.h` | 19 | `(anonymous typedef struct)` | `typedef struct {` |
| `Applications/CausalLM/tokenizers_cpp.h` | 22 | `Tokenizer` | `class Tokenizer {` |
| `Applications/Custom/LayerPlugin/layer_plugin_common_test.h` | 30 | `LayerPluginCommonTest` | `class LayerPluginCommonTest` |
| `Applications/Custom/mae_loss.h` | 27 | `MaeLossLayer` | `class MaeLossLayer final : public nntrainer::Layer {` |
| `Applications/Custom/momentum.cpp` | 30 | `MomentumParams` | `enum MomentumParams { wm };` |
| `Applications/Custom/momentum.h` | 28 | `PropsM` | `class PropsM : public nntrainer::Property<double> {` |
| `Applications/Custom/momentum.h` | 38 | `Momentum` | `class Momentum final : public nntrainer::Optimizer {` |
| `Applications/Custom/OptimizerPlugin/optimizer_plugin_common_test.h` | 42 | `OptimizerSemantics` | `class OptimizerSemantics` |
| `Applications/Custom/OptimizerPlugin/optimizer_plugin_common_test.h` | 83 | `OptimizerPluginCommonTest` | `class OptimizerPluginCommonTest` |
| `Applications/Custom/pow.cpp` | 28 | `Entry` | `struct Entry {` |
| `Applications/Custom/pow.h` | 29 | `PowLayer` | `class PowLayer final : public nntrainer::Layer {` |
| `Applications/Custom/rnnt_loss.h` | 27 | `RNNTLossLayer` | `class RNNTLossLayer final : public nntrainer::Layer {` |
| `Applications/Layers/PyTorch/Conv.py` | 30 | `CustomDatset` | `class CustomDatset(torch.utils.data.Dataset):` |
| `Applications/Layers/PyTorch/Conv.py` | 44 | `NeuralNetwork` | `class NeuralNetwork(nn.Module):` |
| `Applications/Layers/PyTorch/Linear.py` | 30 | `CustomDatset` | `class CustomDatset(torch.utils.data.Dataset):` |
| `Applications/Layers/PyTorch/Linear.py` | 44 | `NeuralNetwork` | `class NeuralNetwork(nn.Module):` |
| `Applications/Layers/PyTorch/LSTM.py` | 28 | `LSTM` | `class LSTM(nn.Module):` |
| `Applications/Layers/PyTorch/Model_A_Conv.py` | 27 | `CustomDatset` | `class CustomDatset(torch.utils.data.Dataset):` |
| `Applications/Layers/PyTorch/Model_A_Conv.py` | 41 | `NeuralNetwork` | `class NeuralNetwork(nn.Module):` |
| `Applications/Layers/PyTorch/Model_A_Linear.py` | 34 | `CustomDatset` | `class CustomDatset(torch.utils.data.Dataset):` |
| `Applications/Layers/PyTorch/Model_A_Linear.py` | 48 | `NeuralNetwork` | `class NeuralNetwork(nn.Module):` |
| `Applications/Layers/PyTorch/Model_C_Conv.py` | 27 | `CustomDatset` | `class CustomDatset(torch.utils.data.Dataset):` |
| `Applications/Layers/PyTorch/Model_C_Conv.py` | 41 | `NeuralNetwork` | `class NeuralNetwork(nn.Module):` |
| `Applications/Layers/PyTorch/Model_C_Linear.py` | 27 | `CustomDatset` | `class CustomDatset(torch.utils.data.Dataset):` |
| `Applications/Layers/PyTorch/Model_C_Linear.py` | 41 | `NeuralNetwork` | `class NeuralNetwork(nn.Module):` |
| `Applications/Layers/Tensorflow/Conv.py` | 21 | `Model` | `class Model(tf.Module):` |
| `Applications/Layers/Tensorflow/Linear.py` | 21 | `Model` | `class Model(tf.Module):` |
| `Applications/Layers/Tensorflow/LSTM.py` | 21 | `Model` | `class Model(tf.Module):` |
| `Applications/Layers/Tensorflow/Model_A_Conv.py` | 22 | `Model` | `class Model(tf.Module):` |
| `Applications/Layers/Tensorflow/Model_A_Linear.py` | 24 | `Model` | `class Model(tf.Module):` |
| `Applications/Layers/Tensorflow/Model_C_Conv.py` | 24 | `Model` | `class Model(tf.Module):` |
| `Applications/Layers/Tensorflow/Model_C_Linear.py` | 24 | `Model` | `class Model(tf.Module):` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 248 | `length_value_t` | `struct length_value_t {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 276 | `construct_from_pointer_t` | `struct construct_from_pointer_t { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 280 | `fixed_string` | `template <size_t N> struct fixed_string {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 509 | `conditional_helper` | `template <bool> struct conditional_helper;` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 527 | `list` | `template <typename... Ts> struct list { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 529 | `_nothing` | `struct _nothing { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 552 | `list_pop_pair` | `template <typename Front, typename List> struct list_pop_pair {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 566 | `rotate_item` | `template <typename T> struct rotate_item {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 575 | `item_matcher` | `template <typename T> struct item_matcher {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 576 | `not_selected` | `struct not_selected {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 579 | `wrapper` | `template <typename Y> struct wrapper {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 616 | `term` | `template <auto v> struct term {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 622 | `epsilon` | `struct epsilon {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 627 | `empty_stack_symbol` | `struct empty_stack_symbol {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 633 | `accept` | `struct accept { constexpr explicit operator bool() noexcept { return true; } };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 634 | `reject` | `struct reject { constexpr explicit operator bool() noexcept { return false; } };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 637 | `action` | `struct action {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 638 | `action_tag` | `struct action_tag { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 642 | `pop_input` | `struct pop_input {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 643 | `pop_input_tag` | `struct pop_input_tag { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 660 | `anything` | `struct anything {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 666 | `range` | `template <auto A, decltype(A) B> struct range {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 673 | `contains` | `template <auto V, auto... Set> struct contains {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 679 | `set` | `template <auto... Def> struct set {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 689 | `neg_set` | `template <auto... Def> struct neg_set {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 700 | `augment_grammar` | `template <typename Grammar> struct augment_grammar: public Grammar {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 731 | `empty_subject` | `struct empty_subject { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 733 | `empty_actions` | `struct empty_actions {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 740 | `identity` | `template <typename Actions> struct identity: public Actions {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 747 | `ignore_unknown` | `template <typename Actions> struct ignore_unknown: public Actions {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 761 | `decision` | `enum class decision {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 767 | `placeholder` | `struct placeholder { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 772 | `parser` | `template <typename Grammar, ctll::fixed_string input, typename ActionSelector = empty_actions, bool IgnoreUnknownActions = false> struct parser { // in c++20` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 774 | `parser` | `template <typename Grammar, const auto & input, typename ActionSelector = empty_actions, bool IgnoreUnknownActions = false> struct parser {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 790 | `results` | `template <size_t Pos, typename Stack, typename Subject, decision Decision> struct results {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 949 | `pcre` | `struct pcre {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 952 | `a` | `struct a {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 953 | `b` | `struct b {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 954 | `backslash` | `struct backslash {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 955 | `backslash_range` | `struct backslash_range {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 956 | `block` | `struct block {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 957 | `block_name2` | `struct block_name2 {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 958 | `c` | `struct c {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 959 | `class_named_name` | `struct class_named_name {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 960 | `content2` | `struct content2 {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 961 | `content` | `struct content {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 962 | `content_in_capture` | `struct content_in_capture {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 963 | `d` | `struct d {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 964 | `e` | `struct e {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 965 | `f` | `struct f {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 966 | `g` | `struct g {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 967 | `h` | `struct h {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 968 | `hexdec_repeat` | `struct hexdec_repeat {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 969 | `i` | `struct i {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 970 | `j` | `struct j {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 971 | `k` | `struct k {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 972 | `l` | `struct l {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 973 | `m` | `struct m {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 974 | `mod` | `struct mod {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 975 | `mode_switch2` | `struct mode_switch2 {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 976 | `n` | `struct n {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 977 | `number2` | `struct number2 {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 978 | `number` | `struct number {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 979 | `o` | `struct o {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 980 | `p` | `struct p {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 981 | `property_name2` | `struct property_name2 {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 982 | `property_name` | `struct property_name {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 983 | `property_value2` | `struct property_value2 {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 984 | `property_value` | `struct property_value {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 985 | `range` | `struct range {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 986 | `repeat` | `struct repeat {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 987 | `s` | `struct s {}; using _start = s;` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 988 | `set2a` | `struct set2a {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 989 | `set2b` | `struct set2b {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 990 | `string2` | `struct string2 {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 993 | `class_digit` | `struct class_digit: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 994 | `class_horizontal_space` | `struct class_horizontal_space: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 995 | `class_named_alnum` | `struct class_named_alnum: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 996 | `class_named_alpha` | `struct class_named_alpha: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 997 | `class_named_ascii` | `struct class_named_ascii: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 998 | `class_named_blank` | `struct class_named_blank: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 999 | `class_named_cntrl` | `struct class_named_cntrl: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1000 | `class_named_digit` | `struct class_named_digit: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1001 | `class_named_graph` | `struct class_named_graph: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1002 | `class_named_lower` | `struct class_named_lower: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1003 | `class_named_print` | `struct class_named_print: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1004 | `class_named_punct` | `struct class_named_punct: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1005 | `class_named_space` | `struct class_named_space: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1006 | `class_named_upper` | `struct class_named_upper: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1007 | `class_named_word` | `struct class_named_word: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1008 | `class_named_xdigit` | `struct class_named_xdigit: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1009 | `class_non_horizontal_space` | `struct class_non_horizontal_space: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1010 | `class_non_vertical_space` | `struct class_non_vertical_space: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1011 | `class_nondigit` | `struct class_nondigit: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1012 | `class_nonnewline` | `struct class_nonnewline: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1013 | `class_nonspace` | `struct class_nonspace: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1014 | `class_nonword` | `struct class_nonword: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1015 | `class_space` | `struct class_space: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1016 | `class_vertical_space` | `struct class_vertical_space: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1017 | `class_word` | `struct class_word: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1018 | `create_hexdec` | `struct create_hexdec: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1019 | `create_number` | `struct create_number: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1020 | `finish_hexdec` | `struct finish_hexdec: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1021 | `look_finish` | `struct look_finish: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1022 | `make_alternate` | `struct make_alternate: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1023 | `make_atomic` | `struct make_atomic: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1024 | `make_back_reference` | `struct make_back_reference: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1025 | `make_capture` | `struct make_capture: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1026 | `make_capture_with_name` | `struct make_capture_with_name: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1027 | `make_lazy` | `struct make_lazy: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1028 | `make_optional` | `struct make_optional: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1029 | `make_possessive` | `struct make_possessive: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1030 | `make_property` | `struct make_property: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1031 | `make_property_negative` | `struct make_property_negative: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1032 | `make_range` | `struct make_range: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1033 | `make_relative_back_reference` | `struct make_relative_back_reference: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1034 | `make_sequence` | `struct make_sequence: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1035 | `mode_case_insensitive` | `struct mode_case_insensitive: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1036 | `mode_case_sensitive` | `struct mode_case_sensitive: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1037 | `mode_multiline` | `struct mode_multiline: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1038 | `mode_singleline` | `struct mode_singleline: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1039 | `negate_class_named` | `struct negate_class_named: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1040 | `prepare_capture` | `struct prepare_capture: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1041 | `push_assert_begin` | `struct push_assert_begin: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1042 | `push_assert_end` | `struct push_assert_end: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1043 | `push_assert_subject_begin` | `struct push_assert_subject_begin: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1044 | `push_assert_subject_end` | `struct push_assert_subject_end: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1045 | `push_assert_subject_end_with_lineend` | `struct push_assert_subject_end_with_lineend: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1046 | `push_character` | `struct push_character: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1047 | `push_character_alarm` | `struct push_character_alarm: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1048 | `push_character_anything` | `struct push_character_anything: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1049 | `push_character_escape` | `struct push_character_escape: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1050 | `push_character_formfeed` | `struct push_character_formfeed: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1051 | `push_character_newline` | `struct push_character_newline: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1052 | `push_character_null` | `struct push_character_null: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1053 | `push_character_return_carriage` | `struct push_character_return_carriage: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1054 | `push_character_tab` | `struct push_character_tab: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1055 | `push_empty` | `struct push_empty: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1056 | `push_hexdec` | `struct push_hexdec: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1057 | `push_name` | `struct push_name: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1058 | `push_not_word_boundary` | `struct push_not_word_boundary: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1059 | `push_number` | `struct push_number: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1060 | `push_property_name` | `struct push_property_name: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1061 | `push_property_value` | `struct push_property_value: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1062 | `push_word_boundary` | `struct push_word_boundary: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1063 | `repeat_ab` | `struct repeat_ab: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1064 | `repeat_at_least` | `struct repeat_at_least: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1065 | `repeat_exactly` | `struct repeat_exactly: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1066 | `repeat_plus` | `struct repeat_plus: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1067 | `repeat_star` | `struct repeat_star: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1068 | `reset_capture` | `struct reset_capture: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1069 | `set_combine` | `struct set_combine: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1070 | `set_make` | `struct set_make: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1071 | `set_make_negative` | `struct set_make_negative: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1072 | `set_start` | `struct set_start: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1073 | `start_atomic` | `struct start_atomic: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1074 | `start_lookahead_negative` | `struct start_lookahead_negative: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1075 | `start_lookahead_positive` | `struct start_lookahead_positive: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1076 | `start_lookbehind_negative` | `struct start_lookbehind_negative: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1077 | `start_lookbehind_positive` | `struct start_lookbehind_positive: ctll::action {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1443 | `singleline` | `struct singleline { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1444 | `multiline` | `struct multiline { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1446 | `case_sensitive` | `struct case_sensitive { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1447 | `case_insensitive` | `struct case_insensitive { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1452 | `flag_list` | `template <typename... Flags> struct flag_list { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1454 | `flags` | `struct flags {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1541 | `MatchesCharacter` | `template <typename T> class MatchesCharacter {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1560 | `character` | `template <auto V> struct character {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1573 | `negative_set` | `template <typename... Content> struct negative_set {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1579 | `set` | `template <typename... Content> struct set {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1585 | `enumeration` | `template <auto... Cs> struct enumeration : set<character<Cs>...> { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1587 | `negate` | `template <typename... Content> struct negate {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1593 | `char_range` | `template <auto A, auto B> struct char_range {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1671 | `accept` | `struct accept { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1672 | `reject` | `struct reject { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1673 | `start_mark` | `struct start_mark { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1674 | `end_mark` | `struct end_mark { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1675 | `end_cycle_mark` | `struct end_cycle_mark { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1676 | `end_lookahead_mark` | `struct end_lookahead_mark { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1677 | `end_lookbehind_mark` | `struct end_lookbehind_mark { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1678 | `numeric_mark` | `template <size_t Id> struct numeric_mark { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1680 | `any` | `struct any { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1683 | `string` | `template <auto... Str> struct string { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1684 | `select` | `template <typename... Opts> struct select { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1685 | `sequence` | `template <typename... Content> struct sequence { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1686 | `empty` | `struct empty { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1688 | `repeat` | `template <size_t a, size_t b, typename... Content> struct repeat { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1692 | `lazy_repeat` | `template <size_t a, size_t b, typename... Content> struct lazy_repeat { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1696 | `possessive_repeat` | `template <size_t a, size_t b, typename... Content> struct possessive_repeat { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1704 | `capture` | `template <size_t Index, typename... Content> struct capture { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1706 | `capture_with_name` | `template <size_t Index, typename Name, typename... Content> struct capture_with_name { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1708 | `back_reference` | `template <size_t Index> struct back_reference { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1709 | `back_reference_with_name` | `template <typename Name> struct back_reference_with_name { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1711 | `look_start` | `template <typename Type> struct look_start { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1713 | `lookahead_positive` | `template <typename... Content> struct lookahead_positive { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1714 | `lookahead_negative` | `template <typename... Content> struct lookahead_negative { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1716 | `lookbehind_positive` | `template <typename... Content> struct lookbehind_positive { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1717 | `lookbehind_negative` | `template <typename... Content> struct lookbehind_negative { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1719 | `atomic_start` | `struct atomic_start { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1721 | `atomic_group` | `template <typename... Content> struct atomic_group { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1723 | `boundary` | `template <typename... Content> struct boundary { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1724 | `not_boundary` | `template <typename... Content> struct not_boundary { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1729 | `assert_subject_begin` | `struct assert_subject_begin { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1730 | `assert_subject_end` | `struct assert_subject_end { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1731 | `assert_subject_end_line` | `struct assert_subject_end_line{ };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1732 | `assert_line_begin` | `struct assert_line_begin { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1733 | `assert_line_end` | `struct assert_line_end { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1735 | `mode_switch` | `template <typename> struct mode_switch { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1752 | `category` | `enum class category;` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1753 | `property` | `enum class property;` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1754 | `version` | `enum class version : unsigned char;` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1755 | `script` | `enum class script ;` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1756 | `block` | `enum class block;` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1758 | `script_extensions_view` | `struct script_extensions_view {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1761 | `sentinel` | `struct sentinel {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1762 | `iterator` | `struct iterator {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1787 | `numeric_value` | `struct numeric_value {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1822 | `binary_prop` | `enum class binary_prop;` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1848 | `property_name` | `template <auto... Str> struct property_name { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1849 | `property_value` | `template <auto... Str> struct property_value { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1857 | `binary_property` | `template <typename T, T Type> struct binary_property;` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1858 | `property` | `template <typename T, T Type, auto Value> struct property;` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1864 | `binary_property` | `template <uni::detail::binary_prop Property> struct binary_property<uni::detail::binary_prop, Property> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1872 | `property_type` | `enum class property_type {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1878 | `binary_property` | `template <uni::script Script> struct binary_property<uni::script, Script> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1884 | `property` | `template <uni::script Script> struct property<property_type, property_type::script_extension, Script> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1893 | `binary_property` | `template <uni::version Age> struct binary_property<uni::version, Age> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1899 | `binary_property` | `template <uni::block Block> struct binary_property<uni::block, Block> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1923 | `property_type_builder` | `template <property_type Property> struct property_type_builder {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 1929 | `property_builder` | `template <auto... Name> struct property_builder {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 2002 | `rotate_value` | `template <auto V> struct rotate_value {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 2006 | `rotate_for_lookbehind` | `struct rotate_for_lookbehind {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 2093 | `id` | `template <auto... Name> struct id {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 2114 | `pcre_parameters` | `template <size_t Counter> struct pcre_parameters {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 2132 | `number` | `template <size_t Value> struct number { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 2134 | `capture_id` | `template <size_t Id> struct capture_id { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 2136 | `pcre_actions` | `struct pcre_actions {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 3059 | `utf8_iterator` | `struct utf8_iterator {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 3067 | `sentinel` | `struct sentinel {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 3276 | `utf8_range` | `struct utf8_range {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 3316 | `not_matched_tag_t` | `struct not_matched_tag_t { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 3320 | `captured_content` | `template <size_t Id, typename Name = void> struct captured_content {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 3321 | `storage` | `template <typename Iterator> class storage {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 3409 | `identify` | `template <typename T> struct identify;` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 3470 | `capture_not_exists_tag` | `struct capture_not_exists_tag { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 3474 | `captures` | `template <typename... Captures> struct captures;` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 3476 | `captures` | `template <typename Head, typename... Tail> struct captures<Head, Tail...>: captures<Tail...> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 3583 | `regex_results` | `template <typename Iterator, typename... Captures> class regex_results {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 3721 | `tuple_size` | `template <typename... Captures> struct tuple_size<ctre::regex_results<Captures...>> : public std::integral_constant<size_t, ctre::regex_results<Captures...>::count()> { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 3723 | `tuple_element` | `template <size_t N, typename... Captures> struct tuple_element<N, ctre::regex_results<Captures...>> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 3849 | `can_be_anything` | `struct can_be_anything {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 4225 | `point_set` | `template <size_t Capacity> class point_set {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 4226 | `point` | `struct point {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 4809 | `string_match` | `template <typename Iterator> struct string_match {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 4952 | `regex_end_iterator` | `struct regex_end_iterator {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 4956 | `regex_iterator` | `template <typename BeginIterator, typename EndIterator, typename RE, typename ResultIterator = BeginIterator> struct regex_iterator {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5034 | `regex_split_iterator` | `template <typename BeginIterator, typename EndIterator, typename RE, typename ResultIterator = BeginIterator> struct regex_split_iterator {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5140 | `regex_range` | `template <typename BeginIterator, typename EndIterator, typename RE, typename ResultIterator = BeginIterator> struct regex_range {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5156 | `regex_split_range` | `template <typename BeginIterator, typename EndIterator, typename RE, typename ResultIterator = BeginIterator> struct regex_split_range {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5172 | `multi_subject_range` | `template <typename Range, typename RE> struct multi_subject_range {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5173 | `end_iterator` | `struct end_iterator { };` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5178 | `iterator` | `struct iterator {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5282 | `regular_expression` | `template <typename RE, typename Method = void, typename Modifier = singleline> struct regular_expression;` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5284 | `zero_terminated_string_end_iterator` | `struct zero_terminated_string_end_iterator {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5327 | `RangeLikeType` | `template <typename T> class RangeLikeType {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5334 | `match_method` | `struct match_method {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5346 | `search_method` | `struct search_method {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5374 | `starts_with_method` | `struct starts_with_method {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5386 | `range_method` | `struct range_method {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5395 | `tokenize_method` | `struct tokenize_method {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5404 | `split_method` | `struct split_method {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5413 | `iterator_method` | `struct iterator_method {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5425 | `regular_expression` | `template <typename RE, typename Method, typename Modifier> struct regular_expression {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5546 | `problem_at_position` | `template <size_t> struct problem_at_position; // do not define!` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5562 | `regex_builder` | `template <CTRE_REGEX_TEMPLATE_COPY_TYPE input> struct regex_builder {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5748 | `pattern` | `template <typename CharT, size_t N> class pattern: public ctll::fixed_string<N> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5757 | `fixed_string` | `template <typename CharT, size_t N> class fixed_string: public ctll::fixed_string<N> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5808 | `category` | `enum class category;` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5809 | `property` | `enum class property;` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5810 | `version` | `enum class version : unsigned char;` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5811 | `script` | `enum class script;` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5812 | `block` | `enum class block;` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5814 | `script_extensions_view` | `struct script_extensions_view {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5817 | `sentinel` | `struct sentinel {};` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5818 | `iterator` | `struct iterator {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5843 | `numeric_value` | `struct numeric_value {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5877 | `binary_prop` | `enum class binary_prop;` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5967 | `compact_range` | `struct compact_range {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 5986 | `compact_list` | `struct compact_list {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 6004 | `array` | `struct array {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 6009 | `array` | `struct array<T, 0> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 6019 | `bool_trie` | `struct bool_trie {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 6079 | `flat_array` | `struct flat_array {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 6097 | `range_array` | `struct range_array {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 6155 | `pair` | `struct pair {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 6163 | `string_with_idx` | `struct string_with_idx {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 6193 | `version` | `enum class version : uint8_t {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 6221 | `category` | `enum class category {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 6300 | `block` | `enum class block {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 6755 | `script` | `enum class script {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 9831 | `script_data` | `struct script_data;` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 9833 | `script_data` | `struct script_data<0> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 10072 | `script_data` | `struct script_data<1> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 10102 | `script_data` | `struct script_data<2> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 10129 | `script_data` | `struct script_data<3> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 10142 | `script_data` | `struct script_data<4> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 10153 | `script_data` | `struct script_data<5> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 10163 | `script_data` | `struct script_data<6> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 10172 | `script_data` | `struct script_data<7> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 10179 | `script_data` | `struct script_data<8> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 10185 | `script_data` | `struct script_data<9> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 10191 | `script_data` | `struct script_data<10> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 10197 | `script_data` | `struct script_data<11> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 10203 | `script_data` | `struct script_data<12> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 10209 | `script_data` | `struct script_data<13> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 10215 | `script_data` | `struct script_data<14> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 10220 | `script_data` | `struct script_data<15> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 10225 | `script_data` | `struct script_data<16> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 10230 | `script_data` | `struct script_data<17> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 10235 | `script_data` | `struct script_data<18> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 10240 | `script_data` | `struct script_data<19> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 10245 | `script_data` | `struct script_data<20> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 10250 | `script_data` | `struct script_data<21> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 11080 | `property` | `enum class property {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 13322 | `iterator_traits` | `struct iterator_traits<uni::script_extensions_view::iterator> {` |
| `Applications/LLaMA/jni/ctre-unicode.hpp` | 13332 | `binary_prop` | `enum class binary_prop {` |
| `Applications/LLaMA/jni/custom_multi_head_attention_layer.cpp` | 41 | `INOUT_INDEX` | `enum INOUT_INDEX {` |
| `Applications/LLaMA/jni/custom_multi_head_attention_layer.cpp` | 52 | `AttentionParams` | `enum AttentionParams {` |
| `Applications/LLaMA/jni/custom_multi_head_attention_layer.h` | 33 | `MultiHeadAttentionLayer` | `class MultiHeadAttentionLayer : public LayerImpl {` |
| `Applications/LLaMA/jni/encoder.hpp` | 143 | `GPT2Encoder` | `class GPT2Encoder {` |
| `Applications/LLaMA/jni/encoder.hpp` | 148 | `PairHash` | `struct PairHash {` |
| `Applications/LLaMA/jni/rms_norm.h` | 35 | `RMS_NORM_GAMMA_INIT` | `class RMS_NORM_GAMMA_INIT final` |
| `Applications/LLaMA/jni/rms_norm.h` | 52 | `RMSParams` | `enum RMSParams { gamma };` |
| `Applications/LLaMA/jni/rms_norm.h` | 58 | `RMSNormLayer` | `class RMSNormLayer final : public nntrainer::Layer {` |
| `Applications/LLaMA/jni/rotary_embedding.h` | 29 | `RotaryEmbeddingLayer` | `class RotaryEmbeddingLayer final : public nntrainer::Layer {` |
| `Applications/LLaMA/jni/swiglu.h` | 28 | `SwiGLULayer` | `class SwiGLULayer final : public nntrainer::Layer {` |
| `Applications/LLaMA/jni/transpose_layer.h` | 29 | `TransposeLayer` | `class TransposeLayer final : public nntrainer::Layer {` |
| `Applications/MNIST/jni/main.cpp` | 111 | `DataInformation` | `class DataInformation {` |
| `Applications/MNIST/PyTorch/main.py` | 23 | `Net` | `class Net(nn.Module):` |
| `Applications/Multi_input/jni/multi_loader.h` | 27 | `DataLoader` | `class DataLoader {` |
| `Applications/Multi_input/jni/multi_loader.h` | 53 | `MultiDataLoader` | `class MultiDataLoader final : public DataLoader {` |
| `Applications/Optimizers/jni/main.cpp` | 38 | `AppDataset` | `enum class AppDataset { RANDOM, MNIST };` |
| `Applications/Optimizers/jni/main.cpp` | 44 | `AppConfig` | `struct AppConfig {` |
| `Applications/Optimizers/jni/main.cpp` | 79 | `WeightStats` | `struct WeightStats {` |
| `Applications/Optimizers/jni/main.cpp` | 194 | `MnistDataInfo` | `class MnistDataInfo {` |
| `Applications/ReinforcementLearning/DeepQ/jni/main.cpp` | 119 | `(anonymous typedef struct)` | `typedef struct {` |
| `Applications/ReinforcementLearning/Environment/CartPole/cartpole.h` | 41 | `(anonymous typedef struct)` | `typedef struct {` |
| `Applications/ReinforcementLearning/Environment/CartPole/cartpole.h` | 52 | `CartPole` | `class CartPole {` |
| `Applications/Resnet/PyTorch/main.py` | 48 | `BasicBlock` | `class BasicBlock(nn.Module):` |
| `Applications/Resnet/PyTorch/main.py` | 79 | `ResNet` | `class ResNet(nn.Module):` |
| `Applications/SimpleShot/layers/centering.h` | 31 | `CenteringLayer` | `class CenteringLayer : public nntrainer::Layer {` |
| `Applications/SimpleShot/simpleshot_utils.h` | 22 | `Entry` | `struct Entry {` |
| `Applications/Tizen_native/CustomShortcut/inc/data.h` | 58 | `appdata` | `typedef struct appdata {` |
| `Applications/Tizen_native/CustomShortcut/inc/data.h` | 100 | `train_result` | `typedef struct train_result {` |
| `Applications/TransferLearning/CIFAR_Classification/TensorFlow/Training_Keras.py` | 58 | `History_LAW` | `class History_LAW(Callback):` |
| `Applications/utils/datagen/cifar/cifar_dataloader.h` | 26 | `DataLoader` | `class DataLoader {` |
| `Applications/utils/datagen/cifar/cifar_dataloader.h` | 52 | `RandomDataLoader` | `class RandomDataLoader final : public DataLoader {` |
| `Applications/utils/datagen/cifar/cifar_dataloader.h` | 88 | `Cifar100DataLoader` | `class Cifar100DataLoader final : public DataLoader {` |
| `Applications/utils/npy_reader/npy_reader.h` | 25 | `NpyReader` | `class NpyReader {` |
| `Applications/VGG/PyTorch/main.py` | 56 | `VGG` | `class VGG(nn.Module):` |
| `Applications/YOLOv2/jni/det_dataloader.h` | 26 | `DirDataLoader` | `class DirDataLoader {` |
| `Applications/YOLOv2/jni/reorg_layer.h` | 28 | `ReorgLayer` | `class ReorgLayer final : public nntrainer::Layer {` |
| `Applications/YOLOv2/jni/yolo_v2_loss.cpp` | 21 | `YoloV2LossParams` | `enum YoloV2LossParams {` |
| `Applications/YOLOv2/jni/yolo_v2_loss.h` | 32 | `MaxObjectNumber` | `class MaxObjectNumber final : public nntrainer::PositiveIntegerProperty {` |
| `Applications/YOLOv2/jni/yolo_v2_loss.h` | 43 | `ClassNumber` | `class ClassNumber final : public nntrainer::PositiveIntegerProperty {` |
| `Applications/YOLOv2/jni/yolo_v2_loss.h` | 54 | `GridHeightNumber` | `class GridHeightNumber final : public nntrainer::PositiveIntegerProperty {` |
| `Applications/YOLOv2/jni/yolo_v2_loss.h` | 65 | `GridWidthNumber` | `class GridWidthNumber final : public nntrainer::PositiveIntegerProperty {` |
| `Applications/YOLOv2/jni/yolo_v2_loss.h` | 78 | `YoloV2LossLayer` | `class YoloV2LossLayer final : public nntrainer::Layer {` |
| `Applications/YOLOv2/PyTorch/dataset.py` | 22 | `YOLODataset` | `class YOLODataset(Dataset):` |
| `Applications/YOLOv2/PyTorch/yolo.py` | 16 | `YoloV2` | `class YoloV2(nn.Module):` |
| `Applications/YOLOv2/PyTorch/yolo_loss.py` | 64 | `YoloV2_LOSS` | `class YoloV2_LOSS(nn.Module):` |
| `Applications/YOLOv3/jni/det_dataloader.h` | 26 | `DirDataLoader` | `class DirDataLoader {` |
| `Applications/YOLOv3/jni/upsample_layer.h` | 28 | `UpsampleLayer` | `class UpsampleLayer final : public nntrainer::Layer {` |
| `Applications/YOLOv3/jni/yolo_v3_loss.cpp` | 21 | `YoloV3LossParams` | `enum YoloV3LossParams {` |
| `Applications/YOLOv3/jni/yolo_v3_loss.h` | 34 | `MaxObjectNumber` | `class MaxObjectNumber final : public nntrainer::PositiveIntegerProperty {` |
| `Applications/YOLOv3/jni/yolo_v3_loss.h` | 45 | `ClassNumber` | `class ClassNumber final : public nntrainer::PositiveIntegerProperty {` |
| `Applications/YOLOv3/jni/yolo_v3_loss.h` | 56 | `GridHeightNumber` | `class GridHeightNumber final : public nntrainer::PositiveIntegerProperty {` |
| `Applications/YOLOv3/jni/yolo_v3_loss.h` | 67 | `GridWidthNumber` | `class GridWidthNumber final : public nntrainer::PositiveIntegerProperty {` |
| `Applications/YOLOv3/jni/yolo_v3_loss.h` | 78 | `Scale` | `class Scale final : public nntrainer::PositiveIntegerProperty {` |
| `Applications/YOLOv3/jni/yolo_v3_loss.h` | 91 | `YoloV3LossLayer` | `class YoloV3LossLayer final : public nntrainer::Layer {` |
| `Applications/YOLOv3/PyTorch/yolo.py` | 30 | `ConvBlock` | `class ConvBlock(nn.Module):` |
| `Applications/YOLOv3/PyTorch/yolo.py` | 46 | `DarkNetBlock` | `class DarkNetBlock(nn.Module):` |
| `Applications/YOLOv3/PyTorch/yolo.py` | 66 | `Darknet53` | `class Darknet53(nn.Module):` |
| `Applications/YOLOv3/PyTorch/yolo.py` | 142 | `YoloV3` | `class YoloV3(nn.Module):` |
| `Applications/YOLOv3/PyTorch/yolo_loss.py` | 63 | `YoloV3_LOSS` | `class YoloV3_LOSS(nn.Module):` |
| `benchmarks/fake_data_gen/fake_data_gen.h` | 26 | `DataLoader` | `class DataLoader {` |
| `benchmarks/fake_data_gen/fake_data_gen.h` | 52 | `RandomDataLoader` | `class RandomDataLoader final : public DataLoader {` |
| `benchmarks/fake_data_gen/fake_data_gen.h` | 88 | `Cifar100DataLoader` | `class Cifar100DataLoader final : public DataLoader {` |
| `nnstreamer/tensor_filter/tensor_filter_nntrainer.cc` | 64 | `gNewDeletor` | `struct gNewDeletor {` |
| `nnstreamer/tensor_filter/tensor_filter_nntrainer.hh` | 33 | `(anonymous typedef struct)` | `typedef struct {` |
| `nnstreamer/tensor_filter/tensor_filter_nntrainer.hh` | 42 | `NNTrainerInference` | `class NNTrainerInference {` |
| `nnstreamer/tensor_trainer/tensor_trainer_nntrainer.hh` | 31 | `TensorsData` | `struct TensorsData {` |
| `nnstreamer/tensor_trainer/tensor_trainer_nntrainer.hh` | 36 | `TensorsQueue` | `class TensorsQueue;` |
| `nnstreamer/tensor_trainer/tensor_trainer_nntrainer.hh` | 41 | `NNTrainerImpl` | `class NNTrainerImpl {` |
| `nnstreamer/tensor_trainer/tensor_trainer_nntrainer.hh` | 172 | `TensorsQueue` | `class TensorsQueue {` |
| `nntrainer/app_context.h` | 49 | `AppContext` | `class AppContext : public Context, public Singleton<AppContext> {` |
| `nntrainer/app_context.h` | 309 | `isSupportedHelper` | `template <typename Args, typename T> struct isSupportedHelper;` |
| `nntrainer/app_context.h` | 315 | `isSupportedHelper` | `struct isSupportedHelper<T, AppContext::FactoryMap<Args...>> {` |
| `nntrainer/app_context.h` | 324 | `isSupported` | `struct isSupported : isSupportedHelper<T, decltype(factory_map)> {};` |
| `nntrainer/cl_context.h` | 51 | `ClContext` | `class ClContext : public Context, public Singleton<ClContext> {` |
| `nntrainer/cl_context.h` | 260 | `isSupportedHelper` | `template <typename Args, typename T> struct isSupportedHelper;` |
| `nntrainer/cl_context.h` | 269 | `isSupportedHelper` | `struct isSupportedHelper<T, ClContext::FactoryMap<Args...>> {` |
| `nntrainer/cl_context.h` | 278 | `isSupported` | `struct isSupported : isSupportedHelper<T, decltype(factory_map)> {};` |
| `nntrainer/cl_svm_allocator.h` | 23 | `ContextManager` | `class ContextManager;` |
| `nntrainer/cl_svm_allocator.h` | 41 | `ClSVMAllocator` | `class ClSVMAllocator : public MemAllocator {` |
| `nntrainer/compiler/activation_realizer.h` | 27 | `ActivationRealizer` | `class ActivationRealizer final : public GraphRealizer {` |
| `nntrainer/compiler/bn_realizer.h` | 32 | `BnRealizer` | `class BnRealizer final : public GraphRealizer {` |
| `nntrainer/compiler/compiler.h` | 51 | `GraphCompiler` | `class GraphCompiler {` |
| `nntrainer/compiler/compiler_fwd.h` | 19 | `LayerNode` | `class LayerNode;` |
| `nntrainer/compiler/compiler_fwd.h` | 20 | `NetworkGraph` | `class NetworkGraph;` |
| `nntrainer/compiler/flatbuffer_opnode.h` | 25 | `LayerNode` | `class LayerNode;` |
| `nntrainer/compiler/flatbuffer_opnode.h` | 26 | `RunLayerContext` | `class RunLayerContext;` |
| `nntrainer/compiler/flatbuffer_opnode.h` | 32 | `FlatBufferOpNode` | `class FlatBufferOpNode {` |
| `nntrainer/compiler/flatten_realizer.h` | 26 | `FlattenRealizer` | `class FlattenRealizer final : public GraphRealizer {` |
| `nntrainer/compiler/ini_interpreter.cpp` | 64 | `PlainLayer` | `class PlainLayer {};` |
| `nntrainer/compiler/ini_interpreter.cpp` | 69 | `BackboneLayer` | `class BackboneLayer {};` |
| `nntrainer/compiler/ini_interpreter.h` | 30 | `IniGraphInterpreter` | `class IniGraphInterpreter : public GraphInterpreter {` |
| `nntrainer/compiler/input_realizer.h` | 32 | `InputRealizer` | `class InputRealizer final : public GraphRealizer {` |
| `nntrainer/compiler/interpreter.h` | 54 | `GraphInterpreter` | `class GraphInterpreter {` |
| `nntrainer/compiler/multiout_realizer.h` | 28 | `MultioutRealizer` | `class MultioutRealizer final : public GraphRealizer {` |
| `nntrainer/compiler/onnx_interpreter.h` | 38 | `ONNXInterpreter` | `class ONNXInterpreter : public GraphInterpreter {` |
| `nntrainer/compiler/previous_input_realizer.h` | 29 | `PreviousInputRealizer` | `class PreviousInputRealizer final : public GraphRealizer {` |
| `nntrainer/compiler/realizer.h` | 27 | `GraphRealizer` | `class GraphRealizer {` |
| `nntrainer/compiler/recurrent_realizer.cpp` | 39 | `UnrollFor` | `class UnrollFor final : public PositiveIntegerProperty {` |
| `nntrainer/compiler/recurrent_realizer.cpp` | 53 | `DynamicTimeSequence` | `class DynamicTimeSequence final : public nntrainer::Property<bool> {` |
| `nntrainer/compiler/recurrent_realizer.cpp` | 68 | `RecurrentInput` | `class RecurrentInput final : public Property<Connection> {` |
| `nntrainer/compiler/recurrent_realizer.cpp` | 93 | `RecurrentOutput` | `class RecurrentOutput final : public Property<Connection> {` |
| `nntrainer/compiler/recurrent_realizer.h` | 30 | `UnrollFor` | `class UnrollFor;` |
| `nntrainer/compiler/recurrent_realizer.h` | 31 | `AsSequence` | `class AsSequence;` |
| `nntrainer/compiler/recurrent_realizer.h` | 32 | `InputIsSequence` | `class InputIsSequence;` |
| `nntrainer/compiler/recurrent_realizer.h` | 33 | `OutputLayer` | `class OutputLayer;` |
| `nntrainer/compiler/recurrent_realizer.h` | 34 | `RecurrentInput` | `class RecurrentInput;` |
| `nntrainer/compiler/recurrent_realizer.h` | 35 | `RecurrentOutput` | `class RecurrentOutput;` |
| `nntrainer/compiler/recurrent_realizer.h` | 36 | `DynamicTimeSequence` | `class DynamicTimeSequence;` |
| `nntrainer/compiler/recurrent_realizer.h` | 44 | `RecurrentRealizer` | `class RecurrentRealizer final : public GraphRealizer {` |
| `nntrainer/compiler/remap_realizer.h` | 30 | `RemapRealizer` | `class RemapRealizer final : public GraphRealizer {` |
| `nntrainer/compiler/slice_realizer.cpp` | 42 | `NodeInfo` | `struct NodeInfo {` |
| `nntrainer/compiler/slice_realizer.h` | 24 | `Connection` | `class Connection;` |
| `nntrainer/compiler/slice_realizer.h` | 30 | `SliceRealizer` | `class SliceRealizer final : public GraphRealizer {` |
| `nntrainer/compiler/tflite_export_realizer.h` | 27 | `TfliteExportRealizer` | `class TfliteExportRealizer final : public GraphRealizer {` |
| `nntrainer/compiler/tflite_interpreter.cpp` | 109 | `BidirectionalIndexMap` | `class BidirectionalIndexMap {` |
| `nntrainer/compiler/tflite_interpreter.cpp` | 183 | `TfOpIdxMap` | `class TfOpIdxMap {` |
| `nntrainer/compiler/tflite_interpreter.h` | 24 | `TfliteInterpreter` | `class TfliteInterpreter : public GraphInterpreter {` |
| `nntrainer/compiler/tflite_opnode.h` | 26 | `LayerNode` | `class LayerNode;` |
| `nntrainer/compiler/tflite_opnode.h` | 27 | `RunLayerContext` | `class RunLayerContext;` |
| `nntrainer/compiler/tflite_opnode.h` | 33 | `TfOpNode` | `class TfOpNode {` |
| `nntrainer/context.h` | 48 | `Context` | `class Context {` |
| `nntrainer/context.h` | 216 | `(anonymous typedef struct)` | `typedef struct {` |
| `nntrainer/context_data.h` | 23 | `ComputeOps` | `class ComputeOps;` |
| `nntrainer/context_data.h` | 32 | `ContextData` | `class ContextData {` |
| `nntrainer/dataset/data_iteration.h` | 26 | `Sample` | `class Sample;` |
| `nntrainer/dataset/data_iteration.h` | 32 | `Iteration` | `class Iteration {` |
| `nntrainer/dataset/data_iteration.h` | 137 | `Sample` | `class Sample {` |
| `nntrainer/dataset/data_producer.h` | 27 | `Exporter` | `class Exporter;` |
| `nntrainer/dataset/data_producer.h` | 33 | `DataProducer` | `class DataProducer {` |
| `nntrainer/dataset/databuffer.cpp` | 50 | `PropsBufferSize` | `class PropsBufferSize : public nntrainer::PositiveIntegerProperty {` |
| `nntrainer/dataset/databuffer.cpp` | 85 | `NotifyOnDestruct` | `class NotifyOnDestruct {` |
| `nntrainer/dataset/databuffer.h` | 43 | `Exporter` | `class Exporter;` |
| `nntrainer/dataset/databuffer.h` | 52 | `PropsBufferSize` | `class PropsBufferSize;` |
| `nntrainer/dataset/databuffer.h` | 58 | `DataBuffer` | `class DataBuffer : public ml::train::Dataset {` |
| `nntrainer/dataset/dir_data_producers.h` | 27 | `DirPath` | `class DirPath;` |
| `nntrainer/dataset/dir_data_producers.h` | 34 | `DirDataProducer` | `class DirDataProducer final : public DataProducer {` |
| `nntrainer/dataset/func_data_producer.h` | 26 | `Exporter` | `class Exporter;` |
| `nntrainer/dataset/func_data_producer.h` | 34 | `FuncDataProducer` | `class FuncDataProducer final : public DataProducer {` |
| `nntrainer/dataset/iteration_queue.h` | 38 | `ViewQueue` | `template <typename T> class ViewQueue {` |
| `nntrainer/dataset/iteration_queue.h` | 108 | `ScopedView` | `template <typename T> class ScopedView {` |
| `nntrainer/dataset/iteration_queue.h` | 199 | `IterationQueue` | `class IterationQueue {` |
| `nntrainer/dataset/iteration_queue.h` | 275 | `MarkableIteration` | `class MarkableIteration {` |
| `nntrainer/dataset/iteration_queue.h` | 344 | `FlowState` | `enum class FlowState {` |
| `nntrainer/dataset/random_data_producers.cpp` | 25 | `PropsMin` | `class PropsMin : public Property<float> {` |
| `nntrainer/dataset/random_data_producers.cpp` | 41 | `PropsMax` | `class PropsMax : public Property<float> {` |
| `nntrainer/dataset/random_data_producers.cpp` | 59 | `PropsNumSamples` | `class PropsNumSamples : public Property<unsigned int> {` |
| `nntrainer/dataset/random_data_producers.h` | 25 | `PropsMin` | `class PropsMin;` |
| `nntrainer/dataset/random_data_producers.h` | 26 | `PropsMax` | `class PropsMax;` |
| `nntrainer/dataset/random_data_producers.h` | 27 | `PropsNumSamples` | `class PropsNumSamples;` |
| `nntrainer/dataset/random_data_producers.h` | 33 | `RandomDataOneHotProducer` | `class RandomDataOneHotProducer final : public DataProducer {` |
| `nntrainer/dataset/raw_file_data_producer.h` | 28 | `FilePath` | `class FilePath;` |
| `nntrainer/dataset/raw_file_data_producer.h` | 37 | `RawFileDataProducer` | `class RawFileDataProducer final : public DataProducer {` |
| `nntrainer/engine.h` | 51 | `Engine` | `class Engine : public Singleton<Engine> {` |
| `nntrainer/graph/connection.h` | 24 | `Connection` | `class Connection {` |
| `nntrainer/graph/graph_core.h` | 35 | `GraphCore` | `class GraphCore {` |
| `nntrainer/graph/graph_node.h` | 27 | `GraphNode` | `class GraphNode {` |
| `nntrainer/graph/graph_node.h` | 124 | `GraphNodeIterator` | `class GraphNodeIterator {` |
| `nntrainer/graph/graph_node.h` | 312 | `GraphNodeReverseIterator` | `class GraphNodeReverseIterator : public std::reverse_iterator<T_iterator> {` |
| `nntrainer/graph/network_graph.h` | 32 | `Connection` | `class Connection;` |
| `nntrainer/graph/network_graph.h` | 37 | `NetworkGraph` | `class NetworkGraph {` |
| `nntrainer/layers/acti_func.h` | 29 | `Tensor` | `class Tensor;` |
| `nntrainer/layers/acti_func.h` | 35 | `ActiFunc` | `class ActiFunc {` |
| `nntrainer/layers/activation_layer.h` | 29 | `ActivationLayer` | `class ActivationLayer : public Layer {` |
| `nntrainer/layers/add_layer.h` | 28 | `AddLayer` | `class AddLayer : public BinaryOperationLayer {` |
| `nntrainer/layers/addition_layer.h` | 27 | `AdditionLayer` | `class AdditionLayer : public Layer {` |
| `nntrainer/layers/attention_layer.cpp` | 32 | `AttentionParams` | `enum AttentionParams { query = 0, value = 1, key = 2, weights };` |
| `nntrainer/layers/attention_layer.h` | 29 | `AttentionLayer` | `class AttentionLayer : public virtual Layer {` |
| `nntrainer/layers/bn_layer.cpp` | 36 | `BNParams` | `enum BNParams {` |
| `nntrainer/layers/bn_layer.h` | 40 | `BatchNormalizationLayer` | `class BatchNormalizationLayer : public Layer {` |
| `nntrainer/layers/cast_layer.h` | 28 | `CastLayer` | `class CastLayer : public UnaryOperationLayer {` |
| `nntrainer/layers/centroid_knn.cpp` | 32 | `KNNParams` | `enum KNNParams { map, num_samples };` |
| `nntrainer/layers/centroid_knn.h` | 27 | `CentroidKNN` | `class CentroidKNN : public Layer {` |
| `nntrainer/layers/channel_shuffle.h` | 31 | `ChannelShuffle` | `class ChannelShuffle : public LayerImpl {` |
| `nntrainer/layers/cl_layers/addition_layer_cl.h` | 29 | `AdditionLayerCL` | `class AdditionLayerCL : public LayerImplCl {` |
| `nntrainer/layers/cl_layers/concat_cl.h` | 34 | `ConcatLayerCl` | `class ConcatLayerCl : public LayerImplCl {` |
| `nntrainer/layers/cl_layers/concat_cl.h` | 235 | `Kernels` | `enum Kernels {` |
| `nntrainer/layers/cl_layers/fc_layer_cl.cpp` | 30 | `FCParams` | `enum FCParams { weight, bias };` |
| `nntrainer/layers/cl_layers/fc_layer_cl.h` | 28 | `FullyConnectedLayerCl` | `class FullyConnectedLayerCl : public LayerImplCl {` |
| `nntrainer/layers/cl_layers/layer_impl_cl.h` | 32 | `LayerImplCl` | `class LayerImplCl : public LayerImpl {` |
| `nntrainer/layers/cl_layers/reshape_cl.h` | 30 | `ReshapeLayerCl` | `class ReshapeLayerCl : public LayerImplCl {` |
| `nntrainer/layers/cl_layers/reshape_cl.h` | 158 | `Kernels` | `enum Kernels { COPY_CL, COPY_CL_FP16 };` |
| `nntrainer/layers/cl_layers/rmsnorm_layer_cl.cpp` | 32 | `RMSParams` | `enum RMSParams { gamma };` |
| `nntrainer/layers/cl_layers/rmsnorm_layer_cl.h` | 32 | `RMSNormLayerCl` | `class RMSNormLayerCl : public LayerImplCl {` |
| `nntrainer/layers/cl_layers/rmsnorm_layer_cl.h` | 143 | `Kernels` | `enum Kernels {` |
| `nntrainer/layers/cl_layers/swiglu_cl.h` | 34 | `SwiGLULayerCl` | `class SwiGLULayerCl final : public LayerImplCl {` |
| `nntrainer/layers/cl_layers/swiglu_cl.h` | 137 | `Kernels` | `enum Kernels { SWIGLU_CL, SWIGLU_CL_FP16 }; /** kernels enum */` |
| `nntrainer/layers/cl_layers/transpose_cl.h` | 29 | `TransposeLayerCl` | `class TransposeLayerCl final : public LayerImplCl {` |
| `nntrainer/layers/common_properties.cpp` | 77 | `stat` | `struct stat dir;` |
| `nntrainer/layers/common_properties.cpp` | 135 | `Padding_` | `class Padding_ : public nntrainer::Property<int> {` |
| `nntrainer/layers/common_properties.h` | 33 | `ActivationType` | `enum class ActivationType {` |
| `nntrainer/layers/common_properties.h` | 57 | `Name` | `class Name : public nntrainer::Property<std::string> {` |
| `nntrainer/layers/common_properties.h` | 96 | `Unit` | `class Unit : public PositiveIntegerProperty {` |
| `nntrainer/layers/common_properties.h` | 107 | `Trainable` | `class Trainable : public nntrainer::Property<bool> {` |
| `nntrainer/layers/common_properties.h` | 122 | `TensorDimension` | `class TensorDimension : public TensorDimProperty {` |
| `nntrainer/layers/common_properties.h` | 132 | `InPlaceProp` | `class InPlaceProp : public nntrainer::Property<bool> {` |
| `nntrainer/layers/common_properties.h` | 142 | `InPlaceDirectionProp` | `class InPlaceDirectionProp : public nntrainer::Property<std::string> {` |
| `nntrainer/layers/common_properties.h` | 154 | `Packed` | `class Packed : public nntrainer::Property<bool> {` |
| `nntrainer/layers/common_properties.h` | 171 | `DisableBias` | `class DisableBias : public nntrainer::Property<bool> {` |
| `nntrainer/layers/common_properties.h` | 187 | `IntegrateBias` | `class IntegrateBias : public nntrainer::Property<bool> {` |
| `nntrainer/layers/common_properties.h` | 203 | `Normalization` | `class Normalization : public nntrainer::Property<bool> {` |
| `nntrainer/layers/common_properties.h` | 219 | `Standardization` | `class Standardization : public nntrainer::Property<bool> {` |
| `nntrainer/layers/common_properties.h` | 234 | `connection_prop_tag` | `struct connection_prop_tag {};` |
| `nntrainer/layers/common_properties.h` | 240 | `InputConnection` | `class InputConnection : public nntrainer::Property<Connection> {` |
| `nntrainer/layers/common_properties.h` | 263 | `Epsilon` | `class Epsilon : public nntrainer::Property<float> {` |
| `nntrainer/layers/common_properties.h` | 288 | `Exponent` | `class Exponent : public nntrainer::Property<float> {` |
| `nntrainer/layers/common_properties.h` | 304 | `Momentum` | `class Momentum : public nntrainer::Property<float> {` |
| `nntrainer/layers/common_properties.h` | 331 | `SplitNumber` | `class SplitNumber : public PositiveIntegerProperty {` |
| `nntrainer/layers/common_properties.h` | 342 | `Axis` | `class Axis : public nntrainer::PositiveIntegerProperty {` |
| `nntrainer/layers/common_properties.h` | 363 | `StartDimension` | `class StartDimension : public Axis {` |
| `nntrainer/layers/common_properties.h` | 385 | `EndDimension` | `class EndDimension : public Axis {` |
| `nntrainer/layers/common_properties.h` | 407 | `StartIndex` | `class StartIndex : public PositiveIntegerProperty {` |
| `nntrainer/layers/common_properties.h` | 417 | `EndIndex` | `class EndIndex : public PositiveIntegerProperty {` |
| `nntrainer/layers/common_properties.h` | 427 | `SplitDimension` | `class SplitDimension : public Axis {` |
| `nntrainer/layers/common_properties.h` | 445 | `ConcatDimension` | `class ConcatDimension : public SplitDimension {};` |
| `nntrainer/layers/common_properties.h` | 451 | `ReduceDimension` | `class ReduceDimension : public SplitDimension {};` |
| `nntrainer/layers/common_properties.h` | 458 | `FilterSize` | `class FilterSize : public nntrainer::PositiveIntegerProperty {` |
| `nntrainer/layers/common_properties.h` | 468 | `KernelSize` | `class KernelSize : public nntrainer::PositiveIntegerProperty {` |
| `nntrainer/layers/common_properties.h` | 478 | `PoolSize` | `class PoolSize : public nntrainer::PositiveIntegerProperty {` |
| `nntrainer/layers/common_properties.h` | 500 | `Stride` | `class Stride : public nntrainer::PositiveIntegerProperty {` |
| `nntrainer/layers/common_properties.h` | 516 | `Dilation` | `class Dilation : public nntrainer::PositiveIntegerProperty {` |
| `nntrainer/layers/common_properties.h` | 540 | `Padding2D` | `class Padding2D final : public nntrainer::Property<std::string> {` |
| `nntrainer/layers/common_properties.h` | 578 | `Padding1D` | `class Padding1D final : public nntrainer::Property<std::string> {` |
| `nntrainer/layers/common_properties.h` | 608 | `InDim` | `class InDim : public nntrainer::PositiveIntegerProperty {` |
| `nntrainer/layers/common_properties.h` | 619 | `OutDim` | `class OutDim : public nntrainer::PositiveIntegerProperty {` |
| `nntrainer/layers/common_properties.h` | 630 | `ZeroIdxMask` | `class ZeroIdxMask : public nntrainer::Property<unsigned int> {` |
| `nntrainer/layers/common_properties.h` | 641 | `DropOutRate` | `class DropOutRate : public nntrainer::Property<float> {` |
| `nntrainer/layers/common_properties.h` | 668 | `RandomTranslate` | `class RandomTranslate : public nntrainer::Property<float> {` |
| `nntrainer/layers/common_properties.h` | 687 | `FilePath` | `class FilePath : public Property<std::string> {` |
| `nntrainer/layers/common_properties.h` | 733 | `DirPath` | `class DirPath : public Property<std::string> {` |
| `nntrainer/layers/common_properties.h` | 770 | `ReturnSequences` | `class ReturnSequences : public nntrainer::Property<bool> {` |
| `nntrainer/layers/common_properties.h` | 785 | `Bidirectional` | `class Bidirectional : public nntrainer::Property<bool> {` |
| `nntrainer/layers/common_properties.h` | 801 | `AsSequence` | `class AsSequence : public Property<Connection> {` |
| `nntrainer/layers/common_properties.h` | 812 | `InputIsSequence` | `class InputIsSequence : public Name {` |
| `nntrainer/layers/common_properties.h` | 824 | `ResetAfter` | `class ResetAfter : public nntrainer::Property<bool> {` |
| `nntrainer/layers/common_properties.h` | 840 | `NumClass` | `class NumClass final : public nntrainer::Property<unsigned int> {` |
| `nntrainer/layers/common_properties.h` | 856 | `BasicRegularizerConstant` | `class BasicRegularizerConstant : public nntrainer::Property<float> {` |
| `nntrainer/layers/common_properties.h` | 882 | `WeightRegularizerConstant` | `class WeightRegularizerConstant final : public BasicRegularizerConstant {` |
| `nntrainer/layers/common_properties.h` | 899 | `WeightDecay` | `class WeightDecay final : public BasicRegularizerConstant {` |
| `nntrainer/layers/common_properties.h` | 916 | `BiasDecay` | `class BiasDecay final : public BasicRegularizerConstant {` |
| `nntrainer/layers/common_properties.h` | 932 | `OutputLayer` | `class OutputLayer : public Name {` |
| `nntrainer/layers/common_properties.h` | 955 | `LabelLayer` | `class LabelLayer : public Name {` |
| `nntrainer/layers/common_properties.h` | 977 | `ActivationTypeInfo` | `struct ActivationTypeInfo {` |
| `nntrainer/layers/common_properties.h` | 996 | `Activation` | `class Activation final` |
| `nntrainer/layers/common_properties.h` | 1007 | `HiddenStateActivation` | `class HiddenStateActivation final : public EnumProperty<ActivationTypeInfo> {` |
| `nntrainer/layers/common_properties.h` | 1024 | `RecurrentActivation` | `class RecurrentActivation final : public EnumProperty<ActivationTypeInfo> {` |
| `nntrainer/layers/common_properties.h` | 1040 | `InitializerInfo` | `struct InitializerInfo {` |
| `nntrainer/layers/common_properties.h` | 1057 | `WeightInitializer` | `class WeightInitializer final : public EnumProperty<InitializerInfo> {` |
| `nntrainer/layers/common_properties.h` | 1071 | `BiasInitializer` | `class BiasInitializer final : public EnumProperty<InitializerInfo> {` |
| `nntrainer/layers/common_properties.h` | 1085 | `MuInitializer` | `class MuInitializer final : public EnumProperty<InitializerInfo> {` |
| `nntrainer/layers/common_properties.h` | 1099 | `VarInitializer` | `class VarInitializer final : public EnumProperty<InitializerInfo> {` |
| `nntrainer/layers/common_properties.h` | 1113 | `GammaInitializer` | `class GammaInitializer final : public EnumProperty<InitializerInfo> {` |
| `nntrainer/layers/common_properties.h` | 1127 | `BetaInitializer` | `class BetaInitializer final : public EnumProperty<InitializerInfo> {` |
| `nntrainer/layers/common_properties.h` | 1140 | `RegularizerInfo` | `struct RegularizerInfo {` |
| `nntrainer/layers/common_properties.h` | 1152 | `BasicRegularizer` | `class BasicRegularizer : public EnumProperty<RegularizerInfo> {` |
| `nntrainer/layers/common_properties.h` | 1175 | `WeightRegularizer` | `class WeightRegularizer final : public BasicRegularizer {` |
| `nntrainer/layers/common_properties.h` | 1189 | `UpsampleModeInfo` | `struct UpsampleModeInfo {` |
| `nntrainer/layers/common_properties.h` | 1193 | `Interpolation` | `enum class Interpolation { nearest, bilinear };` |
| `nntrainer/layers/common_properties.h` | 1207 | `UpsampleMode` | `class UpsampleMode final : public EnumProperty<UpsampleModeInfo> {` |
| `nntrainer/layers/common_properties.h` | 1216 | `PoolingTypeInfo` | `struct PoolingTypeInfo {` |
| `nntrainer/layers/common_properties.h` | 1220 | `Enum` | `enum class Enum {` |
| `nntrainer/layers/common_properties.h` | 1239 | `PoolingType` | `class PoolingType final : public EnumProperty<PoolingTypeInfo> {` |
| `nntrainer/layers/common_properties.h` | 1248 | `FlipDirectionInfo` | `struct FlipDirectionInfo {` |
| `nntrainer/layers/common_properties.h` | 1249 | `Enum` | `enum class Enum { horizontal, vertical, horizontal_and_vertical };` |
| `nntrainer/layers/common_properties.h` | 1261 | `FlipDirection` | `class FlipDirection final : public EnumProperty<FlipDirectionInfo> {` |
| `nntrainer/layers/common_properties.h` | 1274 | `Timestep` | `class Timestep : public Property<unsigned> {` |
| `nntrainer/layers/common_properties.h` | 1285 | `MaxTimestep` | `class MaxTimestep : public PositiveIntegerProperty {` |
| `nntrainer/layers/common_properties.h` | 1300 | `GenericShape` | `class GenericShape : public Property<TensorDim> {` |
| `nntrainer/layers/common_properties.h` | 1320 | `TargetShape` | `class TargetShape : public GenericShape {` |
| `nntrainer/layers/common_properties.h` | 1331 | `Scale` | `class Scale : public nntrainer::Property<float> {` |
| `nntrainer/layers/common_properties.h` | 1342 | `ScaledDotProduct` | `class ScaledDotProduct : public nntrainer::Property<bool> {` |
| `nntrainer/layers/common_properties.h` | 1357 | `CausalMask` | `class CausalMask : public nntrainer::Property<bool> {` |
| `nntrainer/layers/common_properties.h` | 1372 | `Print` | `class Print : public nntrainer::Property<bool> {` |
| `nntrainer/layers/common_properties.h` | 1387 | `MoL_K` | `class MoL_K : public PositiveIntegerProperty {` |
| `nntrainer/layers/common_properties.h` | 1397 | `NumHeads` | `class NumHeads : public PositiveIntegerProperty {` |
| `nntrainer/layers/common_properties.h` | 1414 | `ProjectedKeyDim` | `class ProjectedKeyDim : public PositiveIntegerProperty {` |
| `nntrainer/layers/common_properties.h` | 1427 | `ProjectedValueDim` | `class ProjectedValueDim : public PositiveIntegerProperty {` |
| `nntrainer/layers/common_properties.h` | 1440 | `OutputShape` | `class OutputShape : public PositiveIntegerProperty {` |
| `nntrainer/layers/common_properties.h` | 1450 | `ReturnAttentionWeightInfo` | `struct ReturnAttentionWeightInfo {` |
| `nntrainer/layers/common_properties.h` | 1451 | `Enum` | `enum class Enum { none, before, after };` |
| `nntrainer/layers/common_properties.h` | 1467 | `ReturnAttentionWeight` | `class ReturnAttentionWeight : public EnumProperty<ReturnAttentionWeightInfo> {` |
| `nntrainer/layers/common_properties.h` | 1486 | `AverageAttentionWeight` | `class AverageAttentionWeight : public Property<bool> {` |
| `nntrainer/layers/common_properties.h` | 1497 | `LoraRank` | `class LoraRank : public PositiveIntegerProperty {` |
| `nntrainer/layers/common_properties.h` | 1508 | `LoraAlpha` | `class LoraAlpha : public PositiveIntegerProperty {` |
| `nntrainer/layers/common_properties.h` | 1518 | `ClipGradByGlobalNorm` | `class ClipGradByGlobalNorm : public Property<float> {` |
| `nntrainer/layers/common_properties.h` | 1529 | `LossScaleForMixed` | `class LossScaleForMixed : public Property<float> {` |
| `nntrainer/layers/common_properties.h` | 1548 | `LearningRate` | `class LearningRate : public Property<float> {` |
| `nntrainer/layers/common_properties.h` | 1559 | `MaxLearningRate` | `class MaxLearningRate : public Property<float> {` |
| `nntrainer/layers/common_properties.h` | 1570 | `MinLearningRate` | `class MinLearningRate : public Property<float> {` |
| `nntrainer/layers/common_properties.h` | 1581 | `Iteration` | `class Iteration : public Property<unsigned int> {` |
| `nntrainer/layers/common_properties.h` | 1591 | `DecayRate` | `class DecayRate : public Property<float> {` |
| `nntrainer/layers/common_properties.h` | 1601 | `DecaySteps` | `class DecaySteps : public PositiveIntegerProperty {` |
| `nntrainer/layers/common_properties.h` | 1611 | `PropsUserData` | `class PropsUserData final : public Property<void *> {` |
| `nntrainer/layers/common_properties.h` | 1621 | `TensorLifeInfo` | `struct TensorLifeInfo {` |
| `nntrainer/layers/common_properties.h` | 1663 | `TensorLife` | `class TensorLife : public EnumProperty<TensorLifeInfo> {` |
| `nntrainer/layers/common_properties.h` | 1680 | `WeightName` | `class WeightName : public Name {` |
| `nntrainer/layers/common_properties.h` | 1689 | `TensorName` | `class TensorName : public Name {` |
| `nntrainer/layers/concat_layer.h` | 27 | `ConcatLayer` | `class ConcatLayer : public Layer {` |
| `nntrainer/layers/conv1d_layer.h` | 24 | `Conv2DLayer` | `class Conv2DLayer;` |
| `nntrainer/layers/conv1d_layer.h` | 30 | `Conv1DLayer` | `class Conv1DLayer : public LayerImpl {` |
| `nntrainer/layers/conv2d_layer.cpp` | 292 | `ConvParams` | `enum ConvParams { weight, bias };` |
| `nntrainer/layers/conv2d_layer.h` | 31 | `Conv2DLayer` | `class Conv2DLayer : public LayerImpl {` |
| `nntrainer/layers/conv2d_transpose_layer.cpp` | 203 | `ConvParams` | `enum ConvParams { weight, bias };` |
| `nntrainer/layers/conv2d_transpose_layer.h` | 31 | `Conv2DTransposeLayer` | `class Conv2DTransposeLayer : public LayerImpl {` |
| `nntrainer/layers/cosine_layer.h` | 27 | `CosineLayer` | `class CosineLayer : public UnaryOperationLayer {` |
| `nntrainer/layers/depthwise_conv2d_layer.h` | 31 | `DepthwiseConv2DLayer` | `class DepthwiseConv2DLayer : public LayerImpl {` |
| `nntrainer/layers/divide_layer.h` | 28 | `DivideLayer` | `class DivideLayer : public BinaryOperationLayer {` |
| `nntrainer/layers/dropout.h` | 27 | `DropOutLayer` | `class DropOutLayer : public Layer {` |
| `nntrainer/layers/embedding.cpp` | 28 | `EmbeddingParams` | `enum EmbeddingParams { weight };` |
| `nntrainer/layers/embedding.h` | 28 | `EmbeddingLayer` | `class EmbeddingLayer : public LayerImpl {` |
| `nntrainer/layers/entropy_layer.h` | 26 | `LossLayer` | `class LossLayer : public Layer {` |
| `nntrainer/layers/fc_layer.cpp` | 39 | `FCParams` | `enum FCParams { weight, bias };` |
| `nntrainer/layers/fc_layer.cpp` | 40 | `LORAParams` | `enum LORAParams { loraA, loraB, loraTmp, loraOut };` |
| `nntrainer/layers/fc_layer.h` | 27 | `FullyConnectedLayer` | `class FullyConnectedLayer : public LayerImpl {` |
| `nntrainer/layers/flatten_layer.h` | 26 | `FlattenLayer` | `class FlattenLayer : public ReshapeLayer {` |
| `nntrainer/layers/gather_layer.h` | 27 | `GatherLayer` | `class GatherLayer : public BinaryOperationLayer {` |
| `nntrainer/layers/gru.cpp` | 41 | `GRUParams` | `enum GRUParams {` |
| `nntrainer/layers/gru.h` | 28 | `GRULayer` | `class GRULayer : public LayerImpl {` |
| `nntrainer/layers/grucell.cpp` | 254 | `GRUCellParams` | `enum GRUCellParams {` |
| `nntrainer/layers/grucell.h` | 28 | `GRUCellLayer` | `class GRUCellLayer : public LayerImpl {` |
| `nntrainer/layers/grucell.h` | 104 | `INOUT_INDEX` | `enum INOUT_INDEX {` |
| `nntrainer/layers/identity_layer.h` | 28 | `IdentityLayer` | `class IdentityLayer final : public Layer {` |
| `nntrainer/layers/input_layer.h` | 38 | `InputLayer` | `class InputLayer : public Layer {` |
| `nntrainer/layers/layer_context.h` | 31 | `Var_Grad` | `class Var_Grad;` |
| `nntrainer/layers/layer_context.h` | 42 | `InitLayerContext` | `class InitLayerContext {` |
| `nntrainer/layers/layer_context.h` | 460 | `RunLayerContext` | `class RunLayerContext {` |
| `nntrainer/layers/layer_devel.h` | 37 | `Layer` | `class Layer;` |
| `nntrainer/layers/layer_devel.h` | 42 | `InitLayerContext` | `class InitLayerContext;` |
| `nntrainer/layers/layer_devel.h` | 43 | `Exporter` | `class Exporter;` |
| `nntrainer/layers/layer_devel.h` | 49 | `InPlaceType` | `enum class InPlaceType {` |
| `nntrainer/layers/layer_devel.h` | 68 | `InPlaceDirection` | `enum class InPlaceDirection {` |
| `nntrainer/layers/layer_devel.h` | 82 | `Layer` | `class Layer {` |
| `nntrainer/layers/layer_devel.h` | 125 | `PropertyType` | `enum class PropertyType {` |
| `nntrainer/layers/layer_devel.h` | 563 | `(anonymous typedef struct)` | `typedef struct {` |
| `nntrainer/layers/layer_impl.h` | 30 | `InitLayerContext` | `class InitLayerContext;` |
| `nntrainer/layers/layer_impl.h` | 31 | `RunLayerContext` | `class RunLayerContext;` |
| `nntrainer/layers/layer_impl.h` | 32 | `Exporter` | `class Exporter;` |
| `nntrainer/layers/layer_impl.h` | 35 | `WeightRegularizer` | `class WeightRegularizer;` |
| `nntrainer/layers/layer_impl.h` | 36 | `WeightRegularizerConstant` | `class WeightRegularizerConstant;` |
| `nntrainer/layers/layer_impl.h` | 37 | `WeightInitializer` | `class WeightInitializer;` |
| `nntrainer/layers/layer_impl.h` | 38 | `WeightDecay` | `class WeightDecay;` |
| `nntrainer/layers/layer_impl.h` | 39 | `BiasDecay` | `class BiasDecay;` |
| `nntrainer/layers/layer_impl.h` | 40 | `BiasInitializer` | `class BiasInitializer;` |
| `nntrainer/layers/layer_impl.h` | 41 | `DisableBias` | `class DisableBias;` |
| `nntrainer/layers/layer_impl.h` | 42 | `Print` | `class Print;` |
| `nntrainer/layers/layer_impl.h` | 50 | `LayerImpl` | `class LayerImpl : public virtual Layer {` |
| `nntrainer/layers/layer_node.cpp` | 54 | `Flatten` | `class Flatten : public Property<bool> {` |
| `nntrainer/layers/layer_node.cpp` | 65 | `Distribute` | `class Distribute : public Property<bool> {` |
| `nntrainer/layers/layer_node.cpp` | 76 | `Loss` | `class Loss : public Property<float> {` |
| `nntrainer/layers/layer_node.cpp` | 109 | `InputShape` | `class InputShape : public GenericShape {` |
| `nntrainer/layers/layer_node.cpp` | 120 | `SharedFrom` | `class SharedFrom : public Name {` |
| `nntrainer/layers/layer_node.h` | 40 | `Layer` | `class Layer;` |
| `nntrainer/layers/layer_node.h` | 41 | `Connection` | `class Connection;` |
| `nntrainer/layers/layer_node.h` | 42 | `Exporter` | `class Exporter;` |
| `nntrainer/layers/layer_node.h` | 43 | `ContextData` | `class ContextData;` |
| `nntrainer/layers/layer_node.h` | 46 | `Name` | `class Name;` |
| `nntrainer/layers/layer_node.h` | 47 | `Distribute` | `class Distribute;` |
| `nntrainer/layers/layer_node.h` | 48 | `Flatten` | `class Flatten;` |
| `nntrainer/layers/layer_node.h` | 49 | `Loss` | `class Loss;` |
| `nntrainer/layers/layer_node.h` | 50 | `InputShape` | `class InputShape;` |
| `nntrainer/layers/layer_node.h` | 51 | `Activation` | `class Activation;` |
| `nntrainer/layers/layer_node.h` | 52 | `SharedFrom` | `class SharedFrom;` |
| `nntrainer/layers/layer_node.h` | 53 | `InputConnection` | `class InputConnection;` |
| `nntrainer/layers/layer_node.h` | 54 | `ClipGradByGlobalNorm` | `class ClipGradByGlobalNorm;` |
| `nntrainer/layers/layer_node.h` | 55 | `Packed` | `class Packed;` |
| `nntrainer/layers/layer_node.h` | 56 | `LossScaleForMixed` | `class LossScaleForMixed;` |
| `nntrainer/layers/layer_node.h` | 57 | `ComputeEngine` | `class ComputeEngine;` |
| `nntrainer/layers/layer_node.h` | 64 | `LayerNode` | `class LayerNode final : public ml::train::Layer, public GraphNode {` |
| `nntrainer/layers/layer_node.h` | 903 | `PrintPreset` | `enum class PrintPreset {` |
| `nntrainer/layers/layer_normalization_layer.cpp` | 31 | `LNParams` | `enum LNParams {` |
| `nntrainer/layers/layer_normalization_layer.h` | 32 | `LayerNormalizationLayer` | `class LayerNormalizationLayer : public Layer {` |
| `nntrainer/layers/loss/constant_derivative_loss_layer.h` | 26 | `ConstantDerivativeLossLayer` | `class ConstantDerivativeLossLayer final : public LossLayer {` |
| `nntrainer/layers/loss/cross_entropy_loss_layer.h` | 27 | `CrossEntropyLossLayer` | `class CrossEntropyLossLayer : public LossLayer {` |
| `nntrainer/layers/loss/cross_entropy_sigmoid_loss_layer.h` | 27 | `CrossEntropySigmoidLossLayer` | `class CrossEntropySigmoidLossLayer : public LossLayer {` |
| `nntrainer/layers/loss/cross_entropy_softmax_loss_layer.h` | 27 | `CrossEntropySoftmaxLossLayer` | `class CrossEntropySoftmaxLossLayer : public LossLayer {` |
| `nntrainer/layers/loss/kld_loss_layer.h` | 28 | `KLDLossLayer` | `class KLDLossLayer : public LossLayer {` |
| `nntrainer/layers/loss/loss_layer.h` | 28 | `LossLayer` | `class LossLayer : public Layer {` |
| `nntrainer/layers/loss/mse_loss_layer.h` | 26 | `MSELossLayer` | `class MSELossLayer : public LossLayer {` |
| `nntrainer/layers/lstm.cpp` | 25 | `LSTMParams` | `enum LSTMParams {` |
| `nntrainer/layers/lstm.h` | 28 | `LSTMLayer` | `class LSTMLayer : public LSTMCore {` |
| `nntrainer/layers/lstmcell.cpp` | 22 | `LSTMCellParams` | `enum LSTMCellParams {` |
| `nntrainer/layers/lstmcell.h` | 28 | `LSTMCellLayer` | `class LSTMCellLayer : public LSTMCore {` |
| `nntrainer/layers/lstmcell.h` | 92 | `INOUT_INDEX` | `enum INOUT_INDEX {` |
| `nntrainer/layers/lstmcell_core.h` | 29 | `LSTMCore` | `class LSTMCore : public LayerImpl {` |
| `nntrainer/layers/matmul_layer.h` | 27 | `MatMulLayer` | `class MatMulLayer : public BinaryOperationLayer {` |
| `nntrainer/layers/mol_attention_layer.cpp` | 36 | `MoLAttentionParams` | `enum MoLAttentionParams {` |
| `nntrainer/layers/mol_attention_layer.h` | 27 | `MoLAttentionLayer` | `class MoLAttentionLayer : public LayerImpl {` |
| `nntrainer/layers/multi_head_attention_layer.cpp` | 37 | `INOUT_INDEX` | `enum INOUT_INDEX {` |
| `nntrainer/layers/multi_head_attention_layer.cpp` | 48 | `AttentionParams` | `enum AttentionParams {` |
| `nntrainer/layers/multi_head_attention_layer.h` | 29 | `MultiHeadAttentionLayer` | `class MultiHeadAttentionLayer : public LayerImpl {` |
| `nntrainer/layers/multiout_layer.h` | 25 | `MultiOutLayer` | `class MultiOutLayer : public Layer {` |
| `nntrainer/layers/multiply_layer.h` | 28 | `MultiplyLayer` | `class MultiplyLayer : public BinaryOperationLayer {` |
| `nntrainer/layers/negative_layer.h` | 28 | `NegativeLayer` | `class NegativeLayer : public UnaryOperationLayer {` |
| `nntrainer/layers/nnstreamer_layer.cpp` | 31 | `PropsNNSModelPath` | `class PropsNNSModelPath : public Property<std::string> {` |
| `nntrainer/layers/nnstreamer_layer.h` | 26 | `PropsNNSModelPath` | `class PropsNNSModelPath;` |
| `nntrainer/layers/nnstreamer_layer.h` | 32 | `NNStreamerLayer` | `class NNStreamerLayer : public Layer {` |
| `nntrainer/layers/operation_layer.h` | 26 | `UnaryOperationLayer` | `class UnaryOperationLayer : public Layer {` |
| `nntrainer/layers/operation_layer.h` | 90 | `BinaryOperationLayer` | `class BinaryOperationLayer : public Layer {` |
| `nntrainer/layers/permute_layer.h` | 31 | `PermuteDims` | `class PermuteDims : public nntrainer::Property<unsigned int> {` |
| `nntrainer/layers/permute_layer.h` | 50 | `PermuteLayer` | `class PermuteLayer : public Layer {` |
| `nntrainer/layers/plugged_layer.h` | 28 | `PluggedLayer` | `class PluggedLayer : public nntrainer::Layer {` |
| `nntrainer/layers/pooling2d_layer.cpp` | 34 | `PoolFunc` | `template <typename T> struct PoolFunc {` |
| `nntrainer/layers/pooling2d_layer.h` | 33 | `Pooling2DLayer` | `class Pooling2DLayer : public Layer {` |
| `nntrainer/layers/pooling2d_layer.h` | 39 | `PaddingType` | `enum class PaddingType {` |
| `nntrainer/layers/positional_encoding_layer.cpp` | 25 | `PositionalEncodingParams` | `enum PositionalEncodingParams {` |
| `nntrainer/layers/positional_encoding_layer.h` | 31 | `PositionalEncodingLayer` | `class PositionalEncodingLayer : public Layer {` |
| `nntrainer/layers/pow_layer.h` | 28 | `PowLayer` | `class PowLayer : public UnaryOperationLayer {` |
| `nntrainer/layers/preprocess_flip_layer.h` | 29 | `PreprocessFlipLayer` | `class PreprocessFlipLayer : public Layer {` |
| `nntrainer/layers/preprocess_l2norm_layer.h` | 28 | `PreprocessL2NormLayer` | `class PreprocessL2NormLayer : public Layer {` |
| `nntrainer/layers/preprocess_translate_layer.h` | 33 | `PreprocessTranslateLayer` | `class PreprocessTranslateLayer : public Layer {` |
| `nntrainer/layers/reduce_mean_layer.h` | 23 | `RunLayerContext` | `class RunLayerContext;` |
| `nntrainer/layers/reduce_mean_layer.h` | 24 | `InitLayerContext` | `class InitLayerContext;` |
| `nntrainer/layers/reduce_mean_layer.h` | 30 | `ReduceMeanLayer` | `class ReduceMeanLayer : public Layer {` |
| `nntrainer/layers/reduce_sum_layer.h` | 28 | `ReduceSumLayer` | `class ReduceSumLayer : public Layer {` |
| `nntrainer/layers/reshape_layer.h` | 27 | `ReshapeLayer` | `class ReshapeLayer : public Layer {` |
| `nntrainer/layers/rnn.cpp` | 31 | `RNNParams` | `enum RNNParams {` |
| `nntrainer/layers/rnn.h` | 28 | `RNNLayer` | `class RNNLayer : public LayerImpl {` |
| `nntrainer/layers/rnncell.cpp` | 33 | `RNNCellParams` | `enum RNNCellParams {` |
| `nntrainer/layers/rnncell.h` | 28 | `RNNCellLayer` | `class RNNCellLayer : public LayerImpl {` |
| `nntrainer/layers/rnncell.h` | 103 | `INOUT_INDEX` | `enum INOUT_INDEX {` |
| `nntrainer/layers/sine_layer.h` | 28 | `SineLayer` | `class SineLayer : public UnaryOperationLayer {` |
| `nntrainer/layers/slice_layer.h` | 27 | `SliceLayer` | `class SliceLayer : public UnaryOperationLayer {` |
| `nntrainer/layers/split_layer.h` | 30 | `SplitLayer` | `class SplitLayer : public Layer {` |
| `nntrainer/layers/sqrt_layer.h` | 28 | `SQRTLayer` | `class SQRTLayer : public UnaryOperationLayer {` |
| `nntrainer/layers/subtract_layer.h` | 28 | `SubtractLayer` | `class SubtractLayer : public BinaryOperationLayer {` |
| `nntrainer/layers/tangent_layer.h` | 27 | `TangentLayer` | `class TangentLayer : public UnaryOperationLayer {` |
| `nntrainer/layers/tensor_layer.h` | 29 | `TensorLayer` | `class TensorLayer : public Layer {` |
| `nntrainer/layers/tflite_layer.cpp` | 27 | `PropsTflModelPath` | `class PropsTflModelPath : public Property<std::string> {` |
| `nntrainer/layers/tflite_layer.h` | 26 | `TensorDim` | `class TensorDim;` |
| `nntrainer/layers/tflite_layer.h` | 31 | `PropsTflModelPath` | `class PropsTflModelPath;` |
| `nntrainer/layers/tflite_layer.h` | 37 | `TfLiteLayer` | `class TfLiteLayer : public Layer {` |
| `nntrainer/layers/time_dist.h` | 27 | `TimeDistLayer` | `class TimeDistLayer : public Layer {` |
| `nntrainer/layers/upsample2d_layer.h` | 31 | `Upsample2dLayer` | `class Upsample2dLayer : public Layer {` |
| `nntrainer/layers/weight_layer.h` | 28 | `WeightLayer` | `class WeightLayer : public LayerImpl {` |
| `nntrainer/layers/zoneout_lstmcell.cpp` | 24 | `ZoneoutLSTMParams` | `enum ZoneoutLSTMParams {` |
| `nntrainer/layers/zoneout_lstmcell.h` | 30 | `ZoneoutLSTMCellLayer` | `class ZoneoutLSTMCellLayer : public LSTMCore {` |
| `nntrainer/layers/zoneout_lstmcell.h` | 37 | `HiddenStateZoneOutRate` | `class HiddenStateZoneOutRate : public nntrainer::Property<float> {` |
| `nntrainer/layers/zoneout_lstmcell.h` | 67 | `CellStateZoneOutRate` | `class CellStateZoneOutRate : public nntrainer::Property<float> {` |
| `nntrainer/layers/zoneout_lstmcell.h` | 97 | `Test` | `class Test : public nntrainer::Property<bool> {` |
| `nntrainer/layers/zoneout_lstmcell.h` | 172 | `INOUT_INDEX` | `enum INOUT_INDEX {` |
| `nntrainer/mem_allocator.h` | 33 | `MemAllocator` | `class MemAllocator {` |
| `nntrainer/models/dynamic_training_optimization.h` | 47 | `Weight` | `class Weight;` |
| `nntrainer/models/dynamic_training_optimization.h` | 48 | `Var_Grad` | `class Var_Grad;` |
| `nntrainer/models/dynamic_training_optimization.h` | 49 | `OptimizerWrapped` | `class OptimizerWrapped;` |
| `nntrainer/models/dynamic_training_optimization.h` | 55 | `DynamicTrainingOptimization` | `class DynamicTrainingOptimization {` |
| `nntrainer/models/execution_mode.h` | 22 | `ExecutionMode` | `enum class ExecutionMode {` |
| `nntrainer/models/model_common_properties.h` | 25 | `Epochs` | `class Epochs : public PositiveIntegerProperty {` |
| `nntrainer/models/model_common_properties.h` | 41 | `LossType` | `class LossType : public Property<std::string> {` |
| `nntrainer/models/model_common_properties.h` | 59 | `SavePath` | `class SavePath : public Property<std::string> {` |
| `nntrainer/models/model_common_properties.h` | 69 | `SaveBestPath` | `class SaveBestPath : public Property<std::string> {` |
| `nntrainer/models/model_common_properties.h` | 80 | `TrainingBatchSize` | `class TrainingBatchSize : public PositiveIntegerProperty {` |
| `nntrainer/models/model_common_properties.h` | 97 | `ContinueTrain` | `class ContinueTrain : public Property<bool> {` |
| `nntrainer/models/model_common_properties.h` | 115 | `MemoryOptimization` | `class MemoryOptimization : public Property<bool> {` |
| `nntrainer/models/model_common_properties.h` | 133 | `Fsu` | `class Fsu : public Property<bool> {` |
| `nntrainer/models/model_common_properties.h` | 150 | `FsuPath` | `class FsuPath : public Property<std::string> {` |
| `nntrainer/models/model_common_properties.h` | 167 | `FsuLookahead` | `class FsuLookahead : public Property<unsigned int> {` |
| `nntrainer/models/model_common_properties.h` | 184 | `ModelTensorDataTypeInfo` | `struct ModelTensorDataTypeInfo {` |
| `nntrainer/models/model_common_properties.h` | 185 | `Enum` | `enum Enum {` |
| `nntrainer/models/model_common_properties.h` | 225 | `ModelTensorDataType` | `class ModelTensorDataType final : public EnumProperty<ModelTensorDataTypeInfo> {` |
| `nntrainer/models/model_common_properties.h` | 243 | `LossScale` | `class LossScale : public Property<float> {` |
| `nntrainer/models/model_loader.h` | 26 | `OptimizerWrapped` | `class OptimizerWrapped;` |
| `nntrainer/models/model_loader.h` | 32 | `ModelLoader` | `class ModelLoader {` |
| `nntrainer/models/neuralnet.cpp` | 827 | `stat` | `struct stat st {};` |
| `nntrainer/models/neuralnet.cpp` | 1091 | `stat` | `struct stat st {};` |
| `nntrainer/models/neuralnet.h` | 52 | `DataSet` | `class DataSet;` |
| `nntrainer/models/neuralnet.h` | 53 | `DatasetType` | `enum class DatasetType;` |
| `nntrainer/models/neuralnet.h` | 54 | `DatasetModeType` | `enum class DatasetModeType;` |
| `nntrainer/models/neuralnet.h` | 55 | `ExecutionMode` | `enum class ExecutionMode;` |
| `nntrainer/models/neuralnet.h` | 60 | `Exporter` | `class Exporter;` |
| `nntrainer/models/neuralnet.h` | 68 | `DataBuffer` | `class DataBuffer;` |
| `nntrainer/models/neuralnet.h` | 77 | `NeuralNetwork` | `class NeuralNetwork : public ml::train::Model {` |
| `nntrainer/nntrainer_error.h` | 91 | `ErrorNotification` | `class ErrorNotification {` |
| `nntrainer/nntrainer_error.h` | 158 | `not_supported` | `struct not_supported : public std::invalid_argument {` |
| `nntrainer/nntrainer_error.h` | 165 | `permission_denied` | `struct permission_denied : public std::invalid_argument {` |
| `nntrainer/nntrainer_logger.cpp` | 77 | `tm` | `struct tm now;` |
| `nntrainer/nntrainer_logger.cpp` | 130 | `tm` | `struct tm now;` |
| `nntrainer/nntrainer_logger.h` | 61 | `Logger` | `class Logger {` |
| `nntrainer/nntrainer_logger.h` | 107 | `Cleanup` | `class Cleanup {` |
| `nntrainer/opencl/CL/cl.h` | 29 | `_cl_platform_id` | `typedef struct _cl_platform_id *cl_platform_id;` |
| `nntrainer/opencl/CL/cl.h` | 30 | `_cl_device_id` | `typedef struct _cl_device_id *cl_device_id;` |
| `nntrainer/opencl/CL/cl.h` | 31 | `_cl_context` | `typedef struct _cl_context *cl_context;` |
| `nntrainer/opencl/CL/cl.h` | 32 | `_cl_command_queue` | `typedef struct _cl_command_queue *cl_command_queue;` |
| `nntrainer/opencl/CL/cl.h` | 33 | `_cl_mem` | `typedef struct _cl_mem *cl_mem;` |
| `nntrainer/opencl/CL/cl.h` | 34 | `_cl_program` | `typedef struct _cl_program *cl_program;` |
| `nntrainer/opencl/CL/cl.h` | 35 | `_cl_kernel` | `typedef struct _cl_kernel *cl_kernel;` |
| `nntrainer/opencl/CL/cl.h` | 36 | `_cl_event` | `typedef struct _cl_event *cl_event;` |
| `nntrainer/opencl/CL/cl.h` | 37 | `_cl_sampler` | `typedef struct _cl_sampler *cl_sampler;` |
| `nntrainer/opencl/CL/cl.h` | 121 | `_cl_image_format` | `typedef struct _cl_image_format {` |
| `nntrainer/opencl/CL/cl.h` | 128 | `_cl_image_desc` | `typedef struct _cl_image_desc {` |
| `nntrainer/opencl/CL/cl.h` | 181 | `_cl_buffer_region` | `typedef struct _cl_buffer_region {` |
| `nntrainer/opencl/CL/cl.h` | 192 | `_cl_name_version` | `typedef struct _cl_name_version {` |
| `nntrainer/opencl/opencl_buffer.h` | 28 | `Buffer` | `class Buffer : public Noncopyable {` |
| `nntrainer/opencl/opencl_buffer_manager.h` | 33 | `ClBufferManager` | `class ClBufferManager : public Singleton<ClBufferManager> {` |
| `nntrainer/opencl/opencl_command_queue_manager.h` | 30 | `CommandQueueManager` | `class CommandQueueManager : public Singleton<CommandQueueManager> {` |
| `nntrainer/opencl/opencl_context_manager.h` | 31 | `ContextManager` | `class ContextManager : public Singleton<ContextManager> {` |
| `nntrainer/opencl/opencl_device_info.h` | 29 | `DeviceInfo` | `class DeviceInfo {` |
| `nntrainer/opencl/opencl_kernel.h` | 29 | `Kernel` | `class Kernel {` |
| `nntrainer/opencl/opencl_program.h` | 29 | `Program` | `class Program {` |
| `nntrainer/optimizers/adam.cpp` | 36 | `AdamParams` | `enum AdamParams { wm, wv };` |
| `nntrainer/optimizers/adam.h` | 28 | `PropsB1` | `class PropsB1 : public Property<double> {` |
| `nntrainer/optimizers/adam.h` | 38 | `PropsB2` | `class PropsB2 : public Property<double> {` |
| `nntrainer/optimizers/adam.h` | 49 | `PropsEpsilon` | `class PropsEpsilon : public Property<double> {` |
| `nntrainer/optimizers/adam.h` | 59 | `TorchRef` | `class TorchRef : public Property<bool> {` |
| `nntrainer/optimizers/adam.h` | 69 | `Adam` | `class Adam : public Optimizer {` |
| `nntrainer/optimizers/adamw.cpp` | 40 | `AdamParams` | `enum AdamParams { wm, wv };` |
| `nntrainer/optimizers/adamw.h` | 30 | `PropsWeightDecayW` | `class PropsWeightDecayW : public Property<double> {` |
| `nntrainer/optimizers/adamw.h` | 40 | `AdamW` | `class AdamW : public Optimizer {` |
| `nntrainer/optimizers/lion.cpp` | 34 | `LionParams` | `enum LionParams { m };` |
| `nntrainer/optimizers/lion.h` | 29 | `PropsWeightDecayLion` | `class PropsWeightDecayLion : public Property<double> {` |
| `nntrainer/optimizers/lion.h` | 40 | `Lion` | `class Lion : public Optimizer {` |
| `nntrainer/optimizers/lr_scheduler.h` | 24 | `Exporter` | `class Exporter;` |
| `nntrainer/optimizers/lr_scheduler.h` | 29 | `LearningRateSchedulerType` | `enum LearningRateSchedulerType {` |
| `nntrainer/optimizers/lr_scheduler.h` | 41 | `LearningRateScheduler` | `class LearningRateScheduler : public ml::train::LearningRateScheduler {` |
| `nntrainer/optimizers/lr_scheduler_constant.h` | 29 | `ConstantLearningRateScheduler` | `class ConstantLearningRateScheduler : public LearningRateScheduler {` |
| `nntrainer/optimizers/lr_scheduler_cosine.h` | 29 | `CosineAnnealingLearningRateScheduler` | `class CosineAnnealingLearningRateScheduler : public LearningRateScheduler {` |
| `nntrainer/optimizers/lr_scheduler_exponential.h` | 28 | `ExponentialLearningRateScheduler` | `class ExponentialLearningRateScheduler final` |
| `nntrainer/optimizers/lr_scheduler_linear.h` | 29 | `LinearLearningRateScheduler` | `class LinearLearningRateScheduler : public LearningRateScheduler {` |
| `nntrainer/optimizers/lr_scheduler_step.h` | 26 | `LearningRate` | `class LearningRate;` |
| `nntrainer/optimizers/lr_scheduler_step.h` | 27 | `Iteration` | `class Iteration;` |
| `nntrainer/optimizers/lr_scheduler_step.h` | 34 | `StepLearningRateScheduler` | `class StepLearningRateScheduler final : public LearningRateScheduler {` |
| `nntrainer/optimizers/optimizer_context.h` | 23 | `Weight` | `class Weight;` |
| `nntrainer/optimizers/optimizer_context.h` | 31 | `RunOptimizerContext` | `class RunOptimizerContext {` |
| `nntrainer/optimizers/optimizer_devel.h` | 27 | `Exporter` | `class Exporter;` |
| `nntrainer/optimizers/optimizer_devel.h` | 33 | `Optimizer` | `class Optimizer {` |
| `nntrainer/optimizers/optimizer_devel.h` | 123 | `(anonymous typedef struct)` | `typedef struct {` |
| `nntrainer/optimizers/optimizer_wrapped.h` | 36 | `OptimizerWrapped` | `class OptimizerWrapped : public ml::train::Optimizer {` |
| `nntrainer/optimizers/plugged_optimizer.h` | 29 | `PluggedOptimizer` | `class PluggedOptimizer : public nntrainer::Optimizer {` |
| `nntrainer/optimizers/sgd.h` | 25 | `SGD` | `class SGD : public Optimizer {` |
| `nntrainer/qnn/jni/iotensor_wrapper.hpp` | 36 | `IOTensorWrapper` | `class IOTensorWrapper {` |
| `nntrainer/qnn/jni/qnn/op/QNNGraph.h` | 31 | `QNNGraph` | `class QNNGraph : public LayerImpl {` |
| `nntrainer/qnn/jni/qnn/op/QNNLinear.cpp` | 33 | `FCParams` | `enum FCParams { weight, bias };` |
| `nntrainer/qnn/jni/qnn/op/QNNLinear.cpp` | 34 | `LORAParams` | `enum LORAParams { loraA, loraB, loraTmp, loraOut };` |
| `nntrainer/qnn/jni/qnn/op/QNNLinear.h` | 29 | `QNNLinear` | `class QNNLinear : public LayerImpl {` |
| `nntrainer/qnn/jni/qnn/qnn_properties.h` | 30 | `quant_param_prop_tag` | `struct quant_param_prop_tag {};` |
| `nntrainer/qnn/jni/qnn/qnn_properties.h` | 37 | `QuantParam` | `class QuantParam` |
| `nntrainer/qnn/jni/qnn/qnn_properties.h` | 48 | `InputQuantParam` | `class InputQuantParam` |
| `nntrainer/qnn/jni/qnn/qnn_properties.h` | 60 | `OutputQuantParam` | `class OutputQuantParam` |
| `nntrainer/qnn/jni/qnn_context_var.h` | 52 | `StatusCode` | `enum class StatusCode {` |
| `nntrainer/qnn/jni/qnn_context_var.h` | 66 | `Qnn_Context_Graph_t` | `struct Qnn_Context_Graph_t {` |
| `nntrainer/qnn/jni/qnn_context_var.h` | 113 | `QNNVar` | `struct QNNVar {` |
| `nntrainer/qnn/jni/qnn_context_var.h` | 377 | `QNNBackendVar` | `class QNNBackendVar : public ContextData {` |
| `nntrainer/qnn/jni/qnn_rpc_manager.h` | 39 | `QNNRpcManager` | `class QNNRpcManager : public MemAllocator {` |
| `nntrainer/qnn_context.h` | 55 | `QNNContext` | `class QNNContext : public Context, public Singleton<QNNContext> {` |
| `nntrainer/qnn_context.h` | 218 | `isSupportedHelper` | `template <typename Args, typename T> struct isSupportedHelper;` |
| `nntrainer/qnn_context.h` | 224 | `isSupportedHelper` | `struct isSupportedHelper<T, QNNContext::FactoryMap<Args...>> {` |
| `nntrainer/qnn_context.h` | 233 | `isSupported` | `struct isSupported : isSupportedHelper<T, decltype(factory_map)> {};` |
| `nntrainer/tensor/basic_planner.h` | 29 | `BasicPlanner` | `class BasicPlanner : public MemoryPlanner {` |
| `nntrainer/tensor/bcq_tensor.h` | 25 | `BCQTensor` | `class BCQTensor : public TensorBase {` |
| `nntrainer/tensor/cache_elem.h` | 22 | `CachePolicy` | `enum CachePolicy {` |
| `nntrainer/tensor/cache_elem.h` | 45 | `CacheElem` | `class CacheElem {` |
| `nntrainer/tensor/cache_elem.h` | 47 | `Options` | `enum Options {` |
| `nntrainer/tensor/cache_loader.h` | 33 | `LoadState` | `enum class LoadState { Idle, Loading, Loaded, Unloading };` |
| `nntrainer/tensor/cache_loader.h` | 39 | `CacheLoader` | `class CacheLoader {` |
| `nntrainer/tensor/cache_pool.h` | 34 | `CachePool` | `class CachePool : public MemoryPool {` |
| `nntrainer/tensor/char_tensor.h` | 24 | `CharTensor` | `class CharTensor : public TensorBase {` |
| `nntrainer/tensor/cl_operations/cl_compute_ops.cpp` | 32 | `ClComputeOps` | `class ClComputeOps : public ComputeOps {` |
| `nntrainer/tensor/cpu_backend/arm/kai/kai_common.h` | 157 | `kai_datatype` | `enum kai_datatype {` |
| `nntrainer/tensor/cpu_backend/arm/kai/kai_common.h` | 281 | `kai_rhs_pack_qsi8cx_params` | `struct kai_rhs_pack_qsi8cx_params {` |
| `nntrainer/tensor/cpu_backend/arm/kai/kai_common.h` | 289 | `kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params` | `struct kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params {` |
| `nntrainer/tensor/cpu_backend/arm/kai/kai_common.h` | 292 | `kai_datatype` | `enum kai_datatype scale_dt;` |
| `nntrainer/tensor/cpu_backend/arm/kai/kai_common.h` | 296 | `kai_rhs_pack_qs4cxs1s0_param` | `struct kai_rhs_pack_qs4cxs1s0_param {` |
| `nntrainer/tensor/cpu_backend/arm/kai/kai_common.h` | 302 | `kai_matmul_requantize32_params` | `struct kai_matmul_requantize32_params {` |
| `nntrainer/tensor/cpu_backend/arm/kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp_qsi4cxp_interface.h` | 55 | `kai_matmul_clamp_f32_qai8dxp_qsi4cxp_ukernel` | `struct kai_matmul_clamp_f32_qai8dxp_qsi4cxp_ukernel {` |
| `nntrainer/tensor/cpu_backend/arm/kai/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_interface.h` | 60 | `kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_ukernel` | `struct kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_ukernel {` |
| `nntrainer/tensor/cpu_backend/arm/kai/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod.c` | 30 | `(anonymous typedef struct)` | `typedef struct {` |
| `nntrainer/tensor/cpu_backend/arm/kai/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod.c` | 30 | `(anonymous typedef struct)` | `typedef struct {` |
| `nntrainer/tensor/cpu_backend/arm/kai/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm.c` | 30 | `(anonymous typedef struct)` | `typedef struct {` |
| `nntrainer/tensor/cpu_backend/arm/kai/pack/kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0.h` | 30 | `kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0_params` | `struct kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0_params {` |
| `nntrainer/tensor/cpu_backend/arm/kai/pack/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h` | 30 | `kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params` | `struct kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params {` |
| `nntrainer/tensor/cpu_backend/arm/kleidiai_interface_qai8dxp_qsi4cxp.cpp` | 75 | `rhs_format` | `enum class rhs_format {` |
| `nntrainer/tensor/cpu_backend/arm/kleidiai_interface_qai8dxp_qsi4cxp.cpp` | 85 | `kai_matmul_ukernel_f32_qa8dxp_qs4cxp` | `struct kai_matmul_ukernel_f32_qa8dxp_qs4cxp {` |
| `nntrainer/tensor/cpu_backend/arm/kleidiai_interface_qai8dxp_qsi4cxp.cpp` | 232 | `kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params` | `struct kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params nxk_params;` |
| `nntrainer/tensor/cpu_backend/arm/kleidiai_interface_qai8dxp_qsi4cxp.cpp` | 246 | `kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0_params` | `struct kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0_params kxn_params;` |
| `nntrainer/tensor/cpu_backend/arm/kleidiai_interface_qai8dxp_qsi4cxp.cpp` | 341 | `kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params` | `struct kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params nxk_params;` |
| `nntrainer/tensor/cpu_backend/arm/kleidiai_interface_qai8dxp_qsi4cxp.cpp` | 355 | `kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0_params` | `struct kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0_params kxn_params;` |
| `nntrainer/tensor/cpu_backend/arm/kleidiai_interface_qsi8d32p_qsi4c32p.cpp` | 79 | `rhs_format` | `enum class rhs_format {` |
| `nntrainer/tensor/cpu_backend/arm/kleidiai_interface_qsi8d32p_qsi4c32p.cpp` | 89 | `kai_matmul_ukernel_f32_qsi8d32p_qsi4c32p` | `struct kai_matmul_ukernel_f32_qsi8d32p_qsi4c32p {` |
| `nntrainer/tensor/cpu_backend/arm/kleidiai_interface_qsi8d32p_qsi4c32p.cpp` | 319 | `kai_rhs_pack_qs4cxs1s0_param` | `struct kai_rhs_pack_qs4cxs1s0_param params;` |
| `nntrainer/tensor/cpu_backend/arm/kleidiai_interface_qsi8d32p_qsi4c32p.cpp` | 413 | `kai_rhs_pack_qs4cxs1s0_param` | `struct kai_rhs_pack_qs4cxs1s0_param params;` |
| `nntrainer/tensor/cpu_backend/arm/kleidiai_interface_qsi8d32p_qsi4c32p_omp.cpp` | 42 | `kai_matmul_ukernel_f32_qsi8d32p_qsi4c32p` | `struct kai_matmul_ukernel_f32_qsi8d32p_qsi4c32p {` |
| `nntrainer/tensor/cpu_backend/arm/kleidiai_interface_qsi8d32p_qsi4c32p_stdthread.cpp` | 42 | `kai_matmul_ukernel_f32_qsi8d32p_qsi4c32p` | `struct kai_matmul_ukernel_f32_qsi8d32p_qsi4c32p {` |
| `nntrainer/tensor/cpu_backend/arm/neon_impl.cpp` | 2319 | `OutputType` | `enum class OutputType { FP16, FP32 };` |
| `nntrainer/tensor/cpu_backend/arm/neon_impl_fp16.cpp` | 2094 | `OutputType` | `enum class OutputType { FP16, FP32 };` |
| `nntrainer/tensor/cpu_backend/compute_ops.h` | 47 | `ComputeOps` | `class ComputeOps {` |
| `nntrainer/tensor/cpu_backend/cpu_ops_table.cpp` | 24 | `CpuComputeOps` | `class CpuComputeOps : public ComputeOps {` |
| `nntrainer/tensor/cpu_backend/fallback/fallback_internal.cpp` | 45 | `block_q4_0x8` | `struct block_q4_0x8 {` |
| `nntrainer/tensor/cpu_backend/fallback/fallback_kleidiai.h` | 31 | `rhs_format` | `enum class rhs_format {` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_common.h` | 72 | `(anonymous typedef struct)` | `typedef struct {` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_common.h` | 88 | `(anonymous typedef struct)` | `typedef struct {` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_common.h` | 98 | `(anonymous typedef struct)` | `typedef struct {` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_common.h` | 107 | `(anonymous typedef struct)` | `typedef struct {` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_common.h` | 112 | `(anonymous typedef struct)` | `typedef struct {` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_common.h` | 123 | `block` | `template <int K, int N> struct block {` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_common.h` | 136 | `block_q4_Kx8` | `struct block_q4_Kx8 {` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_common.h` | 146 | `block_q8_Kx4` | `struct block_q8_Kx4 {` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_common.h` | 169 | `ggml_type` | `enum ggml_type {` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_fp16_fp32.cpp` | 272 | `ggml_object_type` | `enum ggml_object_type {` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_fp16_fp32.cpp` | 281 | `ggml_object` | `struct ggml_object {` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_fp16_fp32.cpp` | 285 | `ggml_object` | `struct ggml_object *next;` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_fp16_fp32.cpp` | 287 | `ggml_object_type` | `enum ggml_object_type type;` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_fp16_fp32.cpp` | 301 | `ggml_context` | `struct ggml_context {` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_fp16_fp32.cpp` | 309 | `ggml_object` | `struct ggml_object *objects_begin;` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_fp16_fp32.cpp` | 310 | `ggml_object` | `struct ggml_object *objects_end;` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_fp16_fp32.cpp` | 316 | `ggml_init_params` | `struct ggml_init_params {` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_fp16_fp32.cpp` | 376 | `timespec` | `struct timespec ts;` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_fp16_fp32.cpp` | 382 | `timespec` | `struct timespec ts;` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_fp16_fp32.cpp` | 388 | `ggml_context` | `struct ggml_context *ggml_init(struct ggml_init_params params) {` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_fp16_fp32.cpp` | 407 | `ggml_context` | `struct ggml_context *ctx = nullptr;` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_fp16_fp32.cpp` | 426 | `ggml_init_params` | `struct ggml_init_params params = {0, NULL, false};` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_fp16_fp32.cpp` | 427 | `ggml_context` | `struct ggml_context *ctx = ggml_init(params);` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_utils.h` | 944 | `ggml_int16x8x2_t` | `typedef struct ggml_int16x8x2_t {` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_utils.h` | 957 | `ggml_uint8x16x2_t` | `typedef struct ggml_uint8x16x2_t {` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_utils.h` | 970 | `ggml_uint8x16x4_t` | `typedef struct ggml_uint8x16x4_t {` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_utils.h` | 985 | `ggml_int8x16x2_t` | `typedef struct ggml_int8x16x2_t {` |
| `nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_utils.h` | 998 | `ggml_int8x16x4_t` | `typedef struct ggml_int8x16x4_t {` |
| `nntrainer/tensor/cpu_backend/x86/avx2_impl.cpp` | 151 | `Exp2Table` | `struct Exp2Table {` |
| `nntrainer/tensor/cpu_backend/x86/avx2_impl.cpp` | 334 | `block_q4_0x8` | `struct block_q4_0x8 {` |
| `nntrainer/tensor/cpu_backend/x86/avx2_impl.cpp` | 1762 | `OutputType` | `enum class OutputType { FP16, FP32 };` |
| `nntrainer/tensor/float_tensor.h` | 30 | `FloatTensor` | `class FloatTensor : public TensorBase {` |
| `nntrainer/tensor/half_tensor.h` | 30 | `HalfTensor` | `class HalfTensor : public TensorBase {` |
| `nntrainer/tensor/int4_tensor.h` | 32 | `Int4QTensor` | `class Int4QTensor : public TensorBase {` |
| `nntrainer/tensor/int4_utils.h` | 25 | `Int4Utils` | `class Int4Utils {` |
| `nntrainer/tensor/lazy_tensor.h` | 28 | `LazyTensor` | `class LazyTensor {` |
| `nntrainer/tensor/manager.h` | 49 | `MMapedMemory` | `class MMapedMemory : public Noncopyable, public Nonmovable {` |
| `nntrainer/tensor/manager.h` | 104 | `Manager` | `class Manager : public Noncopyable, public Nonmovable {` |
| `nntrainer/tensor/manager.h` | 113 | `TensorGroupType` | `enum TensorGroupType {` |
| `nntrainer/tensor/memory_data.h` | 26 | `MemoryData` | `class MemoryData {` |
| `nntrainer/tensor/memory_planner.h` | 26 | `MemoryPlanner` | `class MemoryPlanner {` |
| `nntrainer/tensor/memory_pool.h` | 42 | `MemoryPool` | `class MemoryPool {` |
| `nntrainer/tensor/optimized_v1_planner.cpp` | 28 | `MemoryRequest` | `struct MemoryRequest {` |
| `nntrainer/tensor/optimized_v1_planner.h` | 44 | `OptimizedV1Planner` | `class OptimizedV1Planner : public MemoryPlanner {` |
| `nntrainer/tensor/optimized_v2_planner.cpp` | 30 | `MemoryRequest` | `struct MemoryRequest {` |
| `nntrainer/tensor/optimized_v2_planner.cpp` | 50 | `WGradMemoryRequest` | `struct WGradMemoryRequest {` |
| `nntrainer/tensor/optimized_v2_planner.h` | 45 | `OptimizedV2Planner` | `class OptimizedV2Planner : public MemoryPlanner {` |
| `nntrainer/tensor/optimized_v3_planner.cpp` | 28 | `MemoryRequest` | `struct MemoryRequest {` |
| `nntrainer/tensor/optimized_v3_planner.h` | 31 | `OptimizedV3Planner` | `class OptimizedV3Planner : public MemoryPlanner {` |
| `nntrainer/tensor/q4_0_tensor.h` | 26 | `block_q4_0` | `struct block_q4_0 {` |
| `nntrainer/tensor/q4_0_tensor.h` | 37 | `Q4_0_Tensor` | `class Q4_0_Tensor : public TensorBase {` |
| `nntrainer/tensor/q4_0_utils.h` | 33 | `block` | `template <int K, int N> struct block {` |
| `nntrainer/tensor/q4_0_utils.h` | 44 | `(anonymous typedef struct)` | `typedef struct {` |
| `nntrainer/tensor/q4_0_utils.h` | 56 | `Q4_0Utils` | `class Q4_0Utils {` |
| `nntrainer/tensor/q4_k_tensor.h` | 25 | `block_q4_K` | `struct block_q4_K {` |
| `nntrainer/tensor/q4_k_tensor.h` | 39 | `Q4_K_Tensor` | `class Q4_K_Tensor : public Uint4QTensor {` |
| `nntrainer/tensor/q6_k_tensor.h` | 25 | `block_q6_K` | `struct block_q6_K {` |
| `nntrainer/tensor/q6_k_tensor.h` | 38 | `Q6_K_Tensor` | `class Q6_K_Tensor : public TensorBase {` |
| `nntrainer/tensor/quantizer.h` | 23 | `Tensor` | `class Tensor;` |
| `nntrainer/tensor/quantizer.h` | 32 | `QScheme` | `enum class QScheme : uint16_t {` |
| `nntrainer/tensor/quantizer.h` | 55 | `Quantizer` | `class Quantizer {` |
| `nntrainer/tensor/quantizer.h` | 167 | `UniformQuantizer` | `class UniformQuantizer : public Quantizer {` |
| `nntrainer/tensor/quantizer.h` | 177 | `NonUniformQuantizer` | `class NonUniformQuantizer : public Quantizer {` |
| `nntrainer/tensor/quantizer.h` | 191 | `PerTensorAffineQuantizer` | `class PerTensorAffineQuantizer : public UniformQuantizer {` |
| `nntrainer/tensor/quantizer.h` | 248 | `PerChannelAffineQuantizer` | `class PerChannelAffineQuantizer : public UniformQuantizer {` |
| `nntrainer/tensor/quantizer.h` | 299 | `BinaryCodeBasedQuantizer` | `class BinaryCodeBasedQuantizer : public NonUniformQuantizer {` |
| `nntrainer/tensor/quantizer.h` | 353 | `GgmlQuantizer` | `class GgmlQuantizer : public NonUniformQuantizer {` |
| `nntrainer/tensor/quantizer.h` | 414 | `Quantization` | `class Quantization {` |
| `nntrainer/tensor/short_tensor.h` | 23 | `ShortTensor` | `class ShortTensor : public TensorBase {` |
| `nntrainer/tensor/swap_device.h` | 48 | `SwapDevice` | `class SwapDevice {` |
| `nntrainer/tensor/task.h` | 32 | `Task` | `class Task {` |
| `nntrainer/tensor/task.h` | 42 | `Type` | `enum Type {` |
| `nntrainer/tensor/task.h` | 50 | `State` | `enum State {` |
| `nntrainer/tensor/task.h` | 123 | `TaskAsync` | `class TaskAsync : public Task {` |
| `nntrainer/tensor/task.h` | 125 | `Priority` | `enum Priority {` |
| `nntrainer/tensor/task_executor.h` | 44 | `TaskExecutor` | `class TaskExecutor {` |
| `nntrainer/tensor/task_executor.h` | 50 | `CompleteStatus` | `enum CompleteStatus {` |
| `nntrainer/tensor/task_executor.h` | 61 | `TaskDesc` | `struct TaskDesc {` |
| `nntrainer/tensor/task_executor.h` | 147 | `Task` | `struct Task {` |
| `nntrainer/tensor/tensor.h` | 38 | `LazyTensor` | `class LazyTensor;` |
| `nntrainer/tensor/tensor.h` | 54 | `Tensor` | `class Tensor {` |
| `nntrainer/tensor/tensor_base.h` | 55 | `ComputeOps` | `class ComputeOps;` |
| `nntrainer/tensor/tensor_base.h` | 56 | `ContextData` | `class ContextData;` |
| `nntrainer/tensor/tensor_base.h` | 66 | `Initializer` | `enum class Initializer {` |
| `nntrainer/tensor/tensor_base.h` | 78 | `Tensor` | `class Tensor;` |
| `nntrainer/tensor/tensor_base.h` | 79 | `SrcSharedTensorBase` | `class SrcSharedTensorBase;` |
| `nntrainer/tensor/tensor_base.h` | 95 | `TensorBase` | `class TensorBase {` |
| `nntrainer/tensor/tensor_base.h` | 919 | `BroadcastInfo` | `struct BroadcastInfo {` |
| `nntrainer/tensor/tensor_base.h` | 987 | `SrcSharedTensorBase` | `class SrcSharedTensorBase {` |
| `nntrainer/tensor/tensor_pool.h` | 38 | `TensorPool` | `class TensorPool {` |
| `nntrainer/tensor/tensor_pool.h` | 361 | `SourceDetails` | `struct SourceDetails {` |
| `nntrainer/tensor/tensor_pool.h` | 373 | `DependentDetails` | `struct DependentDetails {` |
| `nntrainer/tensor/tensor_pool.h` | 382 | `RequestSpec` | `struct RequestSpec {` |
| `nntrainer/tensor/tensor_wrap_specs.h` | 29 | `WeightRegularizer` | `enum class WeightRegularizer {` |
| `nntrainer/tensor/tensor_wrap_specs.h` | 39 | `TensorLifespan` | `enum class TensorLifespan {` |
| `nntrainer/tensor/tensor_wrap_specs.h` | 104 | `TensorSpecV2` | `struct TensorSpecV2 {` |
| `nntrainer/tensor/tensor_wrap_specs.h` | 111 | `RequestType` | `enum class RequestType {` |
| `nntrainer/tensor/tensor_wrap_specs.h` | 155 | `VarGradSpecV2` | `struct VarGradSpecV2 {` |
| `nntrainer/tensor/tensor_wrap_specs.h` | 204 | `WeightSpecV2` | `struct WeightSpecV2 {` |
| `nntrainer/tensor/uint_tensor.h` | 29 | `UIntTensor` | `template <typename T> class UIntTensor : public TensorBase {` |
| `nntrainer/tensor/uint4_tensor.h` | 31 | `Uint4QTensor` | `class Uint4QTensor : public TensorBase {` |
| `nntrainer/tensor/var_grad.h` | 28 | `Var_Grad` | `class Var_Grad {` |
| `nntrainer/tensor/weight.h` | 29 | `Weight` | `class Weight : public Var_Grad {` |
| `nntrainer/utils/base_properties.h` | 37 | `prop_info` | `template <typename T> struct prop_info {` |
| `nntrainer/utils/base_properties.h` | 50 | `prop_info` | `template <typename T> struct prop_info<std::vector<T>> {` |
| `nntrainer/utils/base_properties.h` | 61 | `prop_info` | `template <typename T, size_t size> struct prop_info<std::array<T, size>> {` |
| `nntrainer/utils/base_properties.h` | 82 | `int_prop_tag` | `struct int_prop_tag {};` |
| `nntrainer/utils/base_properties.h` | 88 | `uint_prop_tag` | `struct uint_prop_tag {};` |
| `nntrainer/utils/base_properties.h` | 94 | `size_t_prop_tag` | `struct size_t_prop_tag {};` |
| `nntrainer/utils/base_properties.h` | 100 | `dimension_prop_tag` | `struct dimension_prop_tag {};` |
| `nntrainer/utils/base_properties.h` | 106 | `float_prop_tag` | `struct float_prop_tag {};` |
| `nntrainer/utils/base_properties.h` | 112 | `double_prop_tag` | `struct double_prop_tag {};` |
| `nntrainer/utils/base_properties.h` | 118 | `str_prop_tag` | `struct str_prop_tag {};` |
| `nntrainer/utils/base_properties.h` | 124 | `bool_prop_tag` | `struct bool_prop_tag {};` |
| `nntrainer/utils/base_properties.h` | 130 | `enum_class_prop_tag` | `struct enum_class_prop_tag {};` |
| `nntrainer/utils/base_properties.h` | 136 | `ptr_prop_tag` | `struct ptr_prop_tag {};` |
| `nntrainer/utils/base_properties.h` | 143 | `Property` | `template <typename T> class Property {` |
| `nntrainer/utils/base_properties.h` | 276 | `EnumProperty` | `class EnumProperty : public Property<typename EnumInfo::Enum> {` |
| `nntrainer/utils/base_properties.h` | 285 | `TensorDimProperty` | `class TensorDimProperty : public Property<TensorDim> {` |
| `nntrainer/utils/base_properties.h` | 298 | `PositiveIntegerProperty` | `class PositiveIntegerProperty : public Property<unsigned int> {` |
| `nntrainer/utils/base_properties.h` | 331 | `tag_cast` | `template <typename... Tags> struct tag_cast;` |
| `nntrainer/utils/base_properties.h` | 339 | `tag_cast` | `template <typename Tag, typename... Others> struct tag_cast<Tag, Others...> {` |
| `nntrainer/utils/base_properties.h` | 351 | `tag_cast` | `struct tag_cast<Tag, BaseTag, Others...> {` |
| `nntrainer/utils/base_properties.h` | 363 | `str_converter` | `template <typename Tag, typename DataType> struct str_converter {` |
| `nntrainer/utils/base_properties.h` | 388 | `str_converter` | `struct str_converter<enum_class_prop_tag, EnumInfo> {` |
| `nntrainer/utils/base_properties.h` | 424 | `str_converter` | `template <typename DataType> struct str_converter<ptr_prop_tag, DataType> {` |
| `nntrainer/utils/base_properties.h` | 662 | `TensorDataTypeInfo` | `struct TensorDataTypeInfo {` |
| `nntrainer/utils/base_properties.h` | 676 | `TensorFormatInfo` | `struct TensorFormatInfo {` |
| `nntrainer/utils/base_properties.h` | 687 | `TensorType_` | `enum class TensorType_ {` |
| `nntrainer/utils/base_properties.h` | 698 | `TensorTypeInfo` | `struct TensorTypeInfo {` |
| `nntrainer/utils/base_properties.h` | 715 | `WeightDtype` | `class WeightDtype final : public EnumProperty<TensorDataTypeInfo> {` |
| `nntrainer/utils/base_properties.h` | 738 | `InputDtype` | `class InputDtype final : public EnumProperty<TensorDataTypeInfo> {` |
| `nntrainer/utils/base_properties.h` | 760 | `TensorDataType` | `class TensorDataType final : public EnumProperty<TensorDataTypeInfo> {` |
| `nntrainer/utils/base_properties.h` | 780 | `TensorFormat` | `class TensorFormat final : public EnumProperty<TensorFormatInfo> {` |
| `nntrainer/utils/base_properties.h` | 800 | `TensorType` | `class TensorType final : public EnumProperty<nntrainer::TensorTypeInfo> {` |
| `nntrainer/utils/base_properties.h` | 812 | `ComputeEngineTypeInfo` | `struct ComputeEngineTypeInfo {` |
| `nntrainer/utils/base_properties.h` | 823 | `ComputeEngine` | `class ComputeEngine final` |
| `nntrainer/utils/dynamic_library_loader.h` | 44 | `DynamicLibraryLoader` | `class DynamicLibraryLoader {` |
| `nntrainer/utils/ini_wrapper.h` | 32 | `IniSection` | `class IniSection {` |
| `nntrainer/utils/ini_wrapper.h` | 232 | `IniWrapper` | `class IniWrapper {` |
| `nntrainer/utils/nntr_threads.h` | 34 | `ParallelBatch` | `class ParallelBatch {` |
| `nntrainer/utils/node_exporter.h` | 40 | `TfOpNode` | `class TfOpNode;` |
| `nntrainer/utils/node_exporter.h` | 49 | `return_type` | `template <ml::train::ExportMethods method> struct return_type {` |
| `nntrainer/utils/node_exporter.h` | 91 | `Exporter` | `class Exporter {` |
| `nntrainer/utils/node_exporter.h` | 218 | `Name` | `class Name;` |
| `nntrainer/utils/node_exporter.h` | 219 | `Unit` | `class Unit;` |
| `nntrainer/utils/node_exporter.h` | 220 | `Flatten` | `class Flatten;` |
| `nntrainer/utils/node_exporter.h` | 221 | `Distribute` | `class Distribute;` |
| `nntrainer/utils/node_exporter.h` | 222 | `Trainable` | `class Trainable;` |
| `nntrainer/utils/node_exporter.h` | 223 | `InputShape` | `class InputShape;` |
| `nntrainer/utils/node_exporter.h` | 224 | `WeightRegularizer` | `class WeightRegularizer;` |
| `nntrainer/utils/node_exporter.h` | 225 | `WeightRegularizerConstant` | `class WeightRegularizerConstant;` |
| `nntrainer/utils/node_exporter.h` | 226 | `WeightInitializer` | `class WeightInitializer;` |
| `nntrainer/utils/node_exporter.h` | 227 | `WeightDecay` | `class WeightDecay;` |
| `nntrainer/utils/node_exporter.h` | 228 | `BiasDecay` | `class BiasDecay;` |
| `nntrainer/utils/node_exporter.h` | 229 | `BiasInitializer` | `class BiasInitializer;` |
| `nntrainer/utils/node_exporter.h` | 230 | `SharedFrom` | `class SharedFrom;` |
| `nntrainer/utils/node_exporter.h` | 231 | `InputConnection` | `class InputConnection;` |
| `nntrainer/utils/node_exporter.h` | 232 | `ClipGradByGlobalNorm` | `class ClipGradByGlobalNorm;` |
| `nntrainer/utils/node_exporter.h` | 233 | `DisableBias` | `class DisableBias;` |
| `nntrainer/utils/node_exporter.h` | 234 | `Activation` | `class Activation;` |
| `nntrainer/utils/node_exporter.h` | 235 | `BatchNormalization` | `class BatchNormalization;` |
| `nntrainer/utils/node_exporter.h` | 236 | `Packed` | `class Packed;` |
| `nntrainer/utils/node_exporter.h` | 237 | `LossScaleForMixed` | `class LossScaleForMixed;` |
| `nntrainer/utils/node_exporter.h` | 238 | `InPlaceProp` | `class InPlaceProp;` |
| `nntrainer/utils/node_exporter.h` | 239 | `InPlaceDirectionProp` | `class InPlaceDirectionProp;` |
| `nntrainer/utils/node_exporter.h` | 240 | `Exponent` | `class Exponent;` |
| `nntrainer/utils/node_exporter.h` | 241 | `StartIndex` | `class StartIndex;` |
| `nntrainer/utils/node_exporter.h` | 242 | `EndIndex` | `class EndIndex;` |
| `nntrainer/utils/node_exporter.h` | 245 | `LayerNode` | `class LayerNode;` |
| `nntrainer/utils/node_exporter.h` | 260 | `BatchNormalizationLayer` | `class BatchNormalizationLayer;` |
| `nntrainer/utils/node_exporter.h` | 273 | `LayerImpl` | `class LayerImpl;` |
| `nntrainer/utils/node_exporter.h` | 287 | `FullyConnectedLayer` | `class FullyConnectedLayer;` |
| `nntrainer/utils/node_exporter.h` | 297 | `ActivationLayer` | `class ActivationLayer;` |
| `nntrainer/utils/node_exporter.h` | 306 | `Conv2DLayer` | `class Conv2DLayer;` |
| `nntrainer/utils/node_exporter.h` | 318 | `InputLayer` | `class InputLayer;` |
| `nntrainer/utils/node_exporter.h` | 328 | `Pooling2DLayer` | `class Pooling2DLayer;` |
| `nntrainer/utils/node_exporter.h` | 339 | `ReshapeLayer` | `class ReshapeLayer;` |
| `nntrainer/utils/node_exporter.h` | 348 | `FlattenLayer` | `class FlattenLayer;` |
| `nntrainer/utils/node_exporter.h` | 357 | `AdditionLayer` | `class AdditionLayer;` |
| `nntrainer/utils/noncopyable.h` | 23 | `Noncopyable` | `class Noncopyable {` |
| `nntrainer/utils/nonmovable.h` | 23 | `Nonmovable` | `class Nonmovable {` |
| `nntrainer/utils/profiler.h` | 89 | `PROFILE_EVENT` | `enum PROFILE_EVENT {` |
| `nntrainer/utils/profiler.h` | 101 | `ProfileEventData` | `struct ProfileEventData {` |
| `nntrainer/utils/profiler.h` | 148 | `Profiler` | `class Profiler;` |
| `nntrainer/utils/profiler.h` | 154 | `ProfileListener` | `class ProfileListener {` |
| `nntrainer/utils/profiler.h` | 204 | `GenericProfileListener` | `class GenericProfileListener : public ProfileListener {` |
| `nntrainer/utils/profiler.h` | 311 | `Profiler` | `class Profiler : public Singleton<Profiler> {` |
| `nntrainer/utils/safetensors_util.cpp` | 72 | `Scanner` | `class Scanner {` |
| `nntrainer/utils/safetensors_util.h` | 28 | `TensorEntry` | `struct TensorEntry {` |
| `nntrainer/utils/singleton.h` | 28 | `Singleton` | `template <typename T> class Singleton : public Noncopyable, public Nonmovable {` |
| `nntrainer/utils/thread_manager.h` | 83 | `ThreadManagerConfig` | `struct ThreadManagerConfig {` |
| `nntrainer/utils/thread_manager.h` | 111 | `CACHELINE_ALIGNED` | `struct CACHELINE_ALIGNED thread_info {` |
| `nntrainer/utils/thread_manager.h` | 119 | `threadpool_command` | `enum threadpool_command {` |
| `nntrainer/utils/thread_manager.h` | 141 | `ThreadManager` | `class ThreadManager : public Singleton<ThreadManager> {` |
| `nntrainer/utils/tracer.h` | 34 | `Tracer` | `class Tracer {` |
| `nntrainer/utils/tracer.h` | 110 | `MemoryTracer` | `class MemoryTracer : public Tracer {` |
| `nntrainer/utils/tracer.h` | 159 | `TimeTracer` | `class TimeTracer : public Tracer {` |
| `nntrainer-windows-resource/arm64/OpenBLAS/openblas_config.h` | 41 | `(anonymous typedef struct)` | `typedef struct {` |
| `nntrainer-windows-resource/arm64/OpenBLAS/openblas_config.h` | 107 | `(anonymous typedef struct)` | `typedef struct { float real, imag; } openblas_complex_float;` |
| `nntrainer-windows-resource/arm64/OpenBLAS/openblas_config.h` | 108 | `(anonymous typedef struct)` | `typedef struct { double real, imag; } openblas_complex_double;` |
| `nntrainer-windows-resource/arm64/OpenBLAS/openblas_config.h` | 109 | `(anonymous typedef struct)` | `typedef struct { xdouble real, imag; } openblas_complex_xdouble;` |
| `nntrainer-windows-resource/x64/OpenBLAS/openblas_config.h` | 45 | `(anonymous typedef struct)` | `typedef struct {` |
| `nntrainer-windows-resource/x64/OpenBLAS/openblas_config.h` | 111 | `(anonymous typedef struct)` | `typedef struct { float real, imag; } openblas_complex_float;` |
| `nntrainer-windows-resource/x64/OpenBLAS/openblas_config.h` | 112 | `(anonymous typedef struct)` | `typedef struct { double real, imag; } openblas_complex_double;` |
| `nntrainer-windows-resource/x64/OpenBLAS/openblas_config.h` | 113 | `(anonymous typedef struct)` | `typedef struct { xdouble real, imag; } openblas_complex_xdouble;` |
| `test/include/nntrainer_test_util.h` | 68 | `ScopedIni` | `class ScopedIni {` |
| `test/include/nntrainer_test_util.h` | 233 | `DataInformation` | `class DataInformation {` |
| `test/include/nntrainer_test_util.h` | 419 | `static_cast_func` | `struct static_cast_func {` |
| `test/include/timer.h` | 26 | `Timer` | `class Timer {` |
| `test/input_gen/gen_layer_tests.py` | 64 | `PositionalEncoding` | `class PositionalEncoding(tf.keras.layers.Layer):` |
| `test/input_gen/gen_layer_tests.py` | 927 | `RMSNorm` | `class RMSNorm(tf.keras.layers.Layer):` |
| `test/input_gen/genModelsMultiout_v2.py` | 14 | `Split` | `class Split(torch.nn.Module):` |
| `test/input_gen/genModelsMultiout_v2.py` | 33 | `SplitAndJoin` | `class SplitAndJoin(torch.nn.Module):` |
| `test/input_gen/genModelsMultiout_v2.py` | 49 | `SplitAndJoinDangle` | `class SplitAndJoinDangle(torch.nn.Module):` |
| `test/input_gen/genModelsMultiout_v2.py` | 82 | `OneToOne` | `class OneToOne(torch.nn.Module):` |
| `test/input_gen/genModelsMultiout_v2.py` | 96 | `OneToMany` | `class OneToMany(torch.nn.Module):` |
| `test/input_gen/genModelsRecurrent_v2.py` | 15 | `FCUnroll` | `class FCUnroll(torch.nn.Module):` |
| `test/input_gen/genModelsRecurrent_v2.py` | 32 | `RNNCellStacked` | `class RNNCellStacked(torch.nn.Module):` |
| `test/input_gen/genModelsRecurrent_v2.py` | 58 | `LSTMStacked` | `class LSTMStacked(torch.nn.Module):` |
| `test/input_gen/genModelsRecurrent_v2.py` | 87 | `LSTMCellStacked` | `class LSTMCellStacked(torch.nn.Module):` |
| `test/input_gen/genModelsRecurrent_v2.py` | 117 | `ZoneoutLSTMStacked` | `class ZoneoutLSTMStacked(torch.nn.Module):` |
| `test/input_gen/genModelsRecurrent_v2.py` | 148 | `GRUCellStacked` | `class GRUCellStacked(torch.nn.Module):` |
| `test/input_gen/genModelsRecurrent_v2.py` | 175 | `GRUCellFC` | `class GRUCellFC(torch.nn.Module):` |
| `test/input_gen/genModelTests_v2.py` | 17 | `ReduceMeanLast` | `class ReduceMeanLast(torch.nn.Module):` |
| `test/input_gen/genModelTests_v2.py` | 30 | `ReduceSumOperation` | `class ReduceSumOperation(torch.nn.Module):` |
| `test/input_gen/genModelTests_v2.py` | 43 | `MolAttention` | `class MolAttention(torch.nn.Module):` |
| `test/input_gen/genModelTests_v2.py` | 100 | `MultiHeadAttention` | `class MultiHeadAttention(torch.nn.Module):` |
| `test/input_gen/genModelTests_v2.py` | 180 | `PositionalEncoding` | `class PositionalEncoding(torch.nn.Module):` |
| `test/input_gen/genModelTests_v2.py` | 205 | `TransformerEncoderLayer` | `class TransformerEncoderLayer(torch.nn.Module):` |
| `test/input_gen/genModelTests_v2.py` | 257 | `TransformerDecoderLayer` | `class TransformerDecoderLayer(torch.nn.Module):` |
| `test/input_gen/genModelTests_v2.py` | 318 | `Transformer` | `class Transformer(torch.nn.Module):` |
| `test/input_gen/genModelTests_v2.py` | 391 | `FCRelu` | `class FCRelu(torch.nn.Module):` |
| `test/input_gen/genModelTests_v2.py` | 426 | `NonTrainableFC` | `class NonTrainableFC(torch.nn.Module):` |
| `test/input_gen/genModelTests_v2.py` | 446 | `LinearMixedPrecision` | `class LinearMixedPrecision(torch.nn.Module):` |
| `test/input_gen/genModelTests_v2.py` | 463 | `AddOperation` | `class AddOperation(torch.nn.Module):` |
| `test/input_gen/genModelTests_v2.py` | 476 | `LinearMixedPrecisionNaNSGD` | `class LinearMixedPrecisionNaNSGD(torch.nn.Module):` |
| `test/input_gen/genModelTests_v2.py` | 495 | `SubtractOperation` | `class SubtractOperation(torch.nn.Module):` |
| `test/input_gen/genModelTests_v2.py` | 508 | `MultiplyOperation` | `class MultiplyOperation(torch.nn.Module):` |
| `test/input_gen/genModelTests_v2.py` | 521 | `DivideOperation` | `class DivideOperation(torch.nn.Module):` |
| `test/input_gen/genModelTests_v2.py` | 534 | `PowOperation` | `class PowOperation(torch.nn.Module):` |
| `test/input_gen/genModelTests_v2.py` | 546 | `SQRTOperation` | `class SQRTOperation(torch.nn.Module):` |
| `test/input_gen/genModelTests_v2.py` | 558 | `NegOperation` | `class NegOperation(torch.nn.Module):` |
| `test/input_gen/genModelTests_v2.py` | 571 | `CosineOperation` | `class CosineOperation(torch.nn.Module):` |
| `test/input_gen/genModelTests_v2.py` | 583 | `SineOperation` | `class SineOperation(torch.nn.Module):` |
| `test/input_gen/genModelTests_v2.py` | 595 | `TangentOperation` | `class TangentOperation(torch.nn.Module):` |
| `test/input_gen/genModelTests_v2.py` | 607 | `MatMulOperation` | `class MatMulOperation(torch.nn.Module):` |
| `test/input_gen/genModelTests_v2.py` | 620 | `ChannelShuffle` | `class ChannelShuffle(torch.nn.Module):` |
| `test/input_gen/transLayer.py` | 25 | `AbstractTransLayer` | `class AbstractTransLayer(K.layers.Layer):` |
| `test/input_gen/transLayer.py` | 85 | `IdentityTransLayer` | `class IdentityTransLayer(AbstractTransLayer):` |
| `test/input_gen/transLayer.py` | 103 | `ChannelLastTransLayer` | `class ChannelLastTransLayer(AbstractTransLayer):` |
| `test/input_gen/transLayer.py` | 166 | `BatchNormTransLayer` | `class BatchNormTransLayer(IdentityTransLayer):` |
| `test/input_gen/transLayer.py` | 186 | `LayerNormTransLayer` | `class LayerNormTransLayer(IdentityTransLayer):` |
| `test/input_gen/transLayer.py` | 213 | `MultiOutLayer` | `class MultiOutLayer(IdentityTransLayer):` |
| `test/input_gen/transLayer.py` | 242 | `RNNCELL_LSTMCellTransLayer` | `class RNNCELL_LSTMCellTransLayer(IdentityTransLayer):` |
| `test/input_gen/transLayer.py` | 259 | `GRUTransLayer` | `class GRUTransLayer(IdentityTransLayer):` |
| `test/input_gen/transLayer.py` | 271 | `GRUCellTransLayer` | `class GRUCellTransLayer(IdentityTransLayer):` |
| `test/input_gen/transLayer.py` | 297 | `MultiHeadAttentionTransLayer` | `class MultiHeadAttentionTransLayer(IdentityTransLayer):` |
| `test/input_gen/zoneout.py` | 14 | `Zoneout` | `class Zoneout(torch.nn.LSTMCell):` |
| `test/nnstreamer/test_nnstreamer_single.cpp` | 28 | `mlInference` | `class mlInference : public testing::Test {` |
| `test/tizen_capi/unittest_tizen_capi_layer.cpp` | 392 | `nntrainerCapiLayerTester` | `class nntrainerCapiLayerTester` |
| `test/unittest/compiler/unittest_interpreter.cpp` | 44 | `nntrainerInterpreterTest` | `class nntrainerInterpreterTest` |
| `test/unittest/datasets/data_producer_common_tests.h` | 35 | `DataProducerSemanticsExpectedResult` | `enum class DataProducerSemanticsExpectedResult {` |
| `test/unittest/datasets/data_producer_common_tests.h` | 52 | `DataProducerSemantics` | `class DataProducerSemantics` |
| `test/unittest/datasets/unittest_iteration_queue.cpp` | 41 | `IterQueueScenarios` | `class IterQueueScenarios` |
| `test/unittest/integration_tests/integration_test_fsu.cpp` | 168 | `LookAheadParm` | `class LookAheadParm` |
| `test/unittest/layers/layers_common_tests.h` | 53 | `LayerSemantics` | `class LayerSemantics` |
| `test/unittest/layers/layers_common_tests.h` | 94 | `LayerSemanticsGpu` | `class LayerSemanticsGpu : public LayerSemantics {};` |
| `test/unittest/layers/layers_common_tests.h` | 100 | `LayerPropertySemantics` | `class LayerPropertySemantics : public LayerSemantics {};` |
| `test/unittest/layers/layers_common_tests.h` | 134 | `LayerGoldenTest` | `class LayerGoldenTest` |
| `test/unittest/layers/layers_golden_tests.cpp` | 53 | `shape_parser_` | `struct shape_parser_ : Property<TensorDim> {` |
| `test/unittest/layers/unittest_kv_cache_manager.cpp` | 29 | `KVCacheManagerTest` | `class KVCacheManagerTest : public ::testing::Test {` |
| `test/unittest/layers/unittest_layers_impl.cpp` | 28 | `MockLayer` | `class MockLayer final : public LayerImpl {` |
| `test/unittest/memory/unittest_cache_loader.cpp` | 28 | `CacheLoaderTest` | `class CacheLoaderTest : public ::testing::Test {` |
| `test/unittest/memory/unittest_cache_pool.cpp` | 32 | `MockCachePool` | `class MockCachePool : public nntrainer::CachePool {` |
| `test/unittest/memory/unittest_cache_pool.cpp` | 57 | `CachePoolTest` | `class CachePoolTest : public ::testing::Test {` |
| `test/unittest/memory/unittest_cache_pool_fsu.cpp` | 23 | `MockCachePoolFSU` | `class MockCachePoolFSU : public nntrainer::CachePool {` |
| `test/unittest/memory/unittest_cache_pool_fsu.cpp` | 50 | `CachePoolFSUTest` | `class CachePoolFSUTest : public ::testing::Test {` |
| `test/unittest/memory/unittest_memory_planner.cpp` | 32 | `MemoryPlannerValidate` | `class MemoryPlannerValidate` |
| `test/unittest/memory/unittest_memory_pool.cpp` | 32 | `MemoryPoolTest` | `class MemoryPoolTest` |
| `test/unittest/memory/unittest_memory_pool.cpp` | 454 | `CountingAllocator` | `class CountingAllocator : public nntrainer::MemAllocator {` |
| `test/unittest/models/causallm_test_utils.h` | 30 | `TinyCausalLMFiles` | `struct TinyCausalLMFiles {` |
| `test/unittest/models/causallm_test_utils.h` | 39 | `TinyCausalLMConfig` | `struct TinyCausalLMConfig {` |
| `test/unittest/models/causallm_test_utils.h` | 48 | `TinyCausalLMDataType` | `struct TinyCausalLMDataType {` |
| `test/unittest/models/causallm_test_utils.h` | 59 | `TinyCausalLMExpectedLogits` | `struct TinyCausalLMExpectedLogits {` |
| `test/unittest/models/causallm_test_utils.h` | 68 | `TinyCausalLMRunner` | `class TinyCausalLMRunner {` |
| `test/unittest/models/causallm_test_utils.h` | 157 | `TinyCausalLMCase` | `struct TinyCausalLMCase {` |
| `test/unittest/models/models_golden_test.h` | 27 | `NeuralNetwork` | `class NeuralNetwork;` |
| `test/unittest/models/models_golden_test.h` | 66 | `nntrainerModelTest` | `class nntrainerModelTest` |
| `test/unittest/models/models_test_utils.cpp` | 133 | `IterationForGolden` | `class IterationForGolden {` |
| `test/unittest/models/models_test_utils.h` | 31 | `NodeWatcher` | `class NodeWatcher {` |
| `test/unittest/models/models_test_utils.h` | 154 | `GraphWatcher` | `class GraphWatcher {` |
| `test/unittest/models/unittest_causallm_gemma3.cpp` | 37 | `TinyGemma3CausalLM` | `class TinyGemma3CausalLM final : public causallm::Gemma3CausalLM,` |
| `test/unittest/models/unittest_causallm_gemma3.cpp` | 199 | `TinyEmbeddingGemmaFiles` | `struct TinyEmbeddingGemmaFiles {` |
| `test/unittest/models/unittest_causallm_gemma3.cpp` | 209 | `TinyEmbeddingGemma` | `class TinyEmbeddingGemma final : public causallm::EmbeddingGemma {` |
| `test/unittest/models/unittest_causallm_gemma3.cpp` | 553 | `Gemma3TinyModelTest` | `class Gemma3TinyModelTest` |
| `test/unittest/models/unittest_causallm_qwen2.cpp` | 38 | `TinyQwen2CausalLM` | `class TinyQwen2CausalLM final : public causallm::Qwen2CausalLM,` |
| `test/unittest/models/unittest_causallm_qwen2.cpp` | 200 | `TinyQwen25EmbeddingFiles` | `struct TinyQwen25EmbeddingFiles {` |
| `test/unittest/models/unittest_causallm_qwen2.cpp` | 210 | `TinyQwen25Embedding` | `class TinyQwen25Embedding final : public causallm::Qwen2Embedding {` |
| `test/unittest/models/unittest_causallm_qwen2.cpp` | 523 | `Qwen2CausalLMTinyModelTest` | `class Qwen2CausalLMTinyModelTest` |
| `test/unittest/models/unittest_causallm_qwen3.cpp` | 38 | `TinyQwen3CausalLM` | `class TinyQwen3CausalLM final : public causallm::Qwen3CausalLM,` |
| `test/unittest/models/unittest_causallm_qwen3.cpp` | 200 | `TinyQwen3EmbeddingFiles` | `struct TinyQwen3EmbeddingFiles {` |
| `test/unittest/models/unittest_causallm_qwen3.cpp` | 210 | `TinyQwen3Embedding` | `class TinyQwen3Embedding final : public causallm::Qwen3Embedding {` |
| `test/unittest/models/unittest_causallm_qwen3.cpp` | 512 | `CausalLMTinyModelTest` | `class CausalLMTinyModelTest` |
| `test/unittest/unittest_base_properties.cpp` | 31 | `banana_prop_tag` | `struct banana_prop_tag : nntrainer::int_prop_tag {};` |
| `test/unittest/unittest_base_properties.cpp` | 37 | `NumBanana` | `class NumBanana : public nntrainer::Property<int> {` |
| `test/unittest/unittest_base_properties.cpp` | 50 | `QualityOfBanana` | `class QualityOfBanana : public nntrainer::Property<std::string> {` |
| `test/unittest/unittest_base_properties.cpp` | 67 | `MarkAsGoodBanana` | `class MarkAsGoodBanana : public nntrainer::Property<bool> {` |
| `test/unittest/unittest_base_properties.cpp` | 78 | `FreshnessOfBanana` | `class FreshnessOfBanana : public nntrainer::Property<float> {` |
| `test/unittest/unittest_base_properties.cpp` | 90 | `DimensionOfBanana` | `class DimensionOfBanana : public nntrainer::Property<nntrainer::TensorDim> {` |
| `test/unittest/unittest_base_properties.cpp` | 104 | `PtrOfBanana` | `class PtrOfBanana : public nntrainer::Property<int *> {` |
| `test/unittest/unittest_base_properties.cpp` | 114 | `BananaEnumInfo` | `struct BananaEnumInfo {` |
| `test/unittest/unittest_base_properties.cpp` | 119 | `Enum` | `enum class Enum {` |
| `test/unittest/unittest_base_properties.cpp` | 135 | `BananaType` | `class BananaType : public nntrainer::EnumProperty<BananaEnumInfo> {` |
| `test/unittest/unittest_common_properties.cpp` | 24 | `NamePropertyTest` | `class NamePropertyTest : public ::testing::TestWithParam<std::string> {};` |
| `test/unittest/unittest_common_properties.cpp` | 42 | `NameTest` | `class NameTest : public ::testing::TestWithParam<std::string> {` |
| `test/unittest/unittest_compute_ops_dispatch.cpp` | 34 | `CallCounters` | `struct CallCounters {` |
| `test/unittest/unittest_compute_ops_dispatch.cpp` | 51 | `MockComputeOps` | `class MockComputeOps : public nntrainer::ComputeOps {` |
| `test/unittest/unittest_compute_ops_dispatch.cpp` | 93 | `ComputeOpsDispatchTest` | `class ComputeOpsDispatchTest : public ::testing::Test {` |
| `test/unittest/unittest_nntrainer_appcontext.cpp` | 33 | `nntrainerAppContextDirectory` | `class nntrainerAppContextDirectory : public ::testing::Test {` |
| `test/unittest/unittest_nntrainer_appcontext.cpp` | 106 | `CustomOptimizer` | `class CustomOptimizer : public nntrainer::Optimizer {` |
| `test/unittest/unittest_nntrainer_appcontext.cpp` | 127 | `CustomOptimizer2` | `class CustomOptimizer2 : public nntrainer::Optimizer {` |
| `test/unittest/unittest_nntrainer_appcontext.cpp` | 147 | `CustomLayer` | `class CustomLayer : public nntrainer::Layer {` |
| `test/unittest/unittest_nntrainer_appcontext.cpp` | 171 | `AppContextTest` | `class AppContextTest` |
| `test/unittest/unittest_nntrainer_cpu_backend.cpp` | 71 | `(anonymous typedef struct)` | `typedef struct {` |
| `test/unittest/unittest_nntrainer_cpu_backend.cpp` | 79 | `(anonymous typedef struct)` | `typedef struct {` |
| `test/unittest/unittest_nntrainer_cpu_backend.cpp` | 94 | `(anonymous typedef struct)` | `typedef struct {` |
| `test/unittest/unittest_nntrainer_cpu_backend.cpp` | 103 | `block_q4_Kx8_testonly` | `struct block_q4_Kx8_testonly {` |
| `test/unittest/unittest_nntrainer_cpu_backend.cpp` | 111 | `(anonymous typedef struct)` | `typedef struct {` |
| `test/unittest/unittest_nntrainer_cpu_backend_fp16.cpp` | 52 | `(anonymous typedef struct)` | `typedef struct {` |
| `test/unittest/unittest_nntrainer_cpu_backend_fp16.cpp` | 61 | `(anonymous typedef struct)` | `typedef struct {` |
| `test/unittest/unittest_nntrainer_cpu_backend_fp16.cpp` | 68 | `(anonymous typedef struct)` | `typedef struct {` |
| `test/unittest/unittest_nntrainer_exe_order.cpp` | 38 | `nntrainerExeOrderTest` | `class nntrainerExeOrderTest` |
| `test/unittest/unittest_nntrainer_graph.cpp` | 56 | `nntrainerGraphTest` | `class nntrainerGraphTest` |
| `test/unittest/unittest_nntrainer_lazy_tensor.cpp` | 26 | `nntrainer_LazyTensorOpsTest` | `class nntrainer_LazyTensorOpsTest : public ::testing::Test {` |
| `test/unittest/unittest_nntrainer_modelfile.cpp` | 38 | `nntrainerIniTest` | `class nntrainerIniTest` |
| `test/unittest/unittest_nntrainer_profiler.cpp` | 28 | `MockProfileListener` | `class MockProfileListener : public ProfileListener {` |
| `test/unittest/unittest_nntrainer_profiler.cpp` | 84 | `ProfileTest` | `class ProfileTest : public ::testing::Test {` |
| `test/unittest/unittest_nntrainer_task.cpp` | 30 | `MockTaskExecutor` | `class MockTaskExecutor : public nntrainer::TaskExecutor {` |
| `test/unittest/unittest_nntrainer_task.cpp` | 54 | `TaskExecutorTest` | `class TaskExecutorTest : public ::testing::Test {` |
| `test/unittest/unittest_opencl_kernels_qk_k.cpp` | 258 | `block_q4_0` | `struct block_q4_0 {` |
| `test/unittest/unittest_opencl_kernels_qk_k.cpp` | 397 | `block_q4_0` | `struct block_q4_0 {` |
