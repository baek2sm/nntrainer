# L0 Visual System Map

> **Layer 0.** Start here when you need to understand the system shape before
> reading code. This page is intentionally diagram-heavy. It explains what
> exists, how the major pieces connect, and which core classes carry each
> responsibility.

---

## 1. One-sentence model

NNTrainer is a model execution runtime: public APIs and applications build a
model graph, the graph wraps concrete layers, layers request tensors and
weights, and tensor operations dispatch through a backend context to CPU,
OpenCL, or QNN.

---

## 2. System overview

```mermaid
flowchart TD
  User[User code / application] --> API[Public API<br/>C API / C++ API]
  App[Applications<br/>CausalLM, Android, examples] --> API
  App --> AppModels[Application model builders<br/>Transformer / CausalLM / custom layers]

  API --> Model[NeuralNetwork<br/>model lifecycle owner]
  AppModels --> Model

  Model --> Compiler[Compiler / interpreters<br/>INI / ONNX / TFLite / realizers]
  Model --> DirectBuild[C++ graph builders<br/>createLayer based APIs]
  Compiler --> Graph[NetworkGraph<br/>executable graph]
  DirectBuild --> Graph

  Graph --> Node[LayerNode<br/>graph adapter for one Layer]
  Node --> Layer[Layer / LayerImpl<br/>concrete math and state]
  Layer --> Ctx[InitLayerContext / RunLayerContext]

  Ctx --> Tensor[Tensor / Weight / Var_Grad]
  Tensor --> Memory[Manager / TensorPool / MemoryPool<br/>lifetime and allocation]
  Tensor --> Dispatch[ComputeOps dispatch]
  Dispatch --> Backend[CPU / OpenCL / QNN]

  Engine[Engine] --> Context[Context registry]
  Context --> AppContext[AppContext CPU]
  Context --> ClContext[ClContext OpenCL]
  Context --> QNNContext[QNNContext]
  AppContext --> Dispatch
  ClContext --> Dispatch
  QNNContext --> Dispatch
```

The most important split is this:

- `NeuralNetwork` owns lifecycle.
- `NetworkGraph` owns execution structure.
- `LayerNode` connects graph semantics to concrete `Layer` instances.
- `Tensor`/`Manager` own storage and lifetime.
- `Engine`/`Context`/`ComputeOps` own backend dispatch.

---

## 3. Runtime lifecycle

```mermaid
sequenceDiagram
  participant Caller as Caller / App
  participant Model as NeuralNetwork
  participant Compiler as GraphCompiler
  participant Graph as NetworkGraph
  participant Node as LayerNode
  participant Manager as Manager
  participant Backend as Context / ComputeOps

  Caller->>Model: create model / add layers / load file
  Model->>Compiler: parse or realize graph description
  Compiler->>Graph: produce graph nodes and connections
  Model->>Graph: compile()
  Graph->>Node: validate ports and execution order
  Graph->>Manager: request tensors and weights
  Graph->>Backend: resolve ContextData for selected engine
  Backend-->>Graph: allocator + ComputeOps
  Graph->>Manager: allocate / plan memory
  Caller->>Model: train() or inference()
  Model->>Graph: forward / backward / apply
  Graph->>Node: ordered layer execution
  Node->>Backend: tensor ops through ComputeOps
```

This is why debugging usually starts from one of two questions:

1. Did graph construction produce the right `LayerNode` topology?
2. Did runtime tensors carry the right `ContextData` and dtype?

---

## 4. Core class relationships

```mermaid
classDiagram
  class Model {
    <<public API>>
  }
  class NeuralNetwork {
    compile()
    initialize()
    train()
    inference()
    save()
  }
  class GraphCompiler
  class GraphInterpreter
  class GraphRealizer
  class NetworkGraph {
    compile()
    initialize()
    forwarding()
    backwarding()
  }
  class GraphCore
  class GraphNode
  class LayerNode
  class Layer
  class LayerImpl
  class InitLayerContext
  class RunLayerContext

  Model <|-- NeuralNetwork
  NeuralNetwork --> GraphCompiler
  GraphCompiler --> GraphInterpreter
  GraphCompiler --> GraphRealizer
  NeuralNetwork --> NetworkGraph
  NetworkGraph --> GraphCore
  GraphCore --> GraphNode
  GraphNode <|-- LayerNode
  LayerNode --> Layer
  Layer <|-- LayerImpl
  Layer --> InitLayerContext
  Layer --> RunLayerContext
```

Read this as ownership and call direction, not as a complete inheritance map.
The full declaration inventory is in
[`09-class-map-inventory.md`](09-class-map-inventory.md).

---

## 5. Tensor, memory, and backend dispatch

```mermaid
classDiagram
  class Tensor {
    public handle
  }
  class TensorBase {
    shape
    dtype
    ContextData
  }
  class Var_Grad
  class Weight
  class Manager
  class TensorPool
  class MemoryPool
  class CachePool
  class MemoryPlanner
  class ComputeOps
  class ContextData
  class MemAllocator
  class Context
  class AppContext
  class ClContext
  class QNNContext
  class Engine

  Tensor --> TensorBase
  Var_Grad --> Tensor
  Weight --|> Var_Grad
  Manager --> TensorPool
  Manager --> MemoryPool
  MemoryPool <|-- CachePool
  MemoryPool --> MemoryPlanner
  TensorBase --> ContextData
  ContextData --> ComputeOps
  ContextData --> MemAllocator
  Engine --> Context
  Context <|-- AppContext
  Context <|-- ClContext
  Context <|-- QNNContext
  Context --> ContextData
```

Backend selection is not scattered across tensor math. The backend identity is
carried by `ContextData` on `TensorBase`, and tensor operations route through
the `ComputeOps` table.

---

## 6. CausalLM as the main application stack

```mermaid
flowchart TD
  Config[model directory<br/>config + tokenizer + weights] --> Main[main.cpp / C API]
  Main --> Factory[architecture factory]
  Factory --> Transformer[Transformer<br/>shared runtime]
  Transformer --> CausalLM[CausalLM<br/>decoder generation]
  Transformer --> Sentence[SentenceTransformer<br/>embedding path]

  CausalLM --> Family[Qwen / Gemma / GPT-OSS / MoE families]
  Sentence --> Embedding[Qwen / Gemma / DeBERTa embeddings]
  Family --> CreateLayer[createLayer composition]
  Embedding --> CreateLayer
  CreateLayer --> Custom[Application custom LayerImpl classes]
  Custom --> Core[NNTrainer core graph runtime]
```

CausalLM is not a thin sample. It is an application-level model system that
uses the same core graph runtime, but it builds graphs from C++ model classes
instead of only from INI/ONNX/TFLite input files.

---

## 7. Navigation by task

| If you need to understand... | Start here | Then read |
|---|---|---|
| Whole system flow | This page | [`01-container-view.md`](01-container-view.md) |
| Core model lifecycle | `NeuralNetwork -> NetworkGraph` diagrams above | [`02-components/models.md`](02-components/models.md), [`02-components/graph.md`](02-components/graph.md) |
| Layer behavior | `LayerNode -> Layer -> Context` diagrams above | [`02-components/layers.md`](02-components/layers.md) |
| Tensor dtype / memory / dispatch | Tensor/backend diagram above | [`02-components/tensor.md`](02-components/tensor.md), [`02-components/backends.md`](02-components/backends.md) |
| CausalLM implementation | CausalLM diagram above | [`06-application-surface-causallm.md`](06-application-surface-causallm.md) |
| Exact class declaration | The owning diagram first | [`09-class-map.md`](09-class-map.md), [`09-class-map-inventory.md`](09-class-map-inventory.md) |
| Whether a source file was scanned | Generated coverage | [`09-source-file-coverage.md`](09-source-file-coverage.md) |

---

## 8. Mental checklist

Before opening code, identify:

1. Entry path: public API, CausalLM, test, or tool.
2. Graph source: parsed file, programmatic layer API, or C++ model builder.
3. Runtime owner: `NeuralNetwork` or application wrapper.
4. Execution graph: `NetworkGraph` and `LayerNode`.
5. Storage path: `Tensor`, `Weight`, `Manager`, and pools.
6. Backend path: `Engine`, `Context`, `ContextData`, and `ComputeOps`.

If those six are clear, the implementation files become much easier to search.
