// Minimal conditional-graph reproducer, independent of all solver code.
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>
void Check(cudaError_t status, const char *where) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(where) + ": " +
                             cudaGetErrorString(status));
}
__global__ void ResetInner(cudaGraphConditionalHandle inner, int *counts) {
  counts[1] = 2;
  cudaGraphSetConditional(inner, 1);
}
__global__ void Loop(cudaGraphConditionalHandle handle, int *counts, int slot) {
  ++counts[2];
  cudaGraphSetConditional(handle, --counts[slot] ? 1 : 0);
}
__global__ void Outer(cudaGraphConditionalHandle handle, int *counts) {
  cudaGraphSetConditional(handle, --counts[0] ? 1 : 0);
}
template <class F, class... A>
void Kernel(cudaGraph_t graph, cudaGraphNode_t &tail, F fn, A... args) {
  void *parameters[]{static_cast<void *>(&args)...};
  cudaKernelNodeParams p{};
  p.func = reinterpret_cast<void *>(fn);
  p.gridDim = p.blockDim = dim3(1);
  p.kernelParams = parameters;
  cudaGraphNode_t next{};
  Check(cudaGraphAddKernelNode(&next, graph, tail ? &tail : nullptr,
                               tail ? 1 : 0, &p),
        "kernel node");
  tail = next;
}
cudaGraph_t Conditional(cudaGraph_t graph, cudaGraphNode_t &tail,
                        cudaGraphConditionalHandle handle) {
  cudaGraphNodeParams p{};
  p.type = cudaGraphNodeTypeConditional;
  p.conditional.handle = handle;
  p.conditional.type = cudaGraphCondTypeWhile;
  p.conditional.size = 1;
  cudaGraphNode_t next{};
  Check(cudaGraphAddNode(&next, graph, tail ? &tail : nullptr, nullptr,
                         tail ? 1 : 0, &p),
        "conditional node");
  tail = next;
  return p.conditional.phGraph_out[0];
}
int main(int argc, char **argv) {
  try {
    bool nested = argc == 2 && std::string(argv[1]) == "nested";
    if (argc > 2 || (argc == 2 && !nested))
      return 2;
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    cudaStream_t stream{};
    int *counts = nullptr;
    int initial[]{3, 0, 0};
    Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    Check(cudaMalloc(&counts, sizeof(initial)), "allocate");
    Check(cudaMemcpy(counts, initial, sizeof(initial), cudaMemcpyHostToDevice),
          "initialize");
    Check(cudaGraphCreate(&graph, 0), "graph create");
    cudaGraphConditionalHandle outer{};
    Check(cudaGraphConditionalHandleCreate(&outer, graph, 1,
                                           cudaGraphCondAssignDefault),
          "outer handle");
    cudaGraphNode_t tail{};
    auto body = Conditional(graph, tail, outer);
    cudaGraphNode_t body_tail{};
    if (nested) {
      cudaGraphConditionalHandle inner{};
      Check(cudaGraphConditionalHandleCreate(&inner, body, 0,
                                             cudaGraphCondAssignDefault),
            "inner handle");
      Kernel(body, body_tail, ResetInner, inner, counts);
      auto inner_body = Conditional(body, body_tail, inner);
      cudaGraphNode_t inner_tail{};
      Kernel(inner_body, inner_tail, Loop, inner, counts, 1);
      Kernel(body, body_tail, Outer, outer, counts);
    } else
      Kernel(body, body_tail, Loop, outer, counts, 0);
    Check(cudaGraphInstantiate(&executable, graph, 0), "instantiate");
    Check(cudaGraphLaunch(executable, stream), "launch");
    Check(cudaStreamSynchronize(stream), "complete");
    Check(cudaMemcpy(initial, counts, sizeof(initial), cudaMemcpyDeviceToHost),
          "readback");
    if (initial[0] != 0 || initial[2] != (nested ? 6 : 3))
      throw std::runtime_error("wrong graph result");
    std::cout << "conditional graph pass: " << initial[2] << " updates\n";
    Check(cudaGraphExecDestroy(executable), "exec destroy");
    Check(cudaGraphDestroy(graph), "graph destroy");
    Check(cudaFree(counts), "free");
    Check(cudaStreamDestroy(stream), "stream destroy");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
