# Communication Layer Design

## Architecture Overview

The communication layer follows the same pattern as the runtime API:
- C API header with function pointers (include/llaisys/comm.h)
- C++ dispatcher interface (src/device/comm_api.hpp)
- Backend dispatcher implementation (src/device/comm_api.cpp)

## Design Decisions

### 1. Backend Abstraction
Three communication backends supported:
- NCCL (NVIDIA Collective Communications Library)
- IXCCL (Iluvatar collective communications)
- MPI (Message Passing Interface)

### 2. Core Operations
Minimal set of collective operations:
- init/destroy: Communicator lifecycle
- get_rank/get_size: Process identification
- allreduce: Collective reduction (sum/prod/min/max)
- broadcast: One-to-all communication
- send/recv: Point-to-point communication

### 3. Stream Integration
All communication operations accept llaisysStream_t for async execution,
matching the runtime API pattern.

### 4. Type Safety
Uses existing llaisysDataType_t enum for data types.

## Implementation Notes

- Dispatcher returns unsupported API stub if backend not available
- Backend implementations will be in separate files (nccl/, ixccl/, mpi/)
- Follows EXCEPTION_UNSUPPORTED_DEVICE pattern for error handling
