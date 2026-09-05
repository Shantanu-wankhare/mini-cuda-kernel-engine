// =============================================================================
//  src/tensor/tensor.cpp -- Storage (owning) and Tensor (view). Phase 4.
//
//  Host-side CUDA runtime calls only (cudaMemcpyAsync); no kernels, so this is
//  a .cpp and needs no nvcc -- the boundary CLAUDE.md section 5 defines.
// =============================================================================
#include "mcke/tensor/tensor.hpp"

#include <cstdio>
#include <cstring>
#include <sstream>

#if MCKE_WITH_CUDA
#include "mcke/runtime/cuda_check.hpp"
#endif

namespace mcke {

// -----------------------------------------------------------------------------
// Storage
// -----------------------------------------------------------------------------
StatusOr<std::shared_ptr<Storage>> Storage::create(DeviceAllocator& alloc,
                                                   std::size_t bytes,
                                                   rt::StreamHandle stream) {
  if (bytes == 0) return InvalidArgumentError("Storage::create: zero bytes");
  MCKE_ASSIGN_OR_RETURN(Allocation a, alloc.allocate(bytes, stream));
  // NOT std::make_shared: the constructor is private (Storage::create is the
  // declared factory, so allocation failure lands in the return type rather than
  // in an exception). make_shared constructs in place and cannot reach a private
  // ctor, so it fails with a distinctly unhelpful __construct_at error. The cost
  // is one extra allocation for the control block, which is irrelevant here --
  // a Storage is created once per arena, not per tensor.
  return std::shared_ptr<Storage>(new Storage(alloc, a, stream));
}

Storage::~Storage() {
  if (!alloc_.ptr) return;
  const Status s = allocator_.deallocate(alloc_, last_use_stream_);
  if (!s.ok()) {
    // A destructor cannot return, and throwing here would terminate during
    // unwinding. Warn instead -- same choice the buddy allocator's destructor
    // makes, and for the same reason.
    std::fprintf(stderr, "[mcke] WARNING: Storage::~Storage deallocate failed: %s\n",
                 s.to_string().c_str());
  }
}

// -----------------------------------------------------------------------------
// Tensor
// -----------------------------------------------------------------------------
StatusOr<Tensor> Tensor::empty(DeviceAllocator& alloc, Shape shape, DType dtype,
                               rt::StreamHandle stream) {
  const std::size_t bytes = shape.bytes(dtype);
  if (bytes == 0) return InvalidArgumentError("Tensor::empty: shape has zero elements");
  MCKE_ASSIGN_OR_RETURN(std::shared_ptr<Storage> s, Storage::create(alloc, bytes, stream));
  return Tensor(std::move(s), 0, shape, dtype);
}

StatusOr<Tensor> Tensor::from_storage(std::shared_ptr<Storage> s, std::size_t byte_offset,
                                      Shape shape, DType dtype) {
  if (!s) return InvalidArgumentError("Tensor::from_storage: null storage");
  const std::size_t need = shape.bytes(dtype);
  if (byte_offset + need > s->bytes())
    return InvalidArgumentError(
        "Tensor::from_storage: view [" + std::to_string(byte_offset) + ", " +
        std::to_string(byte_offset + need) + ") does not fit storage of " +
        std::to_string(s->bytes()) + " bytes");
  // Refuse rather than degrade. A view whose base is not element-aligned is not
  // merely slow -- vectorized loads in the Phase 3 kernels assume it, and a
  // float4 load from a 4-byte-aligned address faults on real hardware.
  if (byte_offset % dtype_size(dtype) != 0)
    return InvalidArgumentError("Tensor::from_storage: byte offset " +
                                std::to_string(byte_offset) + " is not a multiple of the "
                                "dtype size " + std::to_string(dtype_size(dtype)));
  return Tensor(std::move(s), byte_offset, shape, dtype);
}

StatusOr<Tensor> Tensor::reshape(Shape new_shape) const {
  if (!storage_) return FailedPreconditionError("Tensor::reshape: undefined tensor");
  if (new_shape.numel() != shape_.numel())
    return InvalidArgumentError("Tensor::reshape: element count differs (" +
                                std::to_string(new_shape.numel()) + " vs " +
                                std::to_string(shape_.numel()) + ")");
  return Tensor(storage_, byte_offset_, new_shape, dtype_);
}

StatusOr<Tensor> Tensor::slice(dim_t elem_offset, Shape shape) const {
  if (!storage_) return FailedPreconditionError("Tensor::slice: undefined tensor");
  if (elem_offset < 0)
    return InvalidArgumentError("Tensor::slice: negative element offset");
  if (elem_offset + shape.numel() > shape_.numel())
    return InvalidArgumentError("Tensor::slice: [" + std::to_string(elem_offset) + ", " +
                                std::to_string(elem_offset + shape.numel()) +
                                ") exceeds parent numel " + std::to_string(shape_.numel()));
  return Tensor(storage_,
                byte_offset_ + static_cast<std::size_t>(elem_offset) * dtype_size(dtype_),
                shape, dtype_);
}

std::string Tensor::to_string() const {
  std::ostringstream os;
  os << "Tensor" << shape_.to_string() << " " << dtype_name(dtype_);
  if (!storage_) { os << " <undefined>"; return os.str(); }
  os << " @+" << byte_offset_ << " of " << storage_->bytes() << " B";
  return os.str();
}

Status Tensor::copy_from_host(const void* src, std::size_t bytes, rt::StreamHandle stream) {
  if (!storage_) return FailedPreconditionError("Tensor::copy_from_host: undefined tensor");
  if (bytes > nbytes())
    return InvalidArgumentError("Tensor::copy_from_host: " + std::to_string(bytes) +
                                " bytes into a tensor of " + std::to_string(nbytes()));
#if MCKE_WITH_CUDA
  MCKE_CUDA_RETURN_IF_ERROR(
      cudaMemcpyAsync(data_ptr(), src, bytes, cudaMemcpyHostToDevice, stream));
#else
  // In a host-only build raw_device_malloc hands back host memory, so a plain
  // memcpy is the correct backend -- which is what lets the whole graph be
  // planned and executed on a laptop.
  (void)stream;
  std::memcpy(data_ptr(), src, bytes);
#endif
  note_use(stream);
  return OkStatus();
}

Status Tensor::copy_to_host(void* dst, std::size_t bytes, rt::StreamHandle stream) const {
  if (!storage_) return FailedPreconditionError("Tensor::copy_to_host: undefined tensor");
  if (bytes > nbytes())
    return InvalidArgumentError("Tensor::copy_to_host: " + std::to_string(bytes) +
                                " bytes from a tensor of " + std::to_string(nbytes()));
#if MCKE_WITH_CUDA
  MCKE_CUDA_RETURN_IF_ERROR(
      cudaMemcpyAsync(dst, data_ptr(), bytes, cudaMemcpyDeviceToHost, stream));
#else
  (void)stream;
  std::memcpy(dst, data_ptr(), bytes);
#endif
  note_use(stream);
  return OkStatus();
}

}  // namespace mcke
