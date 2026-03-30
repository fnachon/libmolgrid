/** \file managed_grid.h
 *
 * A grid that manages its own memory using a shared pointer.
 * Ported to Apple Silicon MPS (unified memory).
 *
 * On Apple Silicon the CPU and GPU share the same physical memory pool, so
 * there is no need for explicit host↔device copies.  The buffer is allocated
 * once with malloc() and remains accessible from both CPU and Metal GPU
 * kernels throughout its lifetime.
 *
 * The togpu() / tocpu() methods still exist (API compatibility) but they no
 * longer copy data; they simply record which side is "active" so that the
 * MetalContext can insert the appropriate synchronization barriers.
 */

#ifndef MANAGED_GRID_H_
#define MANAGED_GRID_H_

#include <memory>
#include <utility>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/array.hpp>
#include "libmolgrid/grid.h"

namespace libmolgrid
{

  template <typename Dtype>
  struct mgrid_buffer_data
  {
    bool sent_to_gpu;
  };

  /** \brief ManagedGrid base class */
  template <typename Dtype, std::size_t NumDims>
  class ManagedGridBase
  {
  public:
    using gpu_grid_t = Grid<Dtype, NumDims, true>;
    using cpu_grid_t = Grid<Dtype, NumDims, false>;
    using type = Dtype;
    static constexpr size_t N = NumDims;

  protected:
    // On unified memory both views point at the same allocation.
    mutable gpu_grid_t gpu_grid;
    cpu_grid_t cpu_grid;
    std::shared_ptr<Dtype> cpu_ptr;
    size_t capacity = 0;

    using buffer_data = mgrid_buffer_data<Dtype>;
    mutable buffer_data *gpu_info = nullptr;

    /// empty (unusable) grid
    ManagedGridBase() = default;

    // Allocate unified memory and set up cpu/gpu grid views.
    // The data is page-aligned (16 KB) so Metal's newBufferWithBytesNoCopy works correctly.
    // The buffer_data header is allocated separately.
    void alloc_and_set_cpu(size_t sz)
    {
      // Allocate the metadata header.
      gpu_info = new buffer_data;
      gpu_info->sent_to_gpu = false;

      if (sz == 0) {
        buffer_data *info = gpu_info;
        cpu_ptr = std::shared_ptr<Dtype>(nullptr, [info](Dtype*) { delete info; });
        cpu_grid.set_buffer(nullptr);
        gpu_grid.set_buffer(nullptr);
        return;
      }

      // Allocate data with 16 KB page alignment for Metal zero-copy compatibility.
      void *data = nullptr;
      const size_t align = 16384;  // Apple Silicon page size
      if (posix_memalign(&data, align, sz * sizeof(Dtype)) != 0 || !data) {
        delete gpu_info;
        gpu_info = nullptr;
        throw std::runtime_error("Could not allocate " + itoa(sz * sizeof(Dtype)) +
                                 " bytes of page-aligned CPU memory in ManagedGrid");
      }

      buffer_data *info = gpu_info;
      cpu_ptr = std::shared_ptr<Dtype>((Dtype*)data, [info](Dtype* p) {
        free(p);
        delete info;
      });
      cpu_grid.set_buffer(cpu_ptr.get());
      // GPU grid points at the same allocation (unified memory).
      gpu_grid.set_buffer(cpu_ptr.get());
    }

    template <typename... I, typename = typename std::enable_if<sizeof...(I) == NumDims>::type>
    ManagedGridBase(I... sizes) : gpu_grid(nullptr, sizes...), cpu_grid(nullptr, sizes...)
    {
      capacity = this->size();
      alloc_and_set_cpu(capacity);
      memset(cpu_ptr.get(), 0, capacity * sizeof(Dtype));
      gpu_info->sent_to_gpu = false;
    }

    ManagedGridBase(size_t *sizes) : gpu_grid(nullptr, sizes), cpu_grid(nullptr, sizes)
    {
      capacity = this->size();
      alloc_and_set_cpu(capacity);
      memset(cpu_ptr.get(), 0, capacity * sizeof(Dtype));
      gpu_info->sent_to_gpu = false;
    }

    // helper for clone: duplicate memory
    void clone_ptrs()
    {
      if (capacity == 0)
        return;

      std::shared_ptr<Dtype> old = cpu_ptr;
      buffer_data old_info = *gpu_info;
      alloc_and_set_cpu(capacity);
      memcpy(cpu_ptr.get(), old.get(), sizeof(Dtype) * capacity);
      gpu_info->sent_to_gpu = old_info.sent_to_gpu;
    }

  public:
    inline const size_t *dimensions() const { return cpu_grid.dimensions(); }
    inline size_t dimension(size_t i) const { return cpu_grid.dimension(i); }

    inline const size_t *offsets() const { return cpu_grid.offsets(); }
    inline size_t offset(size_t i) const { return cpu_grid.offset(i); }

    inline size_t size() const { return cpu_grid.size(); }

    inline void fill_zero()
    {
      memset(cpu_ptr.get(), 0, capacity * sizeof(Dtype));
    }

    template <typename... I>
    inline Dtype &operator()(I... indices)
    {
      return cpu_grid(indices...);
    }

    template <typename... I>
    inline Dtype operator()(I... indices) const
    {
      return cpu_grid(indices...);
    }

    /** \brief Copy data into dest. */
    size_t copyTo(cpu_grid_t &dest) const
    {
      size_t sz = std::min(size(), dest.size());
      if (sz == 0) return 0;
      memcpy(dest.data(), cpu_grid.data(), sz * sizeof(Dtype));
      return sz;
    }

    size_t copyTo(gpu_grid_t &dest) const
    {
      size_t sz = std::min(size(), dest.size());
      if (sz == 0) return 0;
      // Unified memory: same buffer, just memcpy the pointer region if different.
      if (dest.data() != cpu_grid.data())
        memcpy(dest.data(), cpu_grid.data(), sz * sizeof(Dtype));
      return sz;
    }

    size_t copyTo(ManagedGridBase<Dtype, NumDims> &dest) const
    {
      size_t sz = std::min(size(), dest.size());
      if (sz == 0) return 0;
      if (dest.cpu_ptr.get() != cpu_ptr.get())
        memcpy(dest.cpu_ptr.get(), cpu_ptr.get(), sz * sizeof(Dtype));
      return sz;
    }

    /** \brief Copy data from src. */
    size_t copyFrom(const cpu_grid_t &src)
    {
      size_t sz = std::min(size(), src.size());
      if (sz == 0) return 0;
      memcpy(cpu_grid.data(), src.data(), sz * sizeof(Dtype));
      return sz;
    }

    size_t copyFrom(const gpu_grid_t &src)
    {
      size_t sz = std::min(size(), src.size());
      if (sz == 0) return 0;
      if (src.data() != cpu_grid.data())
        memcpy(cpu_grid.data(), src.data(), sz * sizeof(Dtype));
      return sz;
    }

    size_t copyFrom(const ManagedGridBase<Dtype, NumDims> &src)
    {
      size_t sz = std::min(size(), src.size());
      if (sz == 0) return 0;
      if (src.cpu_ptr.get() != cpu_ptr.get())
        memcpy(cpu_ptr.get(), src.cpu_ptr.get(), sz * sizeof(Dtype));
      return sz;
    }

    size_t copyInto(size_t start, const ManagedGridBase<Dtype, NumDims> &src)
    {
      size_t off = offset(0) * start;
      size_t sz = size() - off;
      sz = std::min(sz, src.size());
      if (sz == 0) return 0;
      memcpy(cpu_ptr.get() + off, src.cpu_ptr.get(), sz * sizeof(Dtype));
      return sz;
    }

    template <typename... I, typename = typename std::enable_if<sizeof...(I) == NumDims>::type>
    ManagedGrid<Dtype, NumDims> resized(I... sizes)
    {
      cpu_grid_t g(nullptr, sizes...);
      if (g.size() <= capacity)
      {
        ManagedGrid<Dtype, NumDims> tmp;
        tmp.cpu_ptr = cpu_ptr;
        tmp.gpu_info = gpu_info;
        tmp.cpu_grid = cpu_grid_t(cpu_ptr.get(), sizes...);
        tmp.gpu_grid = gpu_grid_t(cpu_ptr.get(), sizes...);
        tmp.capacity = capacity;
        return tmp;
      }
      else
      {
        ManagedGrid<Dtype, NumDims> tmp(sizes...);
        if (size() > 0 && tmp.size() > 0)
          copyTo(tmp);
        return tmp;
      }
    }

    // ---- GPU / CPU view accessors ----
    // On Apple Silicon unified memory: both views point at the same buffer.

    const gpu_grid_t &gpu() const { togpu(); return gpu_grid; }
    gpu_grid_t &gpu()             { togpu(); return gpu_grid; }

    const cpu_grid_t &cpu() const { tocpu(); return cpu_grid; }
    cpu_grid_t &cpu()             { tocpu(); return cpu_grid; }

    /** \brief Mark memory as "on GPU". No data copy on unified memory. */
    void togpu(bool /*dotransfer*/ = true) const
    {
      if (capacity == 0) return;
      // Ensure both views point at the shared buffer.
      if (gpu_grid.data() == nullptr)
        gpu_grid.set_buffer(cpu_ptr.get());
      if (gpu_info)
        gpu_info->sent_to_gpu = true;
    }

    /** \brief Mark memory as "on CPU". No data copy on unified memory. */
    void tocpu(bool /*dotransfer*/ = true) const
    {
      if (gpu_info)
        gpu_info->sent_to_gpu = false;
    }

    bool ongpu() const { return gpu_info && gpu_info->sent_to_gpu; }
    bool oncpu() const { return gpu_info == nullptr || !gpu_info->sent_to_gpu; }

    operator cpu_grid_t() const { return cpu(); }
    operator cpu_grid_t &()     { return cpu(); }
    operator gpu_grid_t() const { return gpu(); }
    operator gpu_grid_t &()     { return gpu(); }

    inline const Dtype *data() const { return cpu_ptr.get(); }
    inline Dtype *data()             { return cpu_ptr.get(); }

    bool operator==(const ManagedGridBase<Dtype, NumDims> &rhs) const
    {
      return cpu_ptr == rhs.cpu_ptr;
    }

  protected:
    friend ManagedGridBase<Dtype, NumDims - 1>;
    explicit ManagedGridBase(const ManagedGridBase<Dtype, NumDims + 1> &G, size_t i)
        : gpu_grid(G.gpu_grid, i), cpu_grid(G.cpu_grid, i),
          cpu_ptr(G.cpu_ptr), capacity(G.capacity), gpu_info(G.gpu_info) {}
  };

  /** \brief A dense grid whose memory is managed by the class. */
  template <typename Dtype, std::size_t NumDims>
  class ManagedGrid : public ManagedGridBase<Dtype, NumDims>
  {
  protected:
    ManagedGrid(size_t *sizes) : ManagedGridBase<Dtype, NumDims>(sizes) {}

  public:
    using subgrid_t = ManagedGrid<Dtype, NumDims - 1>;
    using base_t = ManagedGridBase<Dtype, NumDims>;

    ManagedGrid() = default;

    template <typename... I, typename = typename std::enable_if<sizeof...(I) == NumDims>::type>
    ManagedGrid(I... sizes) : ManagedGridBase<Dtype, NumDims>(sizes...) {}

    subgrid_t operator[](size_t i) const
    {
      if (i >= this->cpu_grid.dimension(0))
        throw std::out_of_range("Index " + boost::lexical_cast<std::string>(i) + " out of bounds of dimension " + boost::lexical_cast<std::string>(this->cpu_grid.dimension(0)));
      return ManagedGrid<Dtype, NumDims - 1>(*static_cast<const ManagedGridBase<Dtype, NumDims> *>(this), i);
    }

    ManagedGrid<Dtype, NumDims> clone() const
    {
      ManagedGrid<Dtype, NumDims> ret(*this);
      ret.clone_ptrs();
      return ret;
    }

    template <class Archive>
    void save(Archive &ar, const unsigned int version) const
    {
      for (size_t i = 0; i < NumDims; i++)
        ar << this->dimension(i);
      ar << boost::serialization::make_array(this->data(), this->size());
    }

    template <class Archive>
    void load(Archive &ar, const unsigned int version)
    {
      size_t newdims[NumDims] = {0,};
      for (size_t i = 0; i < NumDims; i++)
        ar >> newdims[i];
      ManagedGrid<Dtype, NumDims> tmp(newdims);
      ar >> boost::serialization::make_array(tmp.data(), tmp.size());
      *this = tmp;
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()

  protected:
    friend ManagedGrid<Dtype, NumDims + 1>;
    explicit ManagedGrid<Dtype, NumDims>(const ManagedGridBase<Dtype, NumDims + 1> &G, size_t i)
        : ManagedGridBase<Dtype, NumDims>(G, i) {}
  };

  // Specialization for 1-D grids.
  template <typename Dtype>
  class ManagedGrid<Dtype, 1> : public ManagedGridBase<Dtype, 1>
  {
  public:
    using subgrid_t = Dtype;
    using base_t = ManagedGridBase<Dtype, 1>;

    ManagedGrid() = default;
    ManagedGrid(size_t sz) : ManagedGridBase<Dtype, 1>(sz) {}

    inline Dtype &operator[](size_t i)       { return this->cpu_grid[i]; }
    inline Dtype  operator[](size_t i) const { return this->cpu_grid[i]; }

    inline Dtype &operator()(size_t a)       { return this->cpu_grid(a); }
    inline Dtype  operator()(size_t a) const { return this->cpu_grid(a); }

    ManagedGrid<Dtype, 1> clone() const
    {
      ManagedGrid<Dtype, 1> ret(*this);
      ret.clone_ptrs();
      return ret;
    }

    template <class Archive>
    void save(Archive &ar, const unsigned int version) const
    {
      ar << this->dimension(0);
      ar << boost::serialization::make_array(this->data(), this->size());
    }

    template <class Archive>
    void load(Archive &ar, const unsigned int version)
    {
      size_t newdim = 0;
      ar >> newdim;
      ManagedGrid<Dtype, 1> tmp(newdim);
      ar >> boost::serialization::make_array(tmp.data(), tmp.size());
      *this = tmp;
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()

  protected:
    friend ManagedGrid<Dtype, 2>;
    explicit ManagedGrid<Dtype, 1>(const ManagedGridBase<Dtype, 2> &G, size_t i)
        : ManagedGridBase<Dtype, 1>(G, i) {}
  };

#define EXPAND_MGRID_DEFINITIONS(Z, SIZE, _)       \
  typedef ManagedGrid<float, SIZE> MGrid##SIZE##f; \
  typedef ManagedGrid<double, SIZE> MGrid##SIZE##d;

  BOOST_PP_REPEAT_FROM_TO(1, LIBMOLGRID_MAX_GRID_DIM, EXPAND_MGRID_DEFINITIONS, 0);

} // namespace libmolgrid
#endif /* MANAGED_GRID_H_ */
