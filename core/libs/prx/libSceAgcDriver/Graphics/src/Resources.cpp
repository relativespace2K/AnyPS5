#include "prx/libSceAgcDriver/Graphics/include/MemoryAccounting.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BufferPool.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DeferredDestruction.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestWorkMark.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/RunReport.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <exception>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <vector>
#ifdef __APPLE__
#include <cstring>
#endif

namespace AgcDriver::Graphics {
void WriteCheckpointSentinel(const Context& context, VkCommandBuffer commands);

namespace {

#ifdef __APPLE__
void ClearPooledTail(void* mapping, std::size_t size, VkDeviceSize mappedBytes, VkMemoryPropertyFlags properties) {
    if (mapping == nullptr || (properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0 || mappedBytes <= size) return;
    std::memset(static_cast<std::byte*>(mapping) + size, 0, static_cast<std::size_t>(mappedBytes - size));
}

#endif
VkDeviceSize ClassAllocationBytes(const VkMemoryRequirements& requirements, std::size_t capacity) {
    const auto alignment = std::max<VkDeviceSize>(requirements.alignment, 1);
    return std::max<VkDeviceSize>(requirements.size, (static_cast<VkDeviceSize>(capacity) + alignment - 1) / alignment * alignment);
}

VkBuffer BindExactBuffer(const Context& context, std::size_t size, VkBufferUsageFlags usage, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize availableBytes) {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    Check(context.Resolved(&DeviceFunctions::createBuffer, "vkCreateBuffer")(context.device, &info, nullptr, &buffer), "vkCreateBuffer pooled");
    try {
        VkMemoryRequirements requirements{};
        context.Resolved(&DeviceFunctions::getBufferMemoryRequirements, "vkGetBufferMemoryRequirements")(context.device, buffer, &requirements);
        Require(requirements.size <= availableBytes, "pooled buffer memory is smaller than the requested buffer");
        Require(requirements.alignment == 0 || offset % requirements.alignment == 0, "pooled buffer offset breaks the requested buffer's alignment");
        Check(context.Resolved(&DeviceFunctions::bindBufferMemory, "vkBindBufferMemory")(context.device, buffer, memory, offset), "vkBindBufferMemory pooled");
    } catch (...) {
        context.Resolved(&DeviceFunctions::destroyBuffer, "vkDestroyBuffer")(context.device, buffer, nullptr);
        throw;
    }
    return buffer;
}

bool SparseUploadOnSettleQueue(const Context& context) {
    return context.settleQueue != VK_NULL_HANDLE && context.settleQueueMutex != nullptr && context.settleFamily == context.queueFamily;
}

void UploadSparsePiece(const Context& context, VkBuffer destination, VkDeviceSize destinationOffset, std::uint64_t source, VkDeviceSize bytes, VkQueue queue, std::uint32_t queueFamily, std::mutex* queueMutex) {
    constexpr VkDeviceSize StagingBytes = VkDeviceSize{16} << 20u;
    const auto stagingBytes = static_cast<std::size_t>(std::min(bytes, StagingBytes));
    auto staging = std::make_shared<Buffer>(context, stagingBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    const auto stagingSpan = staging->Bytes();
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = queueFamily;
    VkCommandPool pool = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkCreateCommandPool>("vkCreateCommandPool")(context.device, &poolInfo, nullptr, &pool), "vkCreateCommandPool sparse upload");
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers")(context.device, &allocate, &commands), "vkAllocateCommandBuffers sparse upload");
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkCreateFence>("vkCreateFence")(context.device, &fenceInfo, nullptr, &fence), "vkCreateFence sparse upload");
    for (VkDeviceSize offset = 0; offset < bytes; offset += StagingBytes) {
        const auto chunk = static_cast<std::size_t>(std::min(bytes - offset, StagingBytes));
        GuestMemory::Read(source + offset, stagingSpan.first(chunk), 1);
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        Check(context.Function<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer")(commands, &begin), "vkBeginCommandBuffer sparse upload");
        const VkBufferCopy region{0, destinationOffset + offset, chunk};
        context.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(commands, staging->Handle(), destination, 1, &region);
        Check(context.Function<PFN_vkEndCommandBuffer>("vkEndCommandBuffer")(commands), "vkEndCommandBuffer sparse upload");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commands;
        {
            std::unique_lock<std::mutex> queueLock;
            if (queueMutex != nullptr) queueLock = std::unique_lock<std::mutex>(*queueMutex);
            const auto submitted = context.Function<PFN_vkQueueSubmit>("vkQueueSubmit")(queue, 1, &submit, fence);
            if (submitted != VK_SUCCESS) std::fprintf(stderr, "[gpu] sparse upload submit failed: dst=0x%llx+0x%llx family=%u result=%d\n", static_cast<unsigned long long>(destinationOffset + offset), static_cast<unsigned long long>(chunk), static_cast<unsigned>(queueFamily), static_cast<int>(submitted));
            Check(submitted, "vkQueueSubmit sparse upload");
        }
        Check(context.Function<PFN_vkWaitForFences>("vkWaitForFences")(context.device, 1, &fence, VK_TRUE, std::numeric_limits<std::uint64_t>::max()), "vkWaitForFences sparse upload");
        Check(context.Function<PFN_vkResetFences>("vkResetFences")(context.device, 1, &fence), "vkResetFences sparse upload");
        Check(context.Function<PFN_vkResetCommandBuffer>("vkResetCommandBuffer")(commands, 0), "vkResetCommandBuffer sparse upload");
    }
    context.Function<PFN_vkDestroyFence>("vkDestroyFence")(context.device, fence, nullptr);
    context.Function<PFN_vkDestroyCommandPool>("vkDestroyCommandPool")(context.device, pool, nullptr);
}

struct SparseInputEntry {
    VkDevice device = VK_NULL_HANDLE;
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    VkBufferUsageFlags usage = 0;
    std::shared_ptr<Buffer> buffer;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> bound;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> seen;
    std::uint64_t generation = 0;
};

std::mutex sparseInputsMutex;
std::list<SparseInputEntry> sparseInputs;
constexpr std::size_t SparseInputEntries = 32;

std::vector<std::pair<std::uint64_t, std::uint64_t>> SparseRanges(const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges, std::uint64_t begin, std::uint64_t end, VkDeviceSize alignment) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> out;
    for (const auto& [rawFirst, rawLast] : ranges) {
        const auto first = std::max<std::uint64_t>(rawFirst, begin);
        const auto last = std::min<std::uint64_t>(rawLast, end);
        if (first >= last) continue;
        const auto alignedFirst = (first + alignment - 1) & ~(alignment - 1);
        const auto alignedLast = last & ~(alignment - 1);
        if (alignedFirst < alignedLast) out.emplace_back(alignedFirst, alignedLast);
    }
    return out;
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> MergeSparseRanges(std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges) {
    std::sort(ranges.begin(), ranges.end());
    std::vector<std::pair<std::uint64_t, std::uint64_t>> merged;
    for (const auto& [first, last] : ranges) {
        if (!merged.empty() && first <= merged.back().second) merged.back().second = std::max(merged.back().second, last);
        else merged.emplace_back(first, last);
    }
    return merged;
}

bool SparseRangeCovered(const std::vector<std::pair<std::uint64_t, std::uint64_t>>& bound, std::uint64_t first, std::uint64_t last) {
    for (const auto& [boundFirst, boundLast] : bound) {
        if (first >= boundFirst && last <= boundLast) return true;
    }
    return false;
}

bool SparseRangesCovered(const std::vector<std::pair<std::uint64_t, std::uint64_t>>& bound, const std::vector<std::pair<std::uint64_t, std::uint64_t>>& wanted) {
    for (const auto& [first, last] : wanted) {
        if (!SparseRangeCovered(bound, first, last)) return false;
    }
    return true;
}

void UploadSparseRanges(const Context& context, VkBuffer destination, std::uint64_t base, const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges, VkQueue queue, std::uint32_t family, std::mutex* mutex) {
    for (const auto& [first, last] : ranges) UploadSparsePiece(context, destination, first - base, first, last - first, queue, family, mutex);
}

}

void ClearSparseInputs(VkDevice device) {
    std::lock_guard lock(sparseInputsMutex);
    sparseInputs.remove_if([device](const SparseInputEntry& entry) { return entry.device == device; });
}

Buffer::Buffer(const Context& context, std::size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties) : Buffer(context, size, usage, properties, false) {}

std::shared_ptr<Buffer> Buffer::WithinDeviceLocalHeadroom(const Context& context, std::size_t size, VkBufferUsageFlags usage) {
    std::shared_ptr<Buffer> buffer(new Buffer(context, size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true));
    return buffer->ready ? buffer : nullptr;
}

Buffer::Buffer(const Context& context, std::size_t size, VkBuffer buffer, std::vector<VkDeviceMemory> sparseMemories, std::vector<std::shared_ptr<void>> hostImportAccesses) : context(context), buffer(buffer), sparseMemories(std::move(sparseMemories)), hostImportAccesses(std::move(hostImportAccesses)), size(size), capacity(size) {
    usage = 0;
    properties = 0;
    ready = true;
}

std::shared_ptr<Buffer> Buffer::SparseGuestImport(const Context& context, const std::uint64_t address, const std::size_t bytes, const VkBufferUsageFlags usage, const VkDeviceSize bindOffset, const VkDeviceSize bindBytes, VkDeviceSize* offset) {
    if (offset == nullptr || bytes == 0 || !context.sparseBinding || !context.sparseResidencyBuffer || context.sparseQueue == VK_NULL_HANDLE) return {};
    const bool settleUpload = SparseUploadOnSettleQueue(context);
    if (!settleUpload && !GuestMemory::GpuMutex().HeldByThisThread()) return {};
    constexpr VkDeviceSize Block = 65536;
    constexpr VkDeviceSize Chunk = VkDeviceSize{1} << 29u;
    constexpr VkDeviceSize Region = VkDeviceSize{4} << 30u;
    const auto alignment = std::max<VkDeviceSize>(Block, context.hostImportAlignment);
    const auto requestedBegin = address & ~(alignment - 1);
    const auto requestedEnd = (address + bytes + alignment - 1) & ~(alignment - 1);
    if (requestedEnd <= requestedBegin) return {};
    const auto bindStart = std::min<VkDeviceSize>(bindOffset, bytes);
    const auto bindLength = std::min<VkDeviceSize>(bindBytes, bytes - bindStart);
    const auto bindBegin = std::max<std::uint64_t>(requestedBegin, (address + bindStart) & ~(alignment - 1));
    const auto bindEnd = std::min<std::uint64_t>(requestedEnd, (address + bindStart + bindLength + alignment - 1) & ~(alignment - 1));
    if (bindEnd <= bindBegin) return {};
    const auto uploadQueue = settleUpload ? context.settleQueue : context.queue;
    const auto uploadFamily = settleUpload ? context.settleFamily : context.queueFamily;
    auto* uploadMutex = settleUpload ? context.settleQueueMutex : nullptr;
    const auto cacheBase = address & ~(Region - 1);
    const auto cacheEnd = cacheBase + Region;
    const auto cacheBytes = Region;
    std::unique_lock<std::mutex> cacheLock(sparseInputsMutex);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> previousBound;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> previousSeen;
    for (auto entry = sparseInputs.begin(); entry != sparseInputs.end(); ++entry) {
        if (entry->device != context.device || entry->begin != cacheBase || entry->end != cacheEnd || entry->usage != usage) continue;
        if (entry->generation != 0 && SparseRangeCovered(entry->seen, bindBegin, bindEnd) && GuestMemory::UnchangedSince(bindBegin, bindEnd - bindBegin, entry->generation)) {
            sparseInputs.splice(sparseInputs.begin(), sparseInputs, entry);
            *offset = address - cacheBase;
            return entry->buffer;
        }
        const auto commit = SparseRanges(GuestMemory::DescribeCommitted(bindBegin, bindEnd - bindBegin).ranges, bindBegin, bindEnd, alignment);
        if (!SparseRangesCovered(entry->bound, commit)) {
            previousBound = std::move(entry->bound);
            previousSeen = std::move(entry->seen);
            sparseInputs.erase(entry);
            break;
        }
        UploadSparseRanges(context, entry->buffer->Handle(), entry->begin, commit, uploadQueue, uploadFamily, uploadMutex);
        entry->seen.emplace_back(bindBegin, bindEnd);
        entry->seen = MergeSparseRanges(std::move(entry->seen));
        entry->generation = GuestMemory::CollectWrites(bindBegin, bindEnd - bindBegin);
        sparseInputs.splice(sparseInputs.begin(), sparseInputs, entry);
        *offset = address - cacheBase;
        return entry->buffer;
    }
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = cacheBytes;
    info.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const std::array<std::uint32_t, 2> families{context.queueFamily, context.sparseFamily};
    info.sharingMode = context.sparseFamily != context.queueFamily ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
    info.queueFamilyIndexCount = context.sparseFamily != context.queueFamily ? static_cast<std::uint32_t>(families.size()) : 0;
    info.pQueueFamilyIndices = context.sparseFamily != context.queueFamily ? families.data() : nullptr;
    info.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT;
    VkBuffer buffer = VK_NULL_HANDLE;
    auto result = context.Function<PFN_vkCreateBuffer>("vkCreateBuffer")(context.device, &info, nullptr, &buffer);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "[gpu] sparse guest import of 0x%llx+0x%llx: vkCreateBuffer failed (%d)\n", static_cast<unsigned long long>(cacheBase), static_cast<unsigned long long>(cacheBytes), static_cast<int>(result));
        return {};
    }
    VkMemoryRequirements requirements{};
    context.Function<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(context.device, buffer, &requirements);
    std::vector<VkSparseMemoryBind> binds;
    std::vector<VkDeviceMemory> memories;
    const auto cleanup = [&] {
        for (const auto memory : memories) MemoryAccounting::Free(context, memory);
        memories.clear();
        context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, buffer, nullptr);
    };
    struct Upload {
        VkDeviceSize destination;
        std::uint64_t source;
        VkDeviceSize bytes;
    };
    std::vector<Upload> uploads;
    auto commit = SparseRanges(GuestMemory::DescribeCommitted(bindBegin, bindEnd - bindBegin).ranges, bindBegin, bindEnd, alignment);
    commit.insert(commit.end(), previousBound.begin(), previousBound.end());
    commit = MergeSparseRanges(std::move(commit));
    previousSeen.emplace_back(bindBegin, bindEnd);
    const auto seen = MergeSparseRanges(std::move(previousSeen));
    std::uint32_t deviceLocalType = 0;
    bool deviceLocalFound = false;
    for (std::uint32_t type = 0; type < context.memory.memoryTypeCount; ++type) {
        if ((requirements.memoryTypeBits & (1u << type)) != 0 && (context.memory.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
            deviceLocalType = type;
            deviceLocalFound = true;
            break;
        }
    }
    if (!deviceLocalFound) {
        cleanup();
        return {};
    }
    for (const auto& [first, last] : commit) {
        auto cursor = first;
        const auto limit = last;
        while (cursor < limit) {
            const auto piece = std::min<VkDeviceSize>(limit - cursor, Chunk) & ~(alignment - 1);
            if (piece == 0) break;
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize = piece;
            allocation.memoryTypeIndex = deviceLocalType;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            result = MemoryAccounting::Allocate(context, allocation, "sparse guest import", &memory);
            if (result != VK_SUCCESS) {
                std::fprintf(stderr, "[gpu] sparse guest import of 0x%llx+0x%llx: vkAllocateMemory at 0x%llx+0x%llx failed (%d)\n", static_cast<unsigned long long>(cacheBase), static_cast<unsigned long long>(cacheBytes), static_cast<unsigned long long>(cursor), static_cast<unsigned long long>(piece), static_cast<int>(result));
                cursor += piece;
                continue;
            }
            memories.push_back(memory);
            binds.push_back(VkSparseMemoryBind{cursor - cacheBase, piece, memory, 0, 0});
            uploads.push_back({cursor - cacheBase, cursor, piece});
            cursor += piece;
        }
    }
    if (!binds.empty()) {
        VkSparseBufferMemoryBindInfo bufferBind{buffer, static_cast<std::uint32_t>(binds.size()), binds.data()};
        VkBindSparseInfo sparse{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO};
        sparse.bufferBindCount = 1;
        sparse.pBufferBinds = &bufferBind;
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence = VK_NULL_HANDLE;
        result = context.Function<PFN_vkCreateFence>("vkCreateFence")(context.device, &fenceInfo, nullptr, &fence);
        if (result == VK_SUCCESS) {
            std::unique_lock<std::mutex> queueLock;
            if (context.sparseQueueMutex != nullptr) queueLock = std::unique_lock<std::mutex>(*context.sparseQueueMutex);
            result = context.Function<PFN_vkQueueBindSparse>("vkQueueBindSparse")(context.sparseQueue, 1, &sparse, fence);
            if (result == VK_SUCCESS) result = context.Function<PFN_vkWaitForFences>("vkWaitForFences")(context.device, 1, &fence, VK_TRUE, std::numeric_limits<std::uint64_t>::max());
            context.Function<PFN_vkDestroyFence>("vkDestroyFence")(context.device, fence, nullptr);
        }
        if (result != VK_SUCCESS) {
            std::fprintf(stderr, "[gpu] sparse guest import of 0x%llx+0x%llx: sparse bind failed (%d)\n", static_cast<unsigned long long>(cacheBase), static_cast<unsigned long long>(cacheBytes), static_cast<int>(result));
            cleanup();
            return {};
        }
    }
    VkDeviceSize uploadBytes = 0;
    for (const auto& upload : uploads) uploadBytes += upload.bytes;
    std::fprintf(stderr, "[gpu] sparse guest import 0x%llx+0x%llx: created with %zu bind(s), %zu upload(s) of 0x%llx bytes on queue family %u\n", static_cast<unsigned long long>(cacheBase), static_cast<unsigned long long>(cacheBytes), binds.size(), uploads.size(), static_cast<unsigned long long>(uploadBytes), static_cast<unsigned>(uploadFamily));
    for (const auto& upload : uploads) UploadSparsePiece(context, buffer, upload.destination, upload.source, upload.bytes, uploadQueue, uploadFamily, uploadMutex);
    std::shared_ptr<Buffer> created(new Buffer(context, static_cast<std::size_t>(cacheBytes), buffer, std::move(memories), {}));
    sparseInputs.push_front({context.device, cacheBase, cacheEnd, usage, created, commit, seen, GuestMemory::CollectWrites(bindBegin, bindEnd - bindBegin)});
    while (sparseInputs.size() > SparseInputEntries) sparseInputs.pop_back();
    *offset = address - cacheBase;
    return created;
}

Buffer::Buffer(const Context& context, std::size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, bool headroom) : context(context), size(size), capacity(BufferPool::Capacity(size)), usage(BufferPool::Usage(usage, properties)), properties(properties) {
    Require(size != 0, "zero-sized GPU buffer");
    const bool addressable = (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0;
    Require(!addressable || context.bufferDeviceAddress, "buffer device address is not enabled");
    cache = GetBufferPool(context);
    if (const auto allocation = cache->Take(size, this->usage, properties, "vkAllocateMemory buffer")) {
        buffer = allocation->buffer;
        memory = allocation->memory;
        mapping = allocation->mapping;
        deviceAddress = allocation->address;
        allocationBytes = allocation->allocationBytes;
        capacity = allocation->bytes;
        offset = allocation->offset;
        slab = allocation->slab;
#ifdef __APPLE__
        ClearPooledTail(mapping, size, std::min<VkDeviceSize>(capacity, allocationBytes), properties);
#endif
        if (allocation->bufferBytes != size) {
            context.Resolved(&DeviceFunctions::destroyBuffer, "vkDestroyBuffer")(context.device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
            deviceAddress = 0;
            try {
                buffer = BindExactBuffer(context, size, this->usage, memory, offset, allocationBytes);
                initializeAddress(usage);
            } catch (...) {
                release();
                throw;
            }
        }
        ready = true;
        return;
    }
    try {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = this->usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        Check(context.Resolved(&DeviceFunctions::createBuffer, "vkCreateBuffer")(context.device, &info, nullptr, &buffer), "vkCreateBuffer");
        VkMemoryRequirements requirements{};
        context.Resolved(&DeviceFunctions::getBufferMemoryRequirements, "vkGetBufferMemoryRequirements")(context.device, buffer, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        const VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, nullptr, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
        if (addressable) allocation.pNext = &flags;
        allocation.allocationSize = ClassAllocationBytes(requirements, capacity);
        allocationBytes = allocation.allocationSize;
        // The CPU reads most of these buffers back (write-back, diffs), which is very slow from
        // write-combined memory, so the default host properties prefer cached host memory.
        const bool useSlab = (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 && BufferPool::SlabEligible(capacity, requirements.alignment, requirements.size, context.limits.nonCoherentAtomSize);
        allocation.memoryTypeIndex = context.BufferMemoryType(requirements.memoryTypeBits, properties, useSlab ? BufferPool::SlabBlockBytes(capacity) : allocation.allocationSize);
        if (headroom && !context.DeviceLocalHeadroom(allocation.memoryTypeIndex, useSlab ? BufferPool::SlabBlockBytes(capacity) : allocation.allocationSize)) {
            context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
            return;
        }
        if (useSlab) {
            const auto slot = cache->TakeSlot(context, allocation.memoryTypeIndex, capacity, addressable);
            memory = slot.memory;
            offset = slot.offset;
            slab = true;
            allocationBytes = capacity;
            Check(context.Resolved(&DeviceFunctions::bindBufferMemory, "vkBindBufferMemory")(context.device, buffer, memory, offset), "vkBindBufferMemory slab");
            initializeAddress(usage);
            mapping = slot.mapping;
#ifdef __APPLE__
            ClearPooledTail(mapping, size, capacity, properties);
#endif
            ready = true;
            return;
        }
        memory = cache->AllocateMemory(allocation, "vkAllocateMemory buffer");
        Check(context.Resolved(&DeviceFunctions::bindBufferMemory, "vkBindBufferMemory")(context.device, buffer, memory, 0), "vkBindBufferMemory");
        initializeAddress(usage);
        if ((properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) Check(context.Resolved(&DeviceFunctions::mapMemory, "vkMapMemory")(context.device, memory, 0, VK_WHOLE_SIZE, 0, &mapping), "vkMapMemory");
        ready = true;
    } catch (...) {
        release();
        throw;
    }
}

Buffer::~Buffer() {
    release();
}

void Buffer::release() noexcept {
    if (!sparseMemories.empty()) {
        for (const auto memory : sparseMemories) MemoryAccounting::Free(context, memory);
        sparseMemories.clear();
        hostImportAccesses.clear();
        if (buffer != VK_NULL_HANDLE) context.Resolved(&DeviceFunctions::destroyBuffer, "vkDestroyBuffer")(context.device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        ready = false;
        return;
    }
    if (ready && cache) {
        cache->Put({buffer, memory, mapping, deviceAddress, allocationBytes, capacity, usage, properties, offset, slab, size});
        return;
    }
    if (slab) {
        if (buffer) context.Resolved(&DeviceFunctions::destroyBuffer, "vkDestroyBuffer")(context.device, buffer, nullptr);
        cache->PutSlot(memory, offset);
        return;
    }
    if (mapping) context.Resolved(&DeviceFunctions::unmapMemory, "vkUnmapMemory")(context.device, memory);
    if (buffer) context.Resolved(&DeviceFunctions::destroyBuffer, "vkDestroyBuffer")(context.device, buffer, nullptr);
    if (memory) MemoryAccounting::Free(context, memory);
}

VkBuffer Buffer::Handle() const {
    return buffer;
}

std::span<std::byte> Buffer::Bytes() {
    Require(mapping != nullptr, "device-local buffer has no host mapping");
    return {static_cast<std::byte*>(mapping), size};
}

void Buffer::Invalidate() {
    Require(mapping != nullptr, "cannot invalidate an unmapped GPU buffer");
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = memory;
    range.offset = offset;
    range.size = slab ? static_cast<VkDeviceSize>(capacity) : VK_WHOLE_SIZE;
    Check(context.Function<PFN_vkInvalidateMappedMemoryRanges>("vkInvalidateMappedMemoryRanges")(context.device, 1, &range), "vkInvalidateMappedMemoryRanges");
}

DeviceBuffer::DeviceBuffer(const Context& context, std::size_t size, VkBufferUsageFlags usage) : context(context), size(size), capacity(BufferPool::Capacity(size)), usage(BufferPool::Usage(usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
    Require(size != 0, "zero-sized device buffer");
    cache = GetBufferPool(context);
    auto pooled = cache->Take(size, this->usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "vkAllocateMemory device buffer");
    if (!pooled) {
        pooled = cache->Take(size, this->usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "vkAllocateMemory device buffer", true);
        if (pooled && context.DeviceLocalFits(~0u, capacity)) {
            cache->Put(*pooled);
            pooled.reset();
        }
    }
    if (pooled) {
        hostPlaced = pooled->hostPlaced;
        buffer = pooled->buffer;
        memory = pooled->memory;
        allocationBytes = pooled->allocationBytes;
        capacity = pooled->bytes;
        if (pooled->bufferBytes == size) {
            ready = true;
            return;
        }
        context.Resolved(&DeviceFunctions::destroyBuffer, "vkDestroyBuffer")(context.device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        try {
            buffer = BindExactBuffer(context, size, this->usage, memory, 0, allocationBytes);
        } catch (...) {
            release();
            throw;
        }
        ready = true;
        return;
    }
    try {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = this->usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        Check(context.Resolved(&DeviceFunctions::createBuffer, "vkCreateBuffer")(context.device, &info, nullptr, &buffer), "vkCreateBuffer device");
        VkMemoryRequirements requirements{};
        context.Resolved(&DeviceFunctions::getBufferMemoryRequirements, "vkGetBufferMemoryRequirements")(context.device, buffer, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = ClassAllocationBytes(requirements, capacity);
        allocationBytes = allocation.allocationSize;
        allocation.memoryTypeIndex = context.DeviceMemoryType(requirements.memoryTypeBits, allocation.allocationSize, "device buffer");
        const auto deviceType = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        std::uint32_t allocatedType = allocation.memoryTypeIndex;
        memory = cache->AllocateMemory(allocation, "vkAllocateMemory device buffer", context.RefusedDeviceMemoryFallback(requirements.memoryTypeBits, allocation.memoryTypeIndex), &allocatedType);
        hostPlaced = allocatedType != deviceType;
        Check(context.Resolved(&DeviceFunctions::bindBufferMemory, "vkBindBufferMemory")(context.device, buffer, memory, 0), "vkBindBufferMemory device");
        ready = true;
    } catch (...) {
        release();
        throw;
    }
}

DeviceBuffer::~DeviceBuffer() {
    release();
}

void DeviceBuffer::release() noexcept {
    if (ready && buffer && memory && cache) {
        BufferAllocation allocation{buffer, memory, nullptr, 0, allocationBytes, capacity, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, false, size};
        allocation.hostPlaced = hostPlaced;
        cache->Put(allocation);
        return;
    }
    if (buffer) context.Resolved(&DeviceFunctions::destroyBuffer, "vkDestroyBuffer")(context.device, buffer, nullptr);
    if (memory) MemoryAccounting::Free(context, memory);
}

VkBuffer DeviceBuffer::Handle() const {
    return buffer;
}

std::size_t DeviceBuffer::Size() const {
    return size;
}

void CopyBuffer(const Context& context, VkCommandBuffer commands, VkBuffer source, VkDeviceSize sourceOffset, VkBuffer destination, VkDeviceSize destinationOffset, VkDeviceSize bytes) {
    PerformanceTrace::Amount("Graphics.Copy", 1);
    PerformanceTrace::Amount("Graphics.Copy.bytes", bytes);
    const VkBufferCopy region{sourceOffset, destinationOffset, bytes};
    context.Resolved(&DeviceFunctions::cmdCopyBuffer, "vkCmdCopyBuffer")(commands, source, destination, 1, &region);
}

void RecordMemoryBarrier(const Context& context, VkCommandBuffer commands, VkPipelineStageFlags sourceStage, VkPipelineStageFlags destinationStage, VkAccessFlags sourceAccess, VkAccessFlags destinationAccess) {
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, sourceAccess, destinationAccess};
    context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier")(commands, sourceStage, destinationStage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void FillDeviceFunctions(const Context& context, DeviceFunctions& functions) {
    functions.cmdPipelineBarrier = context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
    functions.cmdCopyBuffer = context.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer");
    functions.cmdUpdateBuffer = context.Function<PFN_vkCmdUpdateBuffer>("vkCmdUpdateBuffer");
    functions.cmdFillBuffer = context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer");
    functions.cmdBindPipeline = context.lookup<PFN_vkCmdBindPipeline>("vkCmdBindPipeline");
    functions.cmdBindDescriptorSets = context.Function<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets");
    functions.cmdPushConstants = context.Function<PFN_vkCmdPushConstants>("vkCmdPushConstants");
    functions.cmdDispatch = context.Function<PFN_vkCmdDispatch>("vkCmdDispatch");
    functions.cmdDispatchIndirect = context.Function<PFN_vkCmdDispatchIndirect>("vkCmdDispatchIndirect");
    functions.cmdBeginRenderPass = context.Function<PFN_vkCmdBeginRenderPass>("vkCmdBeginRenderPass");
    functions.cmdEndRenderPass = context.Function<PFN_vkCmdEndRenderPass>("vkCmdEndRenderPass");
    functions.cmdSetViewport = context.lookup<PFN_vkCmdSetViewport>("vkCmdSetViewport");
    functions.cmdSetScissor = context.lookup<PFN_vkCmdSetScissor>("vkCmdSetScissor");
    if (context.dynamicColorWriteMask) functions.cmdSetColorWriteMask = context.lookup<PFN_vkCmdSetColorWriteMaskEXT>("vkCmdSetColorWriteMaskEXT");
    functions.cmdSetDepthBounds = context.lookup<PFN_vkCmdSetDepthBounds>("vkCmdSetDepthBounds");
    functions.cmdSetDepthBias = context.lookup<PFN_vkCmdSetDepthBias>("vkCmdSetDepthBias");
    if (context.graphicsPipelineLibrary) {
        functions.cmdBeginRendering = context.Function<PFN_vkCmdBeginRenderingKHR>("vkCmdBeginRenderingKHR");
        functions.cmdEndRendering = context.Function<PFN_vkCmdEndRenderingKHR>("vkCmdEndRenderingKHR");
        functions.cmdSetCullMode = context.lookup<PFN_vkCmdSetCullModeEXT>("vkCmdSetCullModeEXT");
        functions.cmdSetFrontFace = context.lookup<PFN_vkCmdSetFrontFaceEXT>("vkCmdSetFrontFaceEXT");
        functions.cmdSetDepthTestEnable = context.lookup<PFN_vkCmdSetDepthTestEnableEXT>("vkCmdSetDepthTestEnableEXT");
        functions.cmdSetDepthWriteEnable = context.lookup<PFN_vkCmdSetDepthWriteEnableEXT>("vkCmdSetDepthWriteEnableEXT");
        functions.cmdSetDepthCompareOp = context.lookup<PFN_vkCmdSetDepthCompareOpEXT>("vkCmdSetDepthCompareOpEXT");
        functions.cmdSetDepthBoundsTestEnable = context.lookup<PFN_vkCmdSetDepthBoundsTestEnableEXT>("vkCmdSetDepthBoundsTestEnableEXT");
        functions.cmdSetStencilTestEnable = context.lookup<PFN_vkCmdSetStencilTestEnableEXT>("vkCmdSetStencilTestEnableEXT");
        functions.cmdSetStencilOp = context.lookup<PFN_vkCmdSetStencilOpEXT>("vkCmdSetStencilOpEXT");
        functions.cmdSetStencilCompareMask = context.lookup<PFN_vkCmdSetStencilCompareMask>("vkCmdSetStencilCompareMask");
        functions.cmdSetStencilWriteMask = context.lookup<PFN_vkCmdSetStencilWriteMask>("vkCmdSetStencilWriteMask");
        functions.cmdSetStencilReference = context.lookup<PFN_vkCmdSetStencilReference>("vkCmdSetStencilReference");
    }
    functions.cmdBindVertexBuffers = context.Function<PFN_vkCmdBindVertexBuffers>("vkCmdBindVertexBuffers");
    functions.cmdBindIndexBuffer = context.Function<PFN_vkCmdBindIndexBuffer>("vkCmdBindIndexBuffer");
    functions.cmdDraw = context.Function<PFN_vkCmdDraw>("vkCmdDraw");
    functions.cmdDrawIndexed = context.Function<PFN_vkCmdDrawIndexed>("vkCmdDrawIndexed");
    functions.cmdDrawIndirect = context.Function<PFN_vkCmdDrawIndirect>("vkCmdDrawIndirect");
    functions.cmdDrawIndexedIndirect = context.Function<PFN_vkCmdDrawIndexedIndirect>("vkCmdDrawIndexedIndirect");
    functions.cmdCopyBufferToImage = context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage");
    functions.cmdCopyImageToBuffer = context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer");
    functions.cmdClearColorImage = context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage");
    functions.updateDescriptorSets = context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets");
    functions.allocateDescriptorSets = context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets");
    functions.getFenceStatus = context.Function<PFN_vkGetFenceStatus>("vkGetFenceStatus");
    functions.createBuffer = context.Function<PFN_vkCreateBuffer>("vkCreateBuffer");
    functions.destroyBuffer = context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer");
    functions.getBufferMemoryRequirements = context.Function<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements");
    functions.bindBufferMemory = context.Function<PFN_vkBindBufferMemory>("vkBindBufferMemory");
    functions.allocateMemory = context.Function<PFN_vkAllocateMemory>("vkAllocateMemory");
    functions.freeMemory = context.Function<PFN_vkFreeMemory>("vkFreeMemory");
    functions.mapMemory = context.Function<PFN_vkMapMemory>("vkMapMemory");
    functions.unmapMemory = context.Function<PFN_vkUnmapMemory>("vkUnmapMemory");
}

RenderTarget::RenderTarget(const Context& context, const ColorTarget& target, bool blending) : context(context) {
    VkFormatProperties properties{};
    context.formatProperties(context.physical, target.format, &properties);
    const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT | (blending ? VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT : 0u);
    Require((properties.optimalTilingFeatures & required) == required, "render-target format does not support required operations");
    constexpr auto usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkImageFormatProperties supported{};
    Check(context.imageFormatProperties(context.physical, target.format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, usage, 0, &supported), "vkGetPhysicalDeviceImageFormatProperties");
    Require(target.extent.width <= supported.maxExtent.width && target.extent.height <= supported.maxExtent.height && (supported.sampleCounts & VK_SAMPLE_COUNT_1_BIT) != 0 && target.bytes <= supported.maxResourceSize, "render target exceeds device image limits");
    Require(target.extent.width <= context.limits.maxFramebufferWidth && target.extent.height <= context.limits.maxFramebufferHeight, "render target exceeds framebuffer limits");
    try {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = target.format;
        info.extent = {target.extent.width, target.extent.height, 1};
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        Check(context.Function<PFN_vkCreateImage>("vkCreateImage")(context.device, &info, nullptr, &image), "vkCreateImage");
        VkMemoryRequirements requirements{};
        context.Function<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(context.device, image, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        memory = AllocateImageMemory(context, allocation, requirements.memoryTypeBits, "vkAllocateMemory render target", &memoryType);
        memoryBytes = allocation.allocationSize;
        Check(context.Function<PFN_vkBindImageMemory>("vkBindImageMemory")(context.device, image, memory, 0), "vkBindImageMemory");
        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = target.format;
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        Check(context.Function<PFN_vkCreateImageView>("vkCreateImageView")(context.device, &viewInfo, nullptr, &view), "vkCreateImageView");
    } catch (...) {
        release();
        throw;
    }
}

RenderTarget::~RenderTarget() {
    release();
}

void RenderTarget::release() noexcept {
    if (view) context.Function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, view, nullptr);
    if (image) context.Function<PFN_vkDestroyImage>("vkDestroyImage")(context.device, image, nullptr);
    ReleaseImageMemory(context, memory, memoryBytes, memoryType);
}

VkImage RenderTarget::Image() const {
    return image;
}

VkImageView RenderTarget::View() const {
    return view;
}

void NoteGuestWorkOnCommandBuffer(VkCommandBuffer commands) {
    GuestWorkMark::Note(commands);
}

CommandBatch::CommandBatch(const Context& context) : context(context) {
    // Records into the device's one command pool and submits to its queue: device-lock work only.
    GuestMemory::AssertGpuLockHeld("CommandBatch");
    try {
        VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = context.pool;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        Check(context.Function<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers")(context.device, &allocation, &commands), "vkAllocateCommandBuffers");
        VkFenceCreateInfo info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence created = VK_NULL_HANDLE;
        Check(context.Function<PFN_vkCreateFence>("vkCreateFence")(context.device, &info, nullptr, &created), "vkCreateFence");
        fence = created;
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        Check(context.Function<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer")(commands, &begin), "vkBeginCommandBuffer");
        WriteCheckpointSentinel(context, commands);
        // Work recorded so far goes first in queue order, so this batch sees its results.
        if (auto* recorder = Recorder::Active(); recorder != nullptr && recorder->Recording()) recorder->Submit();
    } catch (...) {
        release();
        throw;
    }
}

CommandBatch::~CommandBatch() {
    release();
}

void CommandBatch::release() noexcept {
    GuestWorkMark::Forget(commands);
    if (pending) {
        auto result = context.Resolved(&DeviceFunctions::getFenceStatus, "vkGetFenceStatus")(context.device, fence);
        if (result == VK_NOT_READY) result = context.Function<PFN_vkQueueWaitIdle>("vkQueueWaitIdle")(context.queue);
        if (result != VK_SUCCESS && result != VK_ERROR_DEVICE_LOST) std::terminate();
    }
    if (commands) context.Function<PFN_vkFreeCommandBuffers>("vkFreeCommandBuffers")(context.device, context.pool, 1, &commands);
    if (fence) context.Function<PFN_vkDestroyFence>("vkDestroyFence")(context.device, fence, nullptr);
}

VkCommandBuffer CommandBatch::Handle() const {
    return commands;
}

void CommandBatch::SubmitAndWait() {
    Submit();
    Wait();
}

void CommandBatch::Submit() {
    PerformanceTimer timing("Graphics.Submit");
    Require(!submitted, "command batch has already been submitted");
    Check(context.Function<PFN_vkEndCommandBuffer>("vkEndCommandBuffer")(commands), "vkEndCommandBuffer");
    VkSubmitInfo submission{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &commands;
    timing.Mark("command_end");
    DestroyDeferredAtIdle(context);
    Check(context.Function<PFN_vkQueueSubmit>("vkQueueSubmit")(context.queue, 1, &submission, fence), "vkQueueSubmit graphics");
    pending = true;
    submitted = true;
    timing.Mark("queue_submit");
    if (GuestWorkMark::Take(commands)) LibcRunReportMilestone_nid_no_patch(2);
}

void CommandBatch::Wait() {
    Require(submitted, "command batch has not been submitted");
    if (!pending) return;
    PerformanceTimer timing("Graphics.Wait");
    const auto result = context.Function<PFN_vkWaitForFences>("vkWaitForFences")(context.device, 1, &fence, VK_TRUE, 5'000'000'000ULL);
    timing.Mark("fence_wait");
    if (result == VK_SUCCESS || result == VK_ERROR_DEVICE_LOST) pending = false;
    Check(result, "vkWaitForFences graphics");
    // Recorded batches preceded this one, so their completions (write-backs) can run now.
    if (auto* recorder = Recorder::Active()) recorder->Reap();
}

}
