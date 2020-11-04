#include "NvmEngine.hpp"
#include <cstdlib>
#include <sstream>
#include <sys/time.h>
#include <cstdio>
#include <libpmem.h>
#include <emmintrin.h>
#include <nmmintrin.h>
#include <signal.h>
#include <execinfo.h>
#include <mutex>

#include <new>
#include <iostream>
#include <string>
#include <atomic>
#include <cassert>
#include <thread>

#include "CityHash.hpp"

#define LOCK_ACQ(lock) while (__atomic_add_fetch(lock, 1, __ATOMIC_ACQUIRE)-1) {_mm_pause();} 
#define LOCK_RLS(lock) __atomic_store_n(lock, 0, __ATOMIC_RELEASE)
#define FIRST_MATCH(m) (__builtin_ffs(m) >> 1)
#define BUCKET_INDEX(h) ((h >> 15) % (kNumOfBuckets - 1))
#define DIGEST(h) ((h & 0x7FFF) + 1)
#define MIN_BLOCK_SIZE(psize) (((sizeof(bctrl_t) + psize) / kBlockSize) + ((sizeof(bctrl_t) + psize) % kBlockSize > 0))
#define CRC_SIZE(psize) (((psize >> 3) + ((psize & 0x07) > 0)) << 3)
#define CTRL_BLOCK_SIZE(usize) static_cast<uint32_t>((usize>>11) & 0x1F)
#define CTRL_PAYLOAD_SIZE(usize) static_cast<uint32_t>(usize & 0x07FF) 
#define CTRL_UNION_SIZE(bsize, psize) static_cast<uint16_t>((bsize << 11) | (psize & 0x07FF))
#define MAX(a, b) (a > b ? a : b)

#ifdef TEST
#define LOG(msg) log(msg)
#else
#define LOG(msg) void();
#endif
#define ERROR(msg) log(msg); std::abort();


static thread_local const __m128i ZERO = _mm_setzero_si128();

inline uint16_t crc32(uint32_t crc, char *input, size_t len) {
    uint32_t *ptr = reinterpret_cast<uint32_t *>(input);
    while (len > 0) {
        crc = _mm_crc32_u32(crc, *ptr);
        len -= sizeof(uint32_t);
        ++ptr;
    }
    return crc ^ (crc >> 16);
}

inline void copy_key(char* dst, const char* src) {
    __m128i m0 = _mm_loadu_si128(((const __m128i*)src));
    _mm_storeu_si128(((__m128i*)dst), m0);
}

inline bool check_key(const char* dst, const char* src) {
    __m128i m0 = _mm_loadu_si128(((const __m128i*)src));
    __m128i m1 = _mm_loadu_si128(((const __m128i*)dst));
    return _mm_movemask_epi8(_mm_cmpeq_epi8(m0, m1)) == 0xFFFF;
}

inline void init_free_frames(FreeFrame *start, FreeFrame *const end) {
    do {
        start->next = start + 1;
    } while (++start < end);
    (end - 1)->next = nullptr;
}

Status DB::CreateOrOpen(const std::string& name, DB** dbptr, FILE* log_file) {
    return NvmEngine::CreateOrOpen(name, dbptr, log_file);
}

DB::~DB() {}

bool Accessor::empty() {
    return bucket == nullptr;
}

void Accessor::lock() {
    LOCK_ACQ(&bucket->lock);
}

void Accessor::unlock() {
    LOCK_RLS(&bucket->lock);
}

baddr_t Accessor::get() {
    return bucket->blocks[slot];
}


NvmEngine::NvmEngine(const std::string& name, FILE * _log_file) {
    log_file = _log_file;

    // LOG("start asserting");

    // assert(sizeof(Block) == kBlockSize);
    // assert(sizeof(Bucket) == kCacheLineSize*3);
    // assert(sizeof(bctrl_t) == sizeof(uint64_t));
    // assert(sizeof(balloc_t) == sizeof(uint64_t));
    // assert(sizeof(Local) == kLocalSize);
    // assert(sizeof(FreeFrame) == kFreeFrameSize);
    // assert(kEntriesPerFreeFrame == 509);
    // assert(kNumOfBlocks * sizeof(Block) <= kPMemSize);

    // assert(((uint64_t)blocks) % kCacheLineSize == 0);
    // assert(((uint64_t)&buckets) % kCacheLineSize == 0);
    // assert(((uint64_t)&free_frames) % kCacheLineSize == 0);
    // assert(((uint64_t)&locals) % kCacheLineSize == 0);

    // LOG("finished asserting");

    if ((blocks = (Block *)pmem_map_file(name.c_str(),
                    kNumOfBlocks * sizeof(Block),
                    PMEM_FILE_CREATE, 0666, nullptr,
                    nullptr)) == NULL) {
        perror("Pmem map file failed");
        exit(1);
    }

    for (int i = 0; i < kNumOfThreads; ++i) {
        locals[i].next_block = blocks + i * kBlocksPerPartition;
        locals[i].end_block = locals[i].next_block + kBlocksPerPartition;
        locals[i].serial = 0;
        locals[i].short_alloc = {0, 0};
        locals[i].next_free_frame = free_frames + i * kFreeFramesPerPartition;
        locals[i].end_free_frame = locals[i].next_free_frame + kFreeFramesPerPartition; 
    }

    ++locals[0].next_block;

    std::thread threads[kNumOfThreads];
    if (check_or_set_header()) {
        LOG("database recovery");
        for (int i = 0; i < kNumOfThreads; ++i) {
            threads[i] = std::thread([&](int t) {
                    recovery();
            }, i);
        }
        for (int i = 0; i < kNumOfThreads; ++i) {
            threads[i].join();
        }
    } else {
        LOG("database initialization");
        for (int i = 0; i < kNumOfThreads; ++i) {
            threads[i] = std::thread([&](int t) {
                init_free_frames(locals[t].next_free_frame, locals[t].end_free_frame);
            }, i);
        }
        for (int i = 0; i < kNumOfThreads; ++i) {
            threads[i].join();
        }
    }
    LOG("finished initialization");
}

bool NvmEngine::check_or_set_header() {
    if (blocks->ctrl.union_size == CTRL_UNION_SIZE(1, 10) &&
            blocks->ctrl.crc == 0x4242 &&
            blocks->ctrl.serial == 0 &&
            memcmp(blocks->payload.raw, "NVM STORE!", 10) == 0) {
        return true;
    } else {
        memset(blocks, 0, sizeof(Block));
        blocks->ctrl.union_size = CTRL_UNION_SIZE(1, 10);
        blocks->ctrl.crc = 0x4242;
        blocks->ctrl.serial = 0;
        memcpy(blocks->payload.raw, "NVM STORE!", 10);
        pmem_persist(blocks, sizeof(Block));
        return false;
    }
}

void *NvmEngine::operator new(size_t requested) {
    static size_t needed = sizeof(NvmEngine) + kCacheLineSize;
    void *alloc_mem = ::operator new(needed);
    return align_mem(kCacheLineSize, sizeof(NvmEngine), alloc_mem, needed);
}

Status NvmEngine::CreateOrOpen(const std::string& name, DB** dbptr, FILE* log_file) {
    NvmEngine* db = new NvmEngine(name, log_file);
    *dbptr = db;
    return Ok;
}

void NvmEngine::log(const char* s) {
   struct timeval NOW;
   gettimeofday(&NOW, NULL);
   stringstream stream;
   stream<<std::this_thread::get_id();
   fprintf(log_file, "%ld.%ld [t%s:c%d]: %s\n", NOW.tv_sec, NOW.tv_usec, stream.str().c_str(), sched_getcpu(), s);
   fflush(log_file);
}

void NvmEngine::recovery() {
    auto *const local = get_local();
    init_free_frames(local->next_free_frame, local->end_free_frame);

    while (local->next_block < local->end_block) {
        auto const block = local->next_block;
        auto const bsize = CTRL_BLOCK_SIZE(block->ctrl.union_size);
        auto const payload_size = CTRL_PAYLOAD_SIZE(block->ctrl.union_size);
        if (bsize >= 2 && bsize < kFreeListSize) {
            if (payload_size > 0 && payload_size <= kMaxPayloadSize
                    && block->ctrl.crc == crc32(0, block->payload.raw, CRC_SIZE(payload_size)))
            {
                auto accessor = search_index(block->payload.key); // try to find other record
                if ((!accessor.empty()) && blocks[accessor.get()].ctrl.serial > block->ctrl.serial) {
                    // found other record, the one with higher serial will win 
                    free({static_cast<uint32_t>(block - blocks), bsize});
                    local->next_block += bsize;
                    continue;
                }
                local->serial = MAX(local->serial, block->ctrl.serial);
                update_index(
                    block->payload.key,
                    { static_cast<uint32_t>(block - blocks), CTRL_BLOCK_SIZE(block->ctrl.union_size) }
                );
            } else {
                // invalid or free block, reset the ctrl
                block->ctrl.union_size = CTRL_UNION_SIZE(bsize, 0);
                block->ctrl.crc = 0;
                block->ctrl.serial = 0;
                pmem_persist(block, sizeof(bctrl_t));
                free({static_cast<uint32_t>(block - blocks), bsize});
            }
        } else {
            break;
        }
        local->next_block += bsize;
    }
}

Status NvmEngine::Get(const Slice& key, std::string* value) {
    auto accessor = search_index(key.data());
    if (!accessor.empty()) {
        std::unique_lock<Accessor> lock(accessor);
        auto payload_size = CTRL_PAYLOAD_SIZE(blocks[accessor.get()].ctrl.union_size);
        value->assign(blocks[accessor.get()].payload.value, payload_size - kKeySize);
        return Ok;
    }
    return NotFound;
}

Status NvmEngine::Set(const Slice& key, const Slice& value) {
    alignas(kCacheLineSize) Block buffer_blocks[kFreeListSize];

    auto hash = CityHash64(key.data(), kKeySize);
    __builtin_prefetch(buckets + BUCKET_INDEX(hash), 1, 2);

    copy_key(buffer_blocks->payload.key, key.data());
    memcpy(buffer_blocks->payload.value, value.data(), value.size());

    auto payload_size = value.size() + kKeySize;
    auto min_block_size = MIN_BLOCK_SIZE(payload_size);
    auto crc_size = CRC_SIZE(payload_size);
    auto balloc = alloc(min_block_size);

    buffer_blocks->ctrl.union_size = CTRL_UNION_SIZE(balloc.block_size, payload_size);
    buffer_blocks->ctrl.crc = crc32(0, buffer_blocks->payload.raw, crc_size);
    buffer_blocks->ctrl.serial = static_cast<uint32_t>(++get_local()->serial);

    pmem_memcpy(
        blocks + balloc.block,
        buffer_blocks,
        min_block_size * kBlockSize,
        PMEM_F_MEM_NONTEMPORAL | PMEM_F_MEM_WC | PMEM_F_MEM_NODRAIN
    );

    update_index(key.data(), balloc);

    pmem_drain();
    return Ok;
}

inline Accessor NvmEngine::search_index(const char* key) {
    auto hash = CityHash64(key, kKeySize); 
    auto bucket = buckets + BUCKET_INDEX(hash);

    digest_t key_digest = DIGEST(hash);

    for (int i = 0; i < kMaxHops; ++i) {
        auto m = match(key_digest, bucket->key_digest);

        while (m) {
            auto j = FIRST_MATCH(m);
            auto baddr = bucket->blocks[j];
            if (check_key(bucket->keys[j].data, key) && baddr > 0) {
                return Accessor(bucket, j);
            }
            m ^= (m & -m);
        }

        m = matchEmpty(bucket->key_digest);
        if (m) {
            return Accessor();
        }

        if ((++bucket) == buckets + kNumOfBuckets) {
            bucket = buckets;
        }
    }
    return Accessor();
}

inline void NvmEngine::update_index(const char* key, const balloc_t balloc) {
    auto hash = CityHash64(key, kKeySize);
    auto bucket = buckets + BUCKET_INDEX(hash);
    digest_t key_digest = DIGEST(hash);

    for (int i = 0; i < kMaxHops; ++i) {
        LOCK_ACQ(&bucket->lock);
        auto m = match(key_digest, bucket->key_digest);
        while (m) {
            auto j = FIRST_MATCH(m);
            if (check_key(bucket->keys[j].data, key)) {
                free({bucket->blocks[j], bucket->block_sizes[j]});
                bucket->blocks[j] = balloc.block;
                bucket->block_sizes[j] = static_cast<uint8_t>(balloc.block_size);
                LOCK_RLS(&bucket->lock);
                return;
            }
            m ^= (m & -m);
        }

        m = matchEmpty(bucket->key_digest);
        if (m) {
            auto j = FIRST_MATCH(m);
            bucket->key_digest[j] = key_digest;
            bucket->blocks[j] = balloc.block;
            bucket->block_sizes[j] = static_cast<uint8_t>(balloc.block_size);
            copy_key(bucket->keys[j].data, key);
            LOCK_RLS(&bucket->lock);
            return;
        }

        LOCK_RLS(&bucket->lock);
        if ((++bucket) == buckets + kNumOfBuckets) {
            bucket = buckets;
        }
    }

    ERROR("set: run out of hops!");
    std::abort();
}

inline balloc_t NvmEngine::alloc(const uint32_t block_size) {
    auto const local = get_local();

    if (local->short_alloc.block_size >= block_size ) {
        if (local->short_alloc.block_size - block_size >= 2) {
            balloc_t alloc = {local->short_alloc.block, block_size};
            local->short_alloc = {alloc.block + block_size, local->short_alloc.block_size - block_size};
            persist_free(local->short_alloc);
            return alloc;
        } else {
            balloc_t alloc = local->short_alloc;
            local->short_alloc = {0, 0};
            return alloc;
        }
    }

    if (local->next_block + kReservedBlocksPerPartition < local->end_block) {
        balloc_t alloc = {static_cast<baddr_t>(local->next_block - blocks), block_size};
        local->next_block += block_size;
        return alloc;
    }

    for (uint32_t i = block_size; i < kFreeListSize; ++i) {
        if (local->free_list[i] != nullptr) {
            auto frame = local->free_list[i];
            balloc_t alloc = {frame->blocks[--frame->top], i};
            if (i - block_size >= 2) {
                alloc.block_size = block_size;
                if (local->short_alloc.block_size >= 2) free(local->short_alloc);
                local->short_alloc = {alloc.block + block_size, i - block_size};
            }
            if (frame->top == 0) {
                local->free_list[i] = frame->next;
                frame->next = local->next_free_frame;
                local->next_free_frame = frame;
            }
            return alloc;
        }
    }

    if (local->next_block + block_size < local->end_block) {
        balloc_t alloc = {static_cast<baddr_t>(local->next_block - blocks), block_size};
        local->next_block += block_size;
        return alloc;
    }

    stringstream stream;
    stream<<"block_size: "<<block_size<<"|"<<local->next_block - blocks<<"|serial:"<<local->serial<<"||";
    for (int i = 0; i < kFreeListSize; ++i) {
        stream<<(local->free_list[i] != nullptr);
    }
    log(stream.str().c_str());

    ERROR("Out of memory!");
}

inline void NvmEngine::free(const balloc_t balloc) {
    auto const local = get_local();

    auto frame = local->free_list[balloc.block_size];
    if (frame == nullptr || frame->top >= kEntriesPerFreeFrame) {
        if (local->next_free_frame) {
            local->free_list[balloc.block_size] = local->next_free_frame;
            local->next_free_frame = local->next_free_frame->next;
            local->free_list[balloc.block_size]->next = frame;
            frame = local->free_list[balloc.block_size];
            frame->top = 0;
        } else {
            ERROR("Out of free frame!");
            std::abort();
        }
    }

    frame->blocks[frame->top++] = balloc.block;
}

inline void NvmEngine::persist_free(const balloc_t balloc) {
    alignas(kCacheLineSize) Block free_block;
    free_block.ctrl.union_size = CTRL_UNION_SIZE(balloc.block_size, 0);
    pmem_memcpy(blocks + balloc.block, &free_block, sizeof(free_block), PMEM_F_MEM_NONTEMPORAL | PMEM_F_MEM_WC);
}

inline int NvmEngine::match(const digest_t digest, const digest_t *target) {
    return _mm_movemask_epi8(
            _mm_cmpeq_epi16(
                _mm_set1_epi16(digest),
                _mm_loadu_si128((const __m128i *)target)
        )) & 0x55555555;
}

inline int NvmEngine::matchEmpty(const digest_t *target) {
    return _mm_movemask_epi8(
            _mm_cmpeq_epi16(ZERO, _mm_loadu_si128((const __m128i *)target))) & 0x55555555;
}

inline Local *NvmEngine::get_local() {
    static thread_local Local *const local = &locals[assign_local()];
    return local;
}

inline int NvmEngine::assign_local() {
    const auto tid = (thread_counter++) % kNumOfThreads;
    LOG("start thread");
    return tid;
}

NvmEngine::~NvmEngine() {
}
