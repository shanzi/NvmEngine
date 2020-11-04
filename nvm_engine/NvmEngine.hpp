#ifndef TAIR_CONTEST_KV_CONTEST_NVM_ENGINE_H_
#define TAIR_CONTEST_KV_CONTEST_NVM_ENGINE_H_

#include "include/db.hpp"

#include <sched.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <atomic>
#include <sched.h>
#include <include/db.hpp>


typedef uint64_t lock_t;
typedef uint16_t digest_t;
typedef uint32_t baddr_t;
typedef union {
    struct __attribute__((packed)) {
        uint16_t union_size; // | 5bit: block size | 11bit payload size |
        uint16_t crc;
        uint32_t serial;
    };
    uint64_t value;
} bctrl_t;
struct key_data_t {
    char data[16];
};

#ifdef DEBUG
static constexpr int kNumOfThreads = 4;
static constexpr size_t kMemSize = 0x80000000;
static constexpr size_t kPMemSize = 0xe0000000;
static constexpr size_t kNumOfFreeFrames = 0x80000;
#else
static constexpr int kNumOfThreads = 16;
static constexpr size_t kMemSize = 0x180000000;
static constexpr size_t kPMemSize = 0x1000000000;
static constexpr size_t kNumOfFreeFrames = 0x80000;
#endif
static constexpr int kKeySize = 16;
static constexpr int kBlockSize = 64;
static constexpr int kCacheLineSize = 64;
static constexpr int kBucketSize = kCacheLineSize * 3;
static constexpr int kEntriesPerBucket = 8;
static constexpr int kLocalSize = kCacheLineSize * 3;
static constexpr int kFreeListSize = 18;
static constexpr int kFreeFrameSize = 2048;
static constexpr int kEntriesPerFreeFrame = (kFreeFrameSize - sizeof(void *) - sizeof(uint32_t))/sizeof(baddr_t);
static constexpr int kMaxPayloadSize = (kFreeListSize - 1)*kBlockSize - sizeof(bctrl_t);
static constexpr size_t kNumOfBlocks = kPMemSize / kBlockSize;
static constexpr size_t kNumOfBuckets = kMemSize / kBucketSize;
static constexpr size_t kBlocksPerPartition = kNumOfBlocks / kNumOfThreads;
static constexpr size_t kReservedBlocksPerPartition = kBlocksPerPartition >> 3;
static constexpr size_t kFreeFramesPerPartition = kNumOfFreeFrames / kNumOfThreads;
static constexpr int kMaxHops = 1024;

struct balloc_t {
    baddr_t block;
    uint32_t block_size;
};

typedef union {
    struct {
        char key[kKeySize];
        char value[];
    }; 
    char raw[kKeySize];
} bpayload_t;

struct alignas(kCacheLineSize) Bucket {
    lock_t lock;
    digest_t key_digest[kEntriesPerBucket];
    baddr_t blocks[kEntriesPerBucket];
    uint8_t block_sizes[kEntriesPerBucket];
    key_data_t keys[8];
};

union alignas(kBlockSize) Block {
    struct {
        bctrl_t ctrl;
        bpayload_t payload;
    };
    char padding[kBlockSize];
};

struct alignas(kCacheLineSize) FreeFrame {
    uint32_t top;
    baddr_t blocks[kEntriesPerFreeFrame];
    FreeFrame *next;
};

struct alignas(kCacheLineSize) Local {
    // memory management
    Block* next_block;
    Block* end_block;
    size_t serial;
    balloc_t short_alloc;
    FreeFrame *next_free_frame;
    FreeFrame *end_free_frame;
    FreeFrame *free_list[kFreeListSize];
};

inline void *align_mem( std::size_t alignment, std::size_t size,
        void *&ptr, std::size_t &space ) {
    std::uintptr_t pn = reinterpret_cast< std::uintptr_t >( ptr );
    std::uintptr_t aligned = ( pn + alignment - 1 ) & - alignment;
    std::size_t padding = aligned - pn;
    if ( space < size + padding ) return nullptr;
    space -= padding;
    return ptr = reinterpret_cast< void * >( aligned );
} 

class Accessor {
    Bucket* const bucket;
    const int slot;

    public:

    Accessor(): bucket(nullptr), slot(-1) {}
    Accessor(Bucket* const b, const int s): bucket(b), slot(s) {}

    void lock();
    void unlock();
    bool empty();
    baddr_t get();
};

class NvmEngine : DB {
public:
    /**
     * @param 
     * name: file in AEP(exist)
     * dbptr: pointer of db object
     *
     */

    static Status CreateOrOpen(const std::string& name, DB** dbptr, FILE* log_file = nullptr);
    void *operator new(size_t size);
    NvmEngine(const std::string& name, FILE* log_file = nullptr);
    Status Get(const Slice& key, std::string* value);
    Status Set(const Slice& key, const Slice& value);
    ~NvmEngine();
private:
    struct alignas(kCacheLineSize) {
        FILE *log_file = nullptr;
        Block *blocks = nullptr;
        char padding[kCacheLineSize - sizeof(log_file) - sizeof(blocks)];
    };

    alignas(kCacheLineSize) std::atomic<int> thread_counter = {0};
    alignas(kCacheLineSize) Bucket buckets[kNumOfBuckets];
    alignas(kCacheLineSize) FreeFrame free_frames[kNumOfFreeFrames];
    alignas(kCacheLineSize) Local locals[kNumOfThreads];

    // methods
    void log(const char* s);
    bool check_or_set_header();
    void recovery();

    void update_index(const char* key, const balloc_t balloc);
    Accessor search_index(const char* key);
    balloc_t alloc(const uint32_t size);
    void free(const balloc_t block);
    void persist_free(const balloc_t block);

    int match(const digest_t digest, const digest_t* target);
    int matchEmpty(const digest_t* target);

    Local *get_local();
    int assign_local();
};

#endif
