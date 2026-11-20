#pragma once

// Linux. This header is included after scan.cu's Output declaration. Only
// callers opting into verified, idempotent two-file publication use it.
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace paired_output {

inline void fail(const char* operation) {
    const int code=errno;
    throw std::system_error(code,std::generic_category(),operation);
}

struct Fd {
    int value=-1;
    Fd()=default;
    explicit Fd(int fd):value(fd){}
    ~Fd(){if(value>=0)::close(value);}
    Fd(const Fd&)=delete;
    Fd& operator=(const Fd&)=delete;
};

inline struct stat stat_fd(int fd) {
    struct stat s{};
    if(::fstat(fd,&s))fail("inspect pair file");
    return s;
}
inline bool same_inode(const struct stat& a,const struct stat& b) {
    return a.st_dev==b.st_dev && a.st_ino==b.st_ino;
}
inline bool same_version(const struct stat& a,const struct stat& b) {
    // A new hard link updates ctime and link count, not data or mtime.
    return S_ISREG(b.st_mode) && same_inode(a,b) && a.st_size==b.st_size &&
           a.st_mtim.tv_sec==b.st_mtim.tv_sec && a.st_mtim.tv_nsec==b.st_mtim.tv_nsec;
}
inline void read_exact(int fd,unsigned char* out,std::size_t bytes,off_t offset) {
    std::size_t count=0;
    while(count<bytes) {
        const auto size=::pread(fd,out+count,bytes-count,offset+off_t(count));
        if(size<0 && errno==EINTR) continue;
        if(size<0)fail("compare pair file");
        if(!size)throw std::runtime_error("pair file changed while reading");
        count+=std::size_t(size);
    }
}
inline bool same_bytes(int a,int b,off_t size) {
    std::array<unsigned char,65536> x,y;
    for(off_t off=0;off<size;) {
        const auto n=std::size_t(std::min<off_t>(size-off,off_t(x.size())));
        read_exact(a,x.data(),n,off);read_exact(b,y.data(),n,off);
        if(std::memcmp(x.data(),y.data(),n))return false;
        off+=off_t(n);
    }
    return true;
}

// Own a random mode-0700 staging directory in the destination filesystem.
// Pin both directory descriptors so moving/symlinking a parent path cannot
// redirect publication or cleanup. Unlinks are confined to our staging area.
struct Directory {
    Fd parent,area;
    std::string destination,private_name,inner_temp;
    explicit Directory(const std::string& target) {
        const std::filesystem::path path(target);
        destination=path.filename().string();
        if(destination.empty() || destination=="." || destination=="..")
            throw std::invalid_argument("pair destination must name a file");
        auto dir=path.parent_path();
        parent.value=::open(dir.empty()?".":dir.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC);
        if(parent.value<0)fail("open pair destination directory");
        private_name=".pearl-out-XXXXXX";
        std::string pattern="/proc/self/fd/"+std::to_string(parent.value)+"/"+private_name;
        inner_temp="data.part."+std::to_string(::getpid());
        if(!::mkdtemp(pattern.data()))fail("create private pair stage");
        std::copy_n(pattern.end()-private_name.size(),private_name.size(),private_name.begin());
        area.value=::openat(parent.value,private_name.c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
        if(area.value<0) {
            const int saved=errno;
            ::unlinkat(parent.value,private_name.c_str(),AT_REMOVEDIR);
            errno=saved;fail("open private pair stage");
        }
    }
    std::string inner_path() const {return "/proc/self/fd/"+std::to_string(area.value)+"/data";}
    ~Directory() {
        if(area.value<0)return;
        ::unlinkat(area.value,inner_temp.c_str(),0);
        const auto path_stat=[&](struct stat& s){return ::fstatat(parent.value,private_name.c_str(),&s,AT_SYMLINK_NOFOLLOW)==0;};
        struct stat named{},owned{};
        if(::fstat(area.value,&owned)==0 && path_stat(named) && same_inode(named,owned))
            ::unlinkat(parent.value,private_name.c_str(),AT_REMOVEDIR);
    }
};

class File {
    Directory dir_;
    Output staged_;
    Fd source_,existing_;
    struct stat fresh_{},verified_{};
    bool checked_=false;

    bool inspect(bool missing_allowed) {
        // O_NONBLOCK avoids waiting on a FIFO; symlinks never qualify.
        Fd file(::openat(dir_.parent.value,dir_.destination.c_str(),O_RDONLY|O_NONBLOCK|O_NOFOLLOW|O_CLOEXEC));
        if(file.value<0) {
            if(errno==ENOENT && missing_allowed)return false;
            fail("open existing pair destination");
        }
        const auto observed=stat_fd(file.value);
        if(!S_ISREG(observed.st_mode))throw std::runtime_error("pair destination is not a regular file");
        if(checked_ && same_version(verified_,observed))return true;
        if(observed.st_size!=fresh_.st_size ||
           (!same_inode(observed,fresh_) && !same_bytes(source_.value,file.value,observed.st_size)) ||
           !same_version(observed,stat_fd(file.value)) || !same_version(fresh_,stat_fd(source_.value)))
            throw std::runtime_error("existing pair destination differs from this request");
        // Keep the verified inode alive through both links and the final checks.
        std::swap(existing_.value,file.value);verified_=observed;checked_=true;
        return true;
    }
public:
    std::ofstream& file;
    explicit File(const std::string& target):dir_(target),staged_(dir_.inner_path()),file(staged_.file){}
    int borrowed_staging_dir_fd() const noexcept { return dir_.area.value; }
    bool same_destination(const File& other) const {
        return dir_.destination==other.dir_.destination && same_inode(stat_fd(dir_.parent.value),stat_fd(other.dir_.parent.value));
    }
    void seal() {
        file.close(); // throws on flush/close errors before either public link
        source_.value=::openat(dir_.area.value,dir_.inner_temp.c_str(),O_RDONLY|O_NOFOLLOW|O_CLOEXEC);
        if(source_.value<0)fail("open staged pair data");
        fresh_=stat_fd(source_.value);
        if(!S_ISREG(fresh_.st_mode))throw std::runtime_error("staged pair data is not regular");
    }
    void preflight(){(void)inspect(true);}
    void publish() {
        if(::linkat(dir_.area.value,dir_.inner_temp.c_str(),dir_.parent.value,dir_.destination.c_str(),0)==0) {
            verified_=fresh_;checked_=true;return;
        }
        if(errno!=EEXIST)fail("publish pair link");
        (void)inspect(false);
    }
    void confirm() const {
        struct stat named{};
        if(!checked_ || ::fstatat(dir_.parent.value,dir_.destination.c_str(),&named,AT_SYMLINK_NOFOLLOW)!=0 ||
           !same_version(verified_,named))throw std::runtime_error("pair destination was replaced during publication");
    }
};

inline void finish(File& roots,File& hits) {
    if(roots.same_destination(hits))throw std::invalid_argument("root receipt and ticket output must be distinct");
    roots.seal();hits.seal();
    roots.preflight();hits.preflight();
    // POSIX cannot expose two unrelated names atomically. Always publish roots
    // first. A retry recomputes both and checks the surviving link byte-for-byte.
    // Never remove a public link, even after a collision or external replacement.
    roots.publish();hits.publish();
    roots.confirm();hits.confirm();
}

} // namespace paired_output
