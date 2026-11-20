#pragma once
#include <array>
#include <cerrno>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <fcntl.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <unistd.h>

namespace pearl_square::private_spill {

class Fd {
    int value_;
public:
    explicit Fd(int value) : value_(value) {}
    ~Fd() { if (value_ >= 0) ::close(value_); }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    int get() const noexcept { return value_; }
    int release() noexcept { return std::exchange(value_, -1); }
};

inline struct stat staging_directory(int fd) {
    struct stat info{};
    const int flags=::fcntl(fd,F_GETFD);
    if(flags<0 || !(flags&FD_CLOEXEC) || ::fstat(fd,&info) || !S_ISDIR(info.st_mode) ||
       info.st_uid!=::geteuid() || (info.st_mode&07777)!=0700)
        throw std::runtime_error("invalid private ticket staging directory");
    return info;
}

inline auto random_name() {
    std::array<unsigned char,16> nonce{};
    for(std::size_t used=0;used<nonce.size();) {
        const auto n=::getrandom(nonce.data()+used,nonce.size()-used,0);
        if(n<0 && errno==EINTR)continue;
        if(n<=0)throw std::runtime_error("cannot select a private ticket file name");
        used+=std::size_t(n);
    }
    std::array<char,34> name{};name[0]='.';
    constexpr char hex[]="0123456789abcdef";
    for(std::size_t i=0;i<nonce.size();++i){name[1+2*i]=hex[nonce[i]>>4];name[2+2*i]=hex[nonce[i]&15];}
    return name;
}

using Unlink=int(*)(int,const char*,int);
// Returned descriptors have no directory entry. This helper never writes to a
// named file, including on unlink failure. The private directory is pinned by
// its FD and the O_EXCL name cannot replace an existing file.
inline int named_unlinked(int dir,Unlink remove=::unlinkat) {
    (void)staging_directory(dir);
    for(unsigned attempt=0;attempt<16;++attempt) {
        const auto name=random_name();
        Fd fd(::openat(dir,name.data(),O_CREAT|O_EXCL|O_NOFOLLOW|O_RDWR|O_CLOEXEC,0600));
        if(fd.get()<0){if(errno==EEXIST)continue;throw std::runtime_error("cannot create private ticket spill");}
        int result;
        do {result=remove(dir,name.data(),0);}while(result<0 && errno==EINTR);
        if(result<0) {
            // Only the empty newly created inode is eligible for cleanup. A
            // permanent filesystem failure can at worst leave an empty file.
            struct stat own{},named{};
            if(::fstat(fd.get(),&own)==0 && ::fstatat(dir,name.data(),&named,AT_SYMLINK_NOFOLLOW)==0 &&
               own.st_dev==named.st_dev && own.st_ino==named.st_ino && own.st_size==0) {
                do {result=::unlinkat(dir,name.data(),0);}while(result<0 && errno==EINTR);
            }
            throw std::runtime_error("cannot unlink private ticket spill before writing");
        }
        return fd.release();
    }
    throw std::runtime_error("cannot allocate unique private ticket spill");
}

inline std::FILE* open(int dir,bool prefer_unnamed=true) {
    const auto directory=staging_directory(dir);
    int raw=-1;
#ifdef O_TMPFILE
    if(prefer_unnamed) {
        raw=::openat(dir,".",O_TMPFILE|O_RDWR|O_CLOEXEC|O_EXCL,0600);
        if(raw<0 && errno!=EOPNOTSUPP && errno!=EINVAL && errno!=EISDIR && errno!=ENOENT && errno!=ENOSYS)
            throw std::runtime_error("cannot allocate unnamed private ticket spill");
    }
#else
    (void)prefer_unnamed;
#endif
    Fd fd(raw<0?named_unlinked(dir):raw);
    struct stat info{};
    const int flags=::fcntl(fd.get(),F_GETFD);
    // umask may be more restrictive than 0600. The file is already nameless,
    // and no ticket bytes are written before its exact access state is checked.
    if(::fchmod(fd.get(),0600) || ::fstat(fd.get(),&info) || !S_ISREG(info.st_mode) || info.st_nlink!=0 ||
       (info.st_mode&07777)!=0600 || info.st_uid!=::geteuid() || info.st_dev!=directory.st_dev ||
       info.st_size!=0 || flags<0 || !(flags&FD_CLOEXEC))
        throw std::runtime_error("private ticket spill failed file checks");
    auto* file=::fdopen(fd.get(),"w+b");
    if(!file)throw std::runtime_error("cannot open checked private ticket stream");
    fd.release();return file;
}
}  // namespace pearl_square::private_spill
