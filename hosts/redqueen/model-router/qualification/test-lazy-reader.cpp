#include "llama-lazy-reader.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iterator>
#include <stdexcept>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>

static void require(bool value, const char * message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

static int temp_file() {
    char path[] = "/tmp/llama-lazy-reader-XXXXXX";
    const int fd = ::mkstemp(path);
    if (fd < 0) {
        throw std::runtime_error("mkstemp failed");
    }
    ::unlink(path);
    return fd;
}

static void write_all(int fd, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    size_t done = 0;
    while (done < size) {
        const ssize_t written = ::write(fd, bytes + done, size - done);
        if (written <= 0) {
            throw std::runtime_error("write failed");
        }
        done += (size_t) written;
    }
}

int main() {
    try {
        constexpr int64_t width = 4;
        constexpr int64_t n_rows = 6;
        constexpr size_t base = 16;
        std::vector<uint8_t> prefix(base, 0xa5);
        std::vector<ggml_fp16_t> encoded((size_t) width * n_rows);
        std::vector<float> expected((size_t) width * n_rows);
        for (int64_t row = 0; row < n_rows; ++row) {
            for (int64_t col = 0; col < width; ++col) {
                const float value = (float) (row * 10 + col) / 3.0f - 4.0f;
                encoded[(size_t) row * width + col] = ggml_fp32_to_fp16(value);
                expected[(size_t) row * width + col] = ggml_fp16_to_fp32(encoded[(size_t) row * width + col]);
            }
        }

        const int fd = temp_file();
        write_all(fd, prefix.data(), prefix.size());
        write_all(fd, encoded.data(), encoded.size() * sizeof(ggml_fp16_t));

        llama_lazy_reader reader(fd, base, ggml_row_size(GGML_TYPE_F16, width), n_rows, 4, GGML_TYPE_F16, width);
        const int32_t rows[] = { 0, 1, 2, 1, 4, 0 };
        constexpr int64_t row_offset = 1;
        std::vector<float> gathered(std::size(rows) * width);
        reader.gather(rows, (int64_t) std::size(rows), row_offset, gathered.data());

        for (size_t slot = 0; slot < std::size(rows); ++slot) {
            for (int64_t col = 0; col < width; ++col) {
                const float want = expected[(size_t) (rows[slot] + row_offset) * width + col];
                require(gathered[slot * width + col] == want, "direct read differs from F16 row dequantization or row-offset gather");
            }
        }

        const int short_fd = temp_file();
        write_all(short_fd, prefix.data(), prefix.size());
        write_all(short_fd, encoded.data(), ggml_row_size(GGML_TYPE_F16, width));
        llama_lazy_reader short_reader(short_fd, base, ggml_row_size(GGML_TYPE_F16, width), 2, 1, GGML_TYPE_F16, width);
        bool rejected_short_read = false;
        try {
            const int32_t short_row[] = { 1 };
            float output[width] = {};
            short_reader.gather(short_row, 1, 0, output);
        } catch (const std::runtime_error &) {
            rejected_short_read = true;
        }
        require(rejected_short_read, "short direct read was silently accepted");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
#else
int main() { return 0; }
#endif
