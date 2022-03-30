CUDA ?= /usr/local/cuda
NVCC ?= ${CUDA}/bin/nvcc

NVCC_FLAGS := -O2 --disable-warnings -ccbin gcc -Xcompiler -fPIC

all: nv6502.so nv6502

nv6502.so: nv6502.cu nv6502.h Makefile
	${NVCC} -ccbin gcc nv6502.cu ${NVCC_FLAGS} -shared -o $@

nv6502: nv6502.cu nv6502.h Makefile
	${NVCC} -ccbin gcc nv6502.cu ${NVCC_FLAGS} -o $@

%.o: %.cu
	${NVCC} ${NVCC_FLAGS} -o $@ -c $<

clean:
	rm -rf *.so *.o nv6502
