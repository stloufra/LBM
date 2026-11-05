# workaround for an Ubuntu issue:
# https://askubuntu.com/q/1351696
SHELL := bash 

include ./src/config/config.mk

SOURCES := $(wildcard *.cpp)
TARGETS := $(SOURCES:%.cpp=%)
CUDA_SOURCES := $(wildcard *.cu)
CUDA_TARGETS := $(CUDA_SOURCES:%.cu=%)

## targets definitions follow
.PHONY: all host cuda host-sim test test-cuda
all: host cuda
host: $(TARGETS)
cuda: $(CUDA_TARGETS)

# CUDA sources built for TNL Host backend (OpenMP), no nvcc needed.
host-sim: $(CUDA_SOURCES:%.cu=%-host)

%-host: %.cu
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -x c++ -o $@ $< $(LDLIBS)

test: tests/conservation-host
	./tests/conservation-host

test-cuda: tests/conservation
	./tests/conservation

.PHONY: clean
clean:
	rm -f *.d *.o *.cuo results/*.vtk results/*.csv $(TARGETS) $(CUDA_TARGETS) *-host tests/*-host tests/*.cuo tests/conservation

.PHONY: run
run:
	make clean
	make
	##./preprocess
	./main
	

$(TARGETS): % : %.o
	$(CXX) $(LDFLAGS) -o $@ $< $(LDLIBS)

# use .cuo instead of .cu.o to avoid problems with the implicit rules: https://stackoverflow.com/q/62967939
# (and use the host compiler for linking CUDA, nvcc does not understand that .cuo is an object file)
$(CUDA_TARGETS): % : %.cuo
	$(CXX) $(CUDA_LDFLAGS) -o $@ $< $(CUDA_LDLIBS)

$(SOURCES:%.cpp=%.o): %.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

tests/conservation: tests/conservation.cuo
	$(CXX) $(CUDA_LDFLAGS) -o $@ $< $(CUDA_LDLIBS)

$(CUDA_SOURCES:%.cu=%.cuo) tests/conservation.cuo: %.cuo : %.cu
	$(CUDA_CXX) $(CUDA_CPPFLAGS) $(CUDA_CXXFLAGS) -c -o $@ $<


