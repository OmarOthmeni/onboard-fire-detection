CXX      ?= g++
CXXFLAGS ?= -O2
CXXFLAGS += -std=c++17 -Wall -Wextra -Isrc

firedet: src/main.cpp src/fire.cpp src/fire.h
	$(CXX) $(CXXFLAGS) src/main.cpp src/fire.cpp -o $@

.PHONY: clean
clean:
	rm -f firedet
