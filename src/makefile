# This is the makefile to output the runTest file which will be used 
# along with the three supplementary files to run the model 

# source files
SRC = runTest_1_0_0.cpp
SRC += HyperParam/HyperParam.cpp
SRC += Metrics/Metrics.cpp
SRC += KMeans/KMeans.cpp

OUT = runTest_1_0_0

#compiler
CPP = g++

#compiler flags (-g -O2 -Wall)
#CPPFLAGS = -Wall
CPPFLAGS = -O2
CPPFLAGS += -msse2
CPPFLAGS += -stdlib=libc++
CPPFLAGS += -I/usr/local/include
CPPFLAGS += -I/usr/include
CPPFLAGS += -L/usr/local/lib

LIBS = -l boost_thread
LIBS += -l boost_system
LIBS += -l boost_date_time

.SUFFIXES : .cpp

OBJ = $(SRC:.cpp = .o)

$(OUT) : $(OBJ)
	$(CPP) $(OBJ) -o $(OUT) $(CPPFLAGS) $(LIBS)

clean :
	rm -f $(OUT)
