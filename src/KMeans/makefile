# This is the makefile to output the Dcups.o object file which will be used 
# along with the Dcups.h file to write the DCuPS model onto the desired platform

# source files
SRC = runKMeans.cpp
SRC += ./KMeans.cpp

OUT = runKMeans

#compiler
CPP = g++

#compiler flags (-g -O2 -Wall)
CPPFLAGS = -Wall
CPPFLAGS += -O2
CPPFLAGS += -msse2
CPPFLAGS += -stdlib=libc++
CPPFLAGS += -I/usr/local/include
CPPFLAGS += -I/usr/include
CPPFLAGS += -L/usr/local/lib

LIBS += -l boost_date_time

.SUFFIXES : .cpp

OBJ = $(SRC:.cpp = .o)

$(OUT) : $(OBJ)
	$(CPP) $(OBJ) -o $(OUT) $(CPPFLAGS) $(LIBS)

clean :
	rm -f $(OUT)
