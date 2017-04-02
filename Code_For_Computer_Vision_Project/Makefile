CXX = g++
CXXFLAGS = -O2 -g -Wall -fmessage-length=0 
CPPFLAGS = -Imingw/include/

OBJS =  main.o binarizewolfjolion.o

LDFLAGS = -L/usr/local/opt/opencv3/lib/
LDLIBS =  -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc

TARGET = main

.PHONY: all
all: $(TARGET)

$(TARGET):  $(OBJS)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)


# OPENCVINCLUDE	= -I/usr/local/Cellar/opencv3/3.1.0_2/include/
# OPENCVLIBPATH   = -L/usr/local/Cellar/opencv3/3.1.0_2/lib/
# OPENCVLIB 		= -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc


# CC = g++
# CFLAGS = -g -Wall 
# OBJ = main.o 


# %.o: %.cpp 
# 	$(CC) $(CFLAGS) -c -o $@ $< $(CFLAGS) $(OPENCVINCLUDE) $(OPENCVLIBPATH) $(OPENCVLIB)


# main: $(OBJ)
# 	$(CC) $(CFLAGS) -o $@ $^ $(CFLAGS) $(OPENCVINCLUDE) $(OPENCVLIBPATH) $(OPENCVLIB)

