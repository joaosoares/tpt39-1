EXE=videofilter
CVINCFLAGS=`pkg-config --libs --cflags opencv`
CVLIBFLAGS=`pkg-config --libs opencv`
DBGFLAGS= -pg
GCC=arm-linux-gnueabihf-g++
SRCS=videofilter.cpp
OCLLIBSDIR=/opt/ComputeLibrary/build/
OCLINCSDIR=/opt/ComputeLibrary/include/
MGD=/opt/Mali_Graphics_Debugger_v4.4.1.0271762a_Linux_x64/target/linux/hard_float/
FLAGS=-g -Wno-deprecated-declarations -Wall -DARCH_ARM -Wextra -Wno-unused-parameter -pedantic -Wdisabled-optimization -Wformat=2 -Winit-self -Wstrict-overflow=2 -Wswitch-default -fpermissive -std=gnu++11 -Wno-vla -Woverloaded-virtual -Wctor-dtor-privacy -Wsign-promo -Weffc++ -Wno-format-nonliteral -Wno-overlength-strings -Wno-strict-overflow -Wlogical-op -Wnoexcept -Wstrict-null-sentinel -march=armv7-a -mthumb -mfpu=neon -mfloat-abi=hard -Werror -O3 -ftree-vectorize -fstack-protector-strong -DARM_COMPUTE_CL -I${OCLINCSDIR} -I.. -I..  
LDFLAGS=-L${OCLLIBSDIR} -larm_compute -larm_compute_core -lOpenCL

OTHER_FILES=gpu.cpp perf.cpp
all:${EXE}

${EXE}: ${SRCS}
	${GCC} ${DBGFLAGS} ${FLAGS} ${CVINCFLAGS} ${SRCS} ${OTHER_FILES} ${CVLIBFLAGS} ${LDFLAGS} -o ${EXE}

debug:${EXE}
	LD_PRELOAD=${MGD}/libinterceptor.so ./${EXE}

profile:
	./${EXE}
	gprof ./${EXE}  gmon.out > prof.txt
clean:
	rm -rf ${EXE} *.o
