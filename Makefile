EXE = solve
OBJ = main.o
CFLAGS = -std=c++11 -O3
COMPILERS = g++ $(CFLAGS)

solve: $(OBJ)
	$(COMPILERS) -o $(EXE) $(OBJ)

main.o: main.cpp Solution.h Structure.h Vector.h Array.h 
	$(COMPILERS) -c main.cpp

.PHONY: clean
clean:
	rm $(EXE) $(OBJ)
