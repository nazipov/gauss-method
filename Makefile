main: main.o
	gcc main.o -L/opt/AMDAPP/lib/x86_64/ -lOpenCL -lm -o main

main.o: main.c
	gcc -c -std=c99 main.c

clean:
	rm *.o main
