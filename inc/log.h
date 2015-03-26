#ifndef __LOG_H__
#define __LOG_H__

#include <string>
#include <fstream>
#include <ctime>
#include <iostream>

std::string date_string();
bool exists(std::string fileName);

struct Log {
	Log(std::string filename);

	// template <typename T1>
	// void print(T1 a);

	// template <typename T1, typename T2>
	// void print(T1 a, T2 b);


	template <typename T1, typename T2, typename T3>
	void print(T1 a, T2 b = "", T3 c = "");

	std::ofstream file;
};

template <typename T>
inline Log &operator<<(Log &out, T const &in) {
	std::cout << in;
	out.file << in;
	out.file.flush();
	return out;
}

#endif