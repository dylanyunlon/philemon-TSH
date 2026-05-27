#ifndef INTERVAL_H
#define INTERVAL_H

#include <sys/time.h>
#include <random>
#include <algorithm>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <execution>
#include <chrono>
#include <sys/resource.h>

using namespace std;

#define CONTAINS_QUERY 1
#define OTHER_QUERY 2

// #define READ_BINARY


long long visited_intervals_;

typedef int Timestamp;
typedef uint32_t SuccessorLoc;
typedef uint32_t RecordId;  


void print_peak_memory_usage() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        // ru_maxrss KB 
        double max_mb = usage.ru_maxrss / 1024.0; // Convert to MB
        double max_gb = usage.ru_maxrss / 1024.0 / 1024.0; // Convert to GB
        std::cout << "Peak memory usage: " << usage.ru_maxrss << " KB or " << max_mb << " MB or " << max_gb << " GB" << std::endl;
    } else {
        std::cerr << "Failed to get memory usage information." << std::endl;
    }
}

struct Interval {
    RecordId id;
    Timestamp start, end;
    Interval(){}
    Interval(RecordId id, int start, int end) : id(id), start(start), end(end) {}
    bool operator<(const Interval& other) const {   // sorted by start increasing order
        if(start == other.start) return end < other.end;
        return start < other.start;
    }


};


class TInterval {
public:
    RecordId id;
    Timestamp l, r;
    TInterval(RecordId _id, int _l, int _r){
        id = _id;
        l = _l;
        r = _r;
    }

    bool operator<(const TInterval& other) const {  
        if (r == other.r && l == other.l) return id < other.id;
        if (r == other.r) return l < other.l;
        return r < other.r;
    }

};


struct OutNeighbor {
    Timestamp l;
    RecordId x;
    SuccessorLoc successor;
    OutNeighbor(int _l, RecordId _x, SuccessorLoc _successor){
        l = _l;
        x = _x;
        successor = _successor;
    }
};


int random(int low, int high){
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(low, high);
    return dist(gen);
}

double GetTime(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

string toUpperCase(char *buf)
{
    auto i = 0;
    while (buf[i])
    {
        buf[i] = toupper(buf[i]);
        i++;
    }
    
    return string(buf);
}
#endif //INTERVAL_H
