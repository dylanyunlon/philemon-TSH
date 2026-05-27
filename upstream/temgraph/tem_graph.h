#ifndef TEM_GRAPH_H
#define TEM_GRAPH_H

#include "interval.h"
#include "dll_list.h"


class TemGraph{
public:
    Timestamp earliest_time_, latest_time_;
    RecordId total_intervals_, unique_intervals_;

    TemGraph(){
        earliest_time_ = -1;
        latest_time_ = -1;
        total_intervals_ = 0;
        my_list = List();
        T.clear();
    }
    void load_intervals(int query_type, const string& input_file);
    void build_index(vector<RecordId> &a, vector<RecordId> &b);
    void build_index_contained_overlaps(vector<RecordId> &a, vector<RecordId> &b);
    int contains_query(Timestamp l, Timestamp r);
    int contained_query(Timestamp l, Timestamp r);


private:
    vector<TInterval> T;
    vector<TInterval> T_unique_;
    vector<RecordId> T_id_;
    vector<RecordId> sorted_by_start_;
    List my_list;

    vector<vector<OutNeighbor>>   next;
    vector<vector<pair<RecordId, RecordId >>> in_neighbors;    // first: location in next, second: offset in next[i]

};


#endif //TEM_GRAPH_H