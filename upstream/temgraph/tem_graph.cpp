

#include "tem_graph.h"

vector<TInterval> _T;
bool comp_L(RecordId x, RecordId y){
    if(_T[x].l == _T[y].l && _T[x].r == _T[y].r)  return _T[x].id > _T[y].id;
    if(_T[x].l == _T[y].l)  return _T[x].r > _T[y].r;
    return _T[x].l < _T[y].l;
}


bool comp_R(RecordId x, RecordId y){
    if(_T[x].l == _T[y].l && _T[x].r == _T[y].r)  return _T[x].id < _T[y].id;
    if(_T[x].r == _T[y].r)  return _T[x].l < _T[y].l;
    return _T[x].r < _T[y].r;
}

bool comp_L1(RecordId x, RecordId y){
    if(_T[x].l == _T[y].l && _T[x].r == _T[y].r)  return _T[x].id < _T[y].id;
    if(_T[x].l == _T[y].l)  return _T[x].r < _T[y].r;
    return _T[x].l < _T[y].l;
}


bool comp_R1(RecordId x, RecordId y){
    if(_T[x].l == _T[y].l && _T[x].r == _T[y].r)  return _T[x].id < _T[y].id;
    if(_T[x].r == _T[y].r)  return _T[x].l < _T[y].l;
    return _T[x].r < _T[y].r;
}



void TemGraph::load_intervals(int query_type, const string& input_file){
    T.clear();
    RecordId id;
    int start_timestamp, end_timestamp;
    total_intervals_ = 0; earliest_time_ = -1; latest_time_ = -1;
#ifdef READ_BINARY
    std::ifstream inFile(input_file, std::ios::binary);  //
    if (!inFile) {
        std::cerr << "Error opening file for reading: " << input_file << std::endl;
        return;
    }
    // while (inFile.read(reinterpret_cast<char*>(&id), sizeof(id))) {
    while (inFile.read(reinterpret_cast<char*>(&start_timestamp), sizeof(start_timestamp))) {
        
        inFile.read(reinterpret_cast<char*>(&end_timestamp), sizeof(end_timestamp));
        T.push_back(TInterval(total_intervals_, start_timestamp, end_timestamp));
        total_intervals_++;
        if(start_timestamp > end_timestamp){
            printf("wrong interval %d %d %d\n", start_timestamp, end_timestamp);
            while(1);
        }
        if(earliest_time_ == -1) {earliest_time_ = start_timestamp; latest_time_ = end_timestamp;}
        else {
            earliest_time_ = std::min(earliest_time_, start_timestamp);
            latest_time_ = std::max(latest_time_, end_timestamp);
        }
    }
    inFile.close();

#else
    FILE* fin = fopen(input_file.c_str(), "r");
    while(fscanf(fin, "%d %d", &start_timestamp, &end_timestamp) != EOF) {
        if (start_timestamp > end_timestamp)
        {
            cerr << endl << "Error - start is after end for interval [" << start_timestamp << ".." << end_timestamp << "]" << endl << endl;
            exit(1);
        }
        T.push_back(TInterval(total_intervals_, start_timestamp, end_timestamp));
        total_intervals_++;
        if(earliest_time_ == -1) {earliest_time_ = start_timestamp; latest_time_ = end_timestamp;}
        else {
            earliest_time_ = std::min(earliest_time_, start_timestamp);
            latest_time_ = std::max(latest_time_, end_timestamp);
        }
    }
    fclose(fin);
#endif
    double t_begin, t_stop, diff_in_second, diff_in_microsecond;
    t_begin = GetTime();
    std::sort(T.begin(), T.end());

    T_unique_.clear();
    T_unique_.push_back(TInterval(0, T[0].l, T[0].r));
    for (size_t i = 1; i < T.size(); i++){
        if (T[i].l != T[i-1].l || T[i].r != T[i-1].r){
            T_unique_.push_back(TInterval(i, T[i].l, T[i].r));
        }
    }
    unique_intervals_ = T_unique_.size();

    T_id_.clear();
    for (size_t i = 0; i < T.size(); i++){
        T_id_.push_back(T[i].id);
    }
    vector<TInterval>().swap(T);
    T.shrink_to_fit();

    cout << "T_unique_.size(): " << T_unique_.size()  << " " << T_id_.size() << endl;

    my_list.list_location.resize(T_unique_.size());

    
    vector<RecordId> sorted_by_start_(T_unique_.size());
    iota(sorted_by_start_.begin(), sorted_by_start_.end(), 0);
    sort(sorted_by_start_.begin(), sorted_by_start_.end(), [&](RecordId x, RecordId y) {
        if (T_unique_[x].l == T_unique_[y].l) return T_unique_[x].r < T_unique_[y].r;
        return T_unique_[x].l < T_unique_[y].l;
    });
    vector<RecordId> sorted_by_end(T_unique_.size());
    iota(sorted_by_end.begin(), sorted_by_end.end(), 0);
    sort(sorted_by_end.begin(), sorted_by_end.end(), [&](RecordId x, RecordId y) {
        if (T_unique_[x].r == T_unique_[y].r) return T_unique_[x].l < T_unique_[y].l;
        return T_unique_[x].r < T_unique_[y].r;
    });

    if(query_type == CONTAINS_QUERY){
        build_index(sorted_by_start_, sorted_by_end);
    }
    else{
        build_index_contained_overlaps(sorted_by_start_, sorted_by_end);
    }
    t_stop = GetTime();

    vector<RecordId>().swap(sorted_by_start_);
    vector<RecordId>().swap(sorted_by_end);

    std::cout << "#intervals: " << total_intervals_  << " earliest_time " << earliest_time_ << " latest_time: " << latest_time_ << " time range: " << latest_time_ -  earliest_time_ << std::endl;


    std::cout << "Index Construction time: " << t_stop - t_begin << " seconds" << std::endl;

    // compute index size of next
    long long edge_cnt = 0, edge_cnt_super = 0;
    double index_size = 0;
    index_size += total_intervals_ * sizeof(RecordId);
    index_size += T_unique_.size() * (sizeof(RecordId)+sizeof(int)*2);
    std::cout << "Interval Vector size: " << index_size / 1024 / 1024 << " MB" << endl;

    index_size += unique_intervals_ * sizeof(RecordId);
    for(size_t i = 0; i < next.size(); i++){
        index_size += next[i].size() * sizeof(RecordId)*2;
        edge_cnt += next[i].size();
    }
    if(query_type == CONTAINS_QUERY){
        std::cout << "And Contains Query Index size: ";
    }
    else {
        std::cout << "Contained Overlaps Query Index size: ";
    }
    std::cout << index_size / 1024 / 1024 << " MB" << endl;
    cout << "Edge count: " << edge_cnt <<  " Average degree: " << (double)edge_cnt / unique_intervals_ << endl;

}

 void TemGraph::build_index(vector< RecordId> &a, vector< RecordId> &b){
    for (RecordId i = b.size() - 1; i >= 0; i--){
        my_list.insert(b[i]);
        if(i == 0) break;
    }
    b.clear();
    b.shrink_to_fit();

    RecordId x;
    vector<OutNeighbor > tmp;
    tmp.clear();
    x = my_list.a[my_list.r[0]];
    tmp.push_back(OutNeighbor(T_unique_[x].l, my_list.r[0], 0));
    vector<pair<RecordId, RecordId> >  tmp_in;
    tmp_in.clear();
    x = my_list.a[my_list.l[0]];
    tmp_in.push_back(make_pair(T_unique_[x].l, my_list.l[0]));
    next.resize(my_list.n+1);
    in_neighbors.resize(my_list.n+1);
    next[0] = tmp;
    in_neighbors[0] = tmp_in;

    for (RecordId i = my_list.r[0]; i != 0; i = my_list.r[i]){
        x = my_list.a[my_list.r[i]];
        tmp[0].l = T_unique_[x].l;
        tmp[0].x = my_list.r[i];
        tmp[0].successor = 0;
        next[i] = tmp;
        tmp_in[0].first = i;
        tmp_in[0].second = 0;
        in_neighbors[my_list.r[i]] = tmp_in;
    }

    List _my_list = my_list;

    for (RecordId i = 0; i < T_unique_.size(); i++){
        my_list.erase(a[i]);
        x = my_list.list_location[a[i]];
        RecordId l_x = my_list.l[x], r_x = my_list.r[x];
        in_neighbors[r_x].push_back(make_pair(l_x, next[l_x].size()));
        next[l_x].push_back(OutNeighbor(T_unique_[my_list.a[r_x]].l, r_x, 0));
        
    }
    my_list = _my_list;

    // add successor
    for(size_t i = 1; i < next.size(); ++i) {
        int pin = 0, pout = 0;
        while(pin < in_neighbors[i].size() && pout < next[i].size()){
            auto in_edge = in_neighbors[i][pin];
            auto out_edge = next[i][pout];
            Timestamp min_l = min(T_unique_[my_list.a[in_edge.first]].l , T_unique_[my_list.a[i]].l);
            if(min_l > out_edge.l){
                pout++;
            }
            else {
                next[in_edge.first][in_edge.second].successor = pout;
                pin++;
            }
        }
        while(pin < in_neighbors[i].size()){
            auto in_edge = in_neighbors[i][pin];
            next[in_edge.first][in_edge.second].successor = next[i].size()-1;
            pin++;
        }
    }
 }



void TemGraph::build_index_contained_overlaps(vector< RecordId> &a, vector< RecordId> &b){
    for (RecordId i = b.size() - 1; i >= 0; i--){
        my_list.insert(b[i]);
        if(i == 0) break;
    }
    RecordId x;
    vector<OutNeighbor > tmp;
    tmp.clear();
    x = my_list.a[my_list.l[0]];
    tmp.push_back(OutNeighbor(T_unique_[x].l, my_list.l[0], 0));
    vector<pair<RecordId, RecordId> >  tmp_in;
    tmp_in.clear();
    x = my_list.a[my_list.r[0]];
    tmp_in.push_back(make_pair(T_unique_[x].l, my_list.r[0]));

    next.resize(my_list.n + 1);
    in_neighbors.resize(my_list.n + 1);
    next[0] = tmp;
    in_neighbors[0] = tmp_in;

    for (RecordId i = my_list.l[0]; i != 0; i = my_list.l[i]){
        x = my_list.a[my_list.l[i]];
        tmp[0].l = T_unique_[x].l;
        tmp[0].x = my_list.l[i];
        tmp[0].successor = 0;
        next[i] = tmp;
        tmp_in[0].first = i;
        tmp_in[0].second = 0;
        in_neighbors[my_list.r[i]] = tmp_in;
    }

    List _my_list = my_list;

    for (RecordId i = a.size() - 1; i >= 0; i--){
        my_list.erase(a[i]);
        x = my_list.list_location[a[i]];
        RecordId l_x = my_list.l[x], r_x = my_list.r[x];

        in_neighbors[l_x].push_back(make_pair(r_x, next[r_x].size()));
        next[r_x].push_back(OutNeighbor(T_unique_[my_list.a[l_x]].l, l_x, 0));
        
        if(i == 0) break;
    }
    my_list = _my_list;

    // add successor
    for(RecordId i = 1; i < next.size(); ++i) {
        for(RecordId j = 0; j < next[i].size();++j){
            RecordId p = next[i][j].x;
            RecordId k = 0;
            Timestamp max_l = max(next[i][j].l, T_unique_[my_list.a[i]].l);
            for(k = 0; k < next[p].size()-1; ++k){
                if(next[p][k].l <= max_l) break;
            }
            next[i][j].successor = k;
        }
    }

    
}




int TemGraph::contains_query(Timestamp l, Timestamp r){
    visited_intervals_ = 0;
    RecordId i = 0;
    RecordId last_tell_loc, next_loc;

    vector<RecordId *> res;
    res.clear();
    RecordId lef = 0, rig = next[i].size() - 1, mid = lef;

    while (lef < rig){
        visited_intervals_++;
        mid = (lef + rig) / 2;
        if (next[i][mid].l >= l)
            rig = mid;
        else lef = mid + 1;
    }
    mid = lef;
    i = next[i][mid].x;
    if (T_unique_[my_list.a[i]].r > r || T_unique_[my_list.a[i]].l < l)
        return res.size();


    RecordId all_n = T_id_.size(), next_x = all_n;
    if(my_list.a[i] != T_unique_.size()-1) next_x = T_unique_[my_list.a[i]+1].id;
    else next_x = all_n;
    for(size_t k = T_unique_[my_list.a[i]].id; k < next_x; k++){
        res.push_back(&T_id_[k]);
        visited_intervals_++;
    }

    // find the next valid Result X1 from X.index by O(log)
    lef = 0, rig = next[i].size() - 1, mid = lef;
    while (lef < rig){
        visited_intervals_++;
        mid = (lef + rig) / 2;
        if (next[i][mid].l > l)
            rig = mid;
        else lef = mid + 1;
    }
    mid = lef;
    last_tell_loc = next[i][mid].successor;
    i = next[i][mid].x;



    do {
        //----next-----

        if (T_unique_[my_list.a[i]].r > r || T_unique_[my_list.a[i]].l < l)  break;

            
        next_x = all_n;
        if(my_list.a[i] != T_unique_.size()-1) next_x = T_unique_[my_list.a[i]+1].id;
        else next_x = all_n;
        for(size_t k = T_unique_[my_list.a[i]].id; k < next_x; k++){
            res.push_back(&T_id_[k]);
            visited_intervals_++;
        }
        if(last_tell_loc == 0) visited_intervals_++; 
        while(last_tell_loc > 0 && next[i][last_tell_loc-1].l >= l){
            visited_intervals_++;
            last_tell_loc--;
        }

        next_loc = next[i][last_tell_loc].successor;
        i = next[i][last_tell_loc].x;
        last_tell_loc = next_loc;

    }while (i != 0);

    return res.size();
}

int TemGraph::contained_query(Timestamp l, Timestamp r){
    visited_intervals_ = 0;
    RecordId i = 0, last_tell_loc, next_loc;;
    vector<RecordId *> res;
    RecordId all_n = T_id_.size(), next_x = all_n;
    res.clear();
    // find first valid result X
    RecordId lef = 0, rig = next[i].size() - 1, mid = lef;
    while (lef < rig){
        visited_intervals_++;
        mid = (lef + rig) / 2;
        if (next[i][mid].l <= l)
            rig = mid;
        else lef = mid + 1;
    }
    mid = lef;
    i = next[i][mid].x;
    if (T_unique_[my_list.a[i]].r < r || T_unique_[my_list.a[i]].l > l)
        return res.size();

    if(my_list.a[i] != T_unique_.size()-1) next_x = T_unique_[my_list.a[i]+1].id;
    else next_x = all_n;
    for(size_t k = T_unique_[my_list.a[i]].id; k < next_x; k++){
        res.push_back(&T_id_[k]);
        visited_intervals_++;
    }

    // find the next valid Result X1 from X.index by O(log)
    lef = 0, rig = next[i].size() - 1, mid = lef;
    while (lef < rig){
        visited_intervals_++;
        mid = (lef + rig) / 2;
        if (next[i][mid].l < l)
            rig = mid;
        else lef = mid + 1;
    }
    mid = lef;
    last_tell_loc = next[i][mid].successor;
    i = next[i][mid].x;

    // find the valid results by successor
    do {
        if (T_unique_[my_list.a[i]].r < r || T_unique_[my_list.a[i]].l > l)
            break;
            
        if(my_list.a[i] != T_unique_.size()-1) next_x = T_unique_[my_list.a[i]+1].id;
        else next_x = all_n;
        for(size_t k = T_unique_[my_list.a[i]].id; k < next_x; k++){
            res.push_back(&T_id_[k]);
            visited_intervals_++;
        }

        if(last_tell_loc == 0) visited_intervals_++; 
        while(last_tell_loc > 0 && next[i][last_tell_loc-1].l <= l){
            visited_intervals_++;
            last_tell_loc--;
        }
        next_loc = next[i][last_tell_loc].successor;   
        i = next[i][last_tell_loc].x;
        last_tell_loc = next_loc;
    }while (i != 0);

    return res.size();
}
