
#ifndef DOUBLY_LINKED_LIST_H
#define DOUBLY_LINKED_LIST_H

#include <iostream>

class List{
public:
    vector<RecordId> list_location;
    vector<RecordId> a, l, r;        // l is the neighbor with smaller end, r is the neighbor with larger end
    RecordId o;
    RecordId n;
    List(){ 
        a.clear();
        l.clear();
        r.clear();
        a.push_back(0);
        l.push_back(0);     
        r.push_back(0);     
        n = 0;
    }
    ~List(){
        a.erase(a.begin(), a.end());
        l.erase(l.begin(), l.end());
        r.erase(r.begin(), r.end());
        list_location.erase(list_location.begin(), list_location.end());
    }
    void clear(){
        a.erase(a.begin(), a.end());
        l.erase(l.begin(), l.end());
        r.erase(r.begin(), r.end());
        list_location.erase(list_location.begin(), list_location.end());
        a.push_back(0);
        l.push_back(0);
        r.push_back(0);
        n = 0;
    }
    RecordId cal_num(){
        RecordId res = 0;
        for (RecordId i = r[0]; i != 0; i = r[i])
            res++;
        return res;
    }
    void insert(RecordId x){
        list_location[x] = a.size();
        l.push_back(0);
        r.push_back(r[0]);
        l[r[0]] = a.size();
        r[0] = a.size();
        a.push_back(x);
        n++;
    }
    void insert_back(RecordId x){        // when x has the largest end time
        list_location[x] = a.size();   
        
        l.push_back(l[0]);
        r.push_back(0);
        r[l[0]] = a.size();
        l[0] = a.size();        
        a.push_back(x);    
        r[0] = a.size();            
        n++;
    }

    void delete_front(RecordId x){    // when x has the smallest end time
        RecordId _x = list_location[x];

        r[0] = r[l[0]];
        l[r[0]] = _x; 
        a[_x] = -1;
        list_location[x] = -1;
        n--;

    }
    void recover(RecordId x){
        RecordId _x = x;
        x = list_location[x];
        l[r[x]] = x;
        r[l[x]] = x;
        n++;
    }
    void erase(RecordId x){
        RecordId _x = x;
        x = list_location[x];
        r[l[x]] = r[x];
        l[r[x]] = l[x];
        n--;
    }

};



#endif //DOUBLY_LINKED_LIST_H
