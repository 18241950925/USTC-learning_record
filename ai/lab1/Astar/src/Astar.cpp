#include<vector>
#include<iostream>
#include<queue>
#include<map>
#include<fstream>
#include<sstream>
#include<string>

using namespace std;

struct Map_Cell
{
    int type;
    // TODO: 定义地图信息

};

struct Search_Cell
{
    int h;
    int g;
    // TODO: 定义搜索状态
    Search_Cell* parent;
    int food;
    pair<int, int> pos;
};

// 自定义比较函数对象，按照 Search_Cell 结构体的 g + h 属性进行比较
struct CompareF {
    bool operator()(const Search_Cell *a, const Search_Cell *b) const {
        return (a->g + a->h) > (b->g + b->h); // 较小的 g + h 值优先级更高
    }
};

// TODO: 定义启发式函数
int Heuristic_Funtion(pair<int, int> cur_point,pair<int, int> end_point)
{   
    //return 0;
    return abs(cur_point.first - end_point.first)+abs( cur_point.second - end_point.second); 
}

Search_Cell* inqueue(Search_Cell* cell, vector<Search_Cell*>* queue, int T)
{
    for(auto i: *queue)
    {
        if(i->pos == cell->pos && (i->food == cell->food || i->food == T))
        {
            return i;
        }
    }
    return nullptr;
}

Search_Cell* inqueue(Search_Cell* cell, priority_queue<Search_Cell *, vector<Search_Cell *>, CompareF>* queue,int T)
{
    Search_Cell* cell_temp=nullptr;
    priority_queue<Search_Cell *, vector<Search_Cell *>, CompareF> temp;
    while(!queue->empty())
    {
        auto top = queue->top();
        queue->pop();
        if(top->pos == cell->pos && (top->food == cell->food || top->food == T))
        {
            cell_temp = top;
        }
        temp.push(top);
    }
    while(!temp.empty())
    {
        auto top = temp.top();
        temp.pop();
        queue->push(top);
    }
    return cell_temp;
}

void Astar_search(const string input_file, int &step_nums, string &way)
{
    ifstream file(input_file);
    if (!file.is_open()) {
        cout << "Error opening file!" << endl;
        return;
    }

    string line;
    getline(file, line); // 读取第一行
    stringstream ss(line);
    string word;
    vector<string> words;
    while (ss >> word) {
        words.push_back(word);
    }
    int M = stoi(words[0]);
    int N = stoi(words[1]);
    int T = stoi(words[2]);

    pair<int, int> start_point; // 起点
    pair<int, int> end_point;   // 终点
    Map_Cell **Map = new Map_Cell *[M];
    // 加载地图
    for(int i = 0; i < M; i++)
    {
        Map[i] = new Map_Cell[N];
        getline(file, line);
        stringstream ss(line);
        string word;
        vector<string> words;
        while (ss >> word) {
            words.push_back(word);
        }
        for(int j = 0; j < N; j++)
        {
            Map[i][j].type = stoi(words[j]);
            if(Map[i][j].type == 3)
            {
                start_point = {i, j};
            }
            else if(Map[i][j].type == 4)
            {
                end_point = {i, j};
            }
        }
    }
    // 以上为预处理部分
    // ------------------------------------------------------------------

    Search_Cell *search_cell = new Search_Cell;
    search_cell->g = 0;
    search_cell->h = Heuristic_Funtion(start_point,end_point); // Heuristic_Funtion();
    search_cell->food = T;
    search_cell->pos = start_point;
    search_cell->parent = nullptr;
    priority_queue<Search_Cell *, vector<Search_Cell *>, CompareF> open_list;
    vector<Search_Cell *> close_list;
    open_list.push(search_cell);
    //pair<int,int> cur_point = start_point;
    Search_Cell* current_node;

    step_nums = -1;
    way = "";

    while(!open_list.empty())
    {
        // TODO: A*搜索过程实现
        current_node = open_list.top();  // 从open_list中取出f值最小的节点作为当前节点
        open_list.pop();
        if (current_node->pos == end_point)
        {
            step_nums = current_node->g;
            break;
        }
        else if(current_node->food == 0)
        {
            continue;
        }
        else
        {
            vector<pair<int, int>> neighbor = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
            for (pair<int, int> p : neighbor) 
            {
                int i = p.first;
                int j = p.second;
                auto cur_point = current_node->pos;
                if(cur_point.first+i>=0 && cur_point.first+i<M && cur_point.second+j>=0 && cur_point.second+j<N)
                {
                    if(Map[cur_point.first+i][cur_point.second+j].type != 1)
                    {
                        Search_Cell* new_node = new Search_Cell;
                        new_node->g = current_node->g + 1;
                        new_node->h = Heuristic_Funtion({cur_point.first+i,cur_point.second+j},end_point);
                        new_node->parent = current_node;
                        if(Map[cur_point.first+i][cur_point.second+j].type==2)
                            new_node->food = T;
                        else
                            new_node->food = current_node->food - 1;
                        new_node->pos = {cur_point.first+i,cur_point.second+j};
                        if(inqueue(new_node,&close_list,T)==nullptr and inqueue(new_node,&open_list,T) == nullptr)
                        {
                            open_list.push(new_node);
                        } 
                        else
                        {
                            if(auto old_cell = inqueue(new_node,&close_list,T))
                            {
                                if(old_cell->g > new_node->g)
                                {
                                    old_cell->g = new_node->g;
                                    old_cell->food = new_node->food;
                                    old_cell->parent = new_node->parent;
                                    for(int i = 0; i < close_list.size(); i++)
                                    {
                                        if(close_list[i]->pos == old_cell->pos)
                                        {
                                            delete close_list[i];
                                            close_list.erase(close_list.begin()+i);
                                            break;
                                        }
                                    }
                                }
                            }
                            else if(auto old_cell = inqueue(new_node,&open_list,T))
                            {
                                if(old_cell->g > new_node->g)
                                {
                                    old_cell->g = new_node->g;
                                    old_cell->food = new_node->food;
                                    old_cell->parent = new_node->parent;
                                }
                            }
                            delete new_node;
                        }
                        
                    }
                    
                }
            }  
        }
        cout << current_node->pos.first << " " << current_node->pos.second << endl;
        close_list.push_back(current_node);
    }

    // ------------------------------------------------------------------
    // TODO: 填充step_nums与way
    if(step_nums>=0)
    {
        Search_Cell* temp = current_node;
        while(temp->parent != nullptr)
        {
            if(temp->pos.first - temp->parent->pos.first == 1)
            {
                way = "D" + way;
            }
            else if(temp->pos.first - temp->parent->pos.first == -1)
            {
                way = "U" + way;
            }
            else if(temp->pos.second - temp->parent->pos.second == 1)
            {
                way = "R" + way;
            }
            else if(temp->pos.second - temp->parent->pos.second == -1)
            {
                way = "L" + way;
            }
            temp = temp->parent;
        }
    }
    // ------------------------------------------------------------------
    // 释放动态内存
    for(int i = 0; i < M; i++)
    {
        delete[] Map[i];
    }
    delete[] Map;
    while(!open_list.empty())
    {
        auto temp = open_list.top();
        delete[] temp;
        open_list.pop();
    }
    for(int i = 0; i < close_list.size(); i++)
    {
        delete[] close_list[i];
    }

    return;
}

void output(const string output_file, int &step_nums, string &way)
{
    ofstream file(output_file);
    if(file.is_open())
    {
        file << step_nums << endl;
        if(step_nums >= 0)
        {
            file << way << endl;
        }

        file.close();
    }
    else
    {
        cerr << "Can not open file: " << output_file << endl;
    }
    return;
}

int main(int argc, char *argv[])
{
    string input_base = "../input/input_";
    string output_base = "../output/output_";
    // input_0为讲义样例，此处不做测试
    for(int i = 1; i < 11; i++)
    {
        int step_nums = -1;
        string way = "";
        Astar_search(input_base + to_string(i) + ".txt", step_nums, way);
        output(output_base + to_string(i) + ".txt", step_nums, way);
    }
    return 0;
}