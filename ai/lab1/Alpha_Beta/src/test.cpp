#include <fstream>
#include "node.h"

using namespace ChineseChess;

//博弈树搜索，depth为搜索深度
int alphaBeta(GameTreeNode* node, int alpha, int beta, int depth, bool isMaximizer) {
    if (depth == 0) {
        return node->getEvaluationScore();
    }
    int i=0;
    //TODO alpha-beta剪枝过程
    if (isMaximizer) {
        int maxEval = std::numeric_limits<int>::min();
        for (GameTreeNode* child : node->getChildren())
        {
            int eval = alphaBeta(child, alpha, beta, depth - 1, false);
            maxEval = std::max(maxEval, eval);
            if(maxEval > alpha) {
                node->best_move = node->getBoardClass().getMoves(true)[i];
                node->name = node->getBoardClass().getBoard()[node->best_move.init_y][node->best_move.init_x];
                //std::cout << node->name << node->best_move.init_x<< node->best_move.init_y<< node->best_move.next_x << node->best_move.next_y<< std::endl;
            }
            alpha = std::max(alpha, eval);
            if (beta <= alpha) { 
                break;
            }
            i++;
        }
        return maxEval;
    } else {
        int minEval = std::numeric_limits<int>::max();
        for (GameTreeNode* child : node->getChildren())
        {
            int eval = alphaBeta(child, alpha, beta, depth - 1, true);
            minEval = std::min(minEval, eval);
            beta = std::min(beta, eval);
            if (beta <= alpha) {
                break;
            }
            i++;
        } 
        return minEval;
    }
}

int main() {
    std::vector<int> filesub = {1,2,3,4,5,6,7,8,9,10};
    for(auto sub:filesub) {
        std::ifstream file("../input/" + std::to_string(sub) + ".txt");
        std::vector<std::vector<char>> board;
        std::string line;
        int n = 0;
        while (std::getline(file, line)) {
            std::vector<char> row;

            for (char ch : line) {
                row.push_back(ch);
            }

            board.push_back(row);
            n = n + 1;
            if (n >= 10) break;
        }
        file.close();
        GameTreeNode root(true, board, std::numeric_limits<int>::min(),0);
        //TODO
        alphaBeta(&root, std::numeric_limits<int>::min(), std::numeric_limits<int>::max(), MAXDEPTH, true);

        std::ofstream output_file("../output/output_"+ std::to_string(sub) + ".txt");
        Move move = root.best_move;
        output_file <<  root.name << "(" << move.init_x << "," << 9-move.init_y << ") (" << move.next_x << "," << 9-move.next_y << ")" << std::endl;

        output_file.close();

    }
    

    //代码测试
    // ChessBoard _board = root.getBoardClass();
    // std::vector<std::vector<char>> cur_board = _board.getBoard();

    // for (int i = 0; i < cur_board.size(); i++) {
    //     for (int j = 0; j < cur_board[0].size(); j++) {
    //         std::cout << cur_board[i][j];
    //     }
    //     std::cout << std::endl;
    // }

    // std::vector<Move> red_moves = _board.getMoves(true);
    // std::vector<Move> black_moves = _board.getMoves(false);

    // for (int i = 0; i < red_moves.size(); i++) {
    //     std::cout << "init: " << red_moves[i].init_x <<  " " << red_moves[i].init_y << std::endl;
    //     std::cout << "next: " << red_moves[i].next_x <<  " " << red_moves[i].next_y << std::endl;
    //     std::cout << "score " << red_moves[i].score << std::endl;
    // }
    // for (int i = 0; i < black_moves.size(); i++) {
    //     std::cout << "init: " << black_moves[i].init_x <<  " " << black_moves[i].init_y << std::endl;
    //     std::cout << "next: " << black_moves[i].next_x <<  " " << black_moves[i].next_y << std::endl;
    //     std::cout << "score " << black_moves[i].score << std::endl;
    // }

    return 0;
}