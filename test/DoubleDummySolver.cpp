// g++ -shared -o DoubleDummySolver.dll DoubleDummySolver.cpp -L<path> -ldds

#include <cstring>
#include "dll.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>

const int UNDER_TRICKS_CHART[3][2][13] = {{{50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650},
                                           {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300}},
                                          {{100, 300, 500, 800, 1100, 1400, 1700, 2000, 2300, 2600, 2900, 3200, 3500},
                                           {200, 500, 800, 1100, 1400, 1700, 2000, 2300, 2600, 2900, 3200, 3500, 3800}},
                                          {{200, 600, 1000, 1600, 2200, 2800, 3400, 4000, 4600, 5200, 5800, 6400, 7000},
                                           {400, 1000, 1600, 2200, 2800, 3400, 4000, 4600, 5200, 5800, 6400, 7000, 7600}}};

const int CONTRACT_VAL[3][2][5][7] = {{{{70, 90, 110, 130, 400, 920, 1440}, {70, 90, 110, 130, 400, 920, 1440}, {80, 110, 140, 420, 450, 980, 1510}, {80, 110, 140, 420, 450, 980, 1510}, {90, 120, 400, 430, 460, 990, 1520}},
                                       {{70, 90, 110, 130, 600, 1370, 2140}, {70, 90, 110, 130, 600, 1370, 2140}, {80, 110, 140, 620, 650, 1430, 2210}, {80, 110, 140, 620, 650, 1430, 2210}, {90, 120, 600, 630, 660, 1440, 2220}}},
                                      {{{140, 180, 470, 510, 550, 1090, 1630}, {140, 180, 470, 510, 550, 1090, 1630}, {160, 470, 530, 590, 650, 1210, 1770}, {160, 470, 530, 590, 650, 1210, 1770}, {180, 490, 550, 610, 670, 1230, 1790}},
                                       {{140, 180, 670, 710, 750, 1540, 2330}, {140, 180, 670, 710, 750, 1540, 2330}, {160, 670, 730, 790, 850, 1660, 2470}, {160, 670, 730, 790, 850, 1660, 2470}, {180, 690, 750, 810, 870, 1680, 2490}}},
                                      {{{230, 560, 640, 720, 800, 1380, 1960}, {230, 560, 640, 720, 800, 1380, 1960}, {520, 640, 760, 880, 1000, 1620, 2240}, {520, 640, 760, 880, 1000, 1620, 2240}, {560, 680, 800, 920, 1040, 1660, 2280}},
                                       {{230, 760, 840, 920, 1000, 1830, 2660}, {230, 760, 840, 920, 1000, 1830, 2660}, {720, 840, 960, 1080, 1200, 2070, 2940}, {720, 840, 960, 1080, 1200, 2070, 2940}, {760, 880, 1000, 1120, 1240, 2110, 2980}}}};
const int OVERTRICK_VAL[3][2][5] = {{{20, 20, 30, 30, 30}, {20, 20, 30, 30, 30}},
                                    {{100, 100, 100, 100, 100}, {200, 200, 200, 200, 200}},
                                    {{200, 200, 200, 200, 200}, {400, 400, 400, 400, 400}}};
const int IMP_CHART[401] = {0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9,
                            9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13,
                            13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
                            15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
                            17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
                            19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                            20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                            21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
                            22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
                            23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24};

struct ddResponse
{
    int imp_loss;   //
    int error_type; // calc*1000 + par
};

extern "C" __attribute__((dllexport)) ddResponse ddAnalize(char *deal, int vul[2], int suit, int level, int doubled, int dealer)
{
    SetMaxThreads(0);
    SetResources(0, 0);
    ddResponse res;
    res.error_type = 0;
    ddTableDealPBN _deal;

    strcpy(_deal.cards, deal);
    ddTableResults _result;
    int error_type = CalcDDtablePBN(_deal, &_result);
    if (error_type != 1)
    {
        res.error_type = error_type * 1000;
        return res;
    }

    parResults _presp;
    int encoded_vul;
    if (vul[0] == 0)
    {
        if (vul[1] == 0)
        {
            encoded_vul = 0;
        }
        else
        {
            encoded_vul = 3;
        }
    }
    else
    {
        if (vul[1] == 0)
        {
            encoded_vul = 2;
        }
        else
        {
            encoded_vul = 1;
        }
    }
    error_type = Par(&_result, &_presp, encoded_vul);
    if (error_type != 1)
    {
        res.error_type = error_type;
        return res;
    }
    /*printf("NS : %s\n", _presp.parContractsString[0]);
    printf("EW : %s\n", _presp.parContractsString[1]);*/
    int best_score;
    int sign = 1;
    int number = 0;
    int found_number = 0;
    char *ptr = _presp.parScore[0];

    while (*ptr)
    {
        if (*ptr == '-')
        {
            sign = -1;
        }
        else if (isdigit(*ptr))
        {
            found_number = 1;
            number = number * 10 + (*ptr - '0');
        }
        else if (found_number)
        {
            break;
        }
        ptr++;
    }
    best_score = number * sign;
    if (dealer % 2 == 1)
    {
        best_score *= -1
    }

    // printf("best_score[0]: %d, best_score[1]: %d\n", best_score[0], best_score[1]);
    /*for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            printf("dealer: %d, suit: %d, trick: %d\n", i, j, _result.resTable[j][i]);
        }
    }*/
    int score;
    int my_encoding = 4;
    if (suit == -1)
    {
        score = 0
    }
    else
    {
        if (suit < 4)
        {
            my_encoding = 3 - suit;
        }
        if (level + 7 <= _result.resTable[suit][dealer])
        {
            int over_trick = _result.resTable[suit][dealer] - (level + 7);
            score = CONTRACT_VAL[doubled][vul[dealer % 2]][my_encoding][level] + OVERTRICK_VAL[doubled][vul[dealer % 2]][my_encoding] * over_trick;
        }
        else
        {
            int down = (level + 7) - _result.resTable[suit][dealer] - 1;
            score = -UNDER_TRICKS_CHART[doubled][vul[dealer % 2]][down];
            // test[doubled][dealer][suit][level] = down;
        }
    }
    // test[doubled][dealer][my_encoding][level] = score;
    int diff = (score - best_score) / 10;
    int sign = 1;
    if (diff < 0)
    {
        sign = -1;
        diff = -diff;
    }
    if (diff > 400)
        diff = 400;
    res.imp_loss = sign * IMP_CHART[diff];

    /*for (int dealer = 0; dealer < 4; dealer++)
    {
        printf("dealer: %d\n", dealer);
        for (int suit = 0; suit < 5; suit++)
        {
            printf("suit-%d: ", suit);
            for (int level = 0; level < 7; level++)
            {

                printf("level %d: %d, %d; ", level, test[0][dealer][suit][level], res.imp_loss[0][dealer][suit][level]);
            }
            printf("\n");
        }
    }*/

    FreeMemory();
    return res;
}