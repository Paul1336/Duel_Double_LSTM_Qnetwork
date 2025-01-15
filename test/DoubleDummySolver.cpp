// g++ -shared -o DoubleDummySolver.dll DoubleDummySolver.cpp -L<path> -ldds

#include <cstring>
#include "dll.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <iostream>

const int UNDER_TRICKS_CHART[3][2][13] = {{{50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650},
                                           {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300}},
                                          {{100, 300, 500, 800, 1100, 1400, 1700, 2000, 2300, 2600, 2900, 3200, 3500},
                                           {200, 500, 800, 1100, 1400, 1700, 2000, 2300, 2600, 2900, 3200, 3500, 3800}},
                                          {{200, 600, 1000, 1600, 2200, 2800, 3400, 4000, 4600, 5200, 5800, 6400, 7000},
                                           {400, 1000, 1600, 2200, 2800, 3400, 4000, 4600, 5200, 5800, 6400, 7000, 7600}}};
// [double(0-p, 1-d, 2-r)][vul(0-non, 1-vul)][n_trick]

const int CONTRACT_VAL[3][2][5][7] = {{{{70, 90, 110, 130, 400, 920, 1440}, {70, 90, 110, 130, 400, 920, 1440}, {80, 110, 140, 420, 450, 980, 1510}, {80, 110, 140, 420, 450, 980, 1510}, {90, 120, 400, 430, 460, 990, 1520}},
                                       {{70, 90, 110, 130, 600, 1370, 2140}, {70, 90, 110, 130, 600, 1370, 2140}, {80, 110, 140, 620, 650, 1430, 2210}, {80, 110, 140, 620, 650, 1430, 2210}, {90, 120, 600, 630, 660, 1440, 2220}}},
                                      {{{140, 180, 470, 510, 550, 1090, 1630}, {140, 180, 470, 510, 550, 1090, 1630}, {160, 470, 530, 590, 650, 1210, 1770}, {160, 470, 530, 590, 650, 1210, 1770}, {180, 490, 550, 610, 670, 1230, 1790}},
                                       {{140, 180, 670, 710, 750, 1540, 2330}, {140, 180, 670, 710, 750, 1540, 2330}, {160, 670, 730, 790, 850, 1660, 2470}, {160, 670, 730, 790, 850, 1660, 2470}, {180, 690, 750, 810, 870, 1680, 2490}}},
                                      {{{230, 560, 640, 720, 800, 1380, 1960}, {230, 560, 640, 720, 800, 1380, 1960}, {520, 640, 760, 880, 1000, 1620, 2240}, {520, 640, 760, 880, 1000, 1620, 2240}, {560, 680, 800, 920, 1040, 1660, 2280}},
                                       {{230, 760, 840, 920, 1000, 1830, 2660}, {230, 760, 840, 920, 1000, 1830, 2660}, {720, 840, 960, 1080, 1200, 2070, 2940}, {720, 840, 960, 1080, 1200, 2070, 2940}, {760, 880, 1000, 1120, 1240, 2110, 2980}}}};
// [double(0-p, 1-d, 2-r)][vul(0-non, 1-vul)][suit(c, d, h, s, n)][level]

const int OVERTRICK_VAL[3][2][5] = {{{20, 20, 30, 30, 30}, {20, 20, 30, 30, 30}},
                                    {{100, 100, 100, 100, 100}, {200, 200, 200, 200, 200}},
                                    {{200, 200, 200, 200, 200}, {400, 400, 400, 400, 400}}};
// [double(0-p, 1-d, 2-r)][vul(0-non, 1-vul)][n_trick]

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
// [score(0.1)-imp]

struct ddResponse
{
    int imp_loss;
    int error_type_calc;
    int error_type_par;
};

ddResponse test(char *deal, int vul[2], int suit, int level, int doubled, int dealer, int view)
{
    SetMaxThreads(0);
    SetResources(0, 0);
    ddResponse res;
    ddTableDealPBN _deal;
    strcpy(_deal.cards, deal);
    printf("_deal : %s\n", _deal.cards);
    ddTableResults _result;
    res.error_type_calc = CalcDDtablePBN(_deal, &_result);
    if (!res.error_type_calc)
    {
        return res;
    }
    std::cout << "ddTableResults:" << std::endl;
    for (int strain = 0; strain < DDS_STRAINS; ++strain)
    {
        std::cout << "Strain " << strain << ": ";
        for (int hand = 0; hand < DDS_HANDS; ++hand)
        {
            std::cout << _result.resTable[strain][hand] << " ";
        }
        std::cout << std::endl;
    }
    parResults _presp;
    int decoded_vul;
    if (vul[0] == 0)
    {
        if (vul[1] == 0)
        {
            decoded_vul = 0;
        }
        else
        {
            decoded_vul = 3;
        }
    }
    else
    {
        if (vul[1] == 0)
        {
            decoded_vul = 2;
        }
        else
        {
            decoded_vul = 1;
        }
    }
    res.error_type_par = Par(&_result, &_presp, decoded_vul);
    if (!res.error_type_par)
    {
        return res;
    }
    int best_score_NS;
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
    best_score_NS = number * sign;
    printf("best_score_NS: %d\n", best_score_NS);
    printf("NS : %s\n", _presp.parContractsString[0]);
    int cur_score = 0;
    if (suit == -1 && level == -1 && doubled == -1 && dealer == -1)
    {
        cur_score = 0;
    }
    else
    {
        if (suit == -1 || level == -1 || doubled == -1 || dealer == -1)
        {
            res.error_type_par = 0;
            return res;
        }
        else
        {
            int decoded_suit = 4;
            if (suit < 4)
            {
                decoded_suit = 3 - suit;
            }
            if (level + 7 <= _result.resTable[decoded_suit][dealer])
            {
                int over_trick = _result.resTable[decoded_suit][dealer] - (level + 7);
                cur_score = CONTRACT_VAL[doubled][vul[dealer % 2]][suit][level] + OVERTRICK_VAL[doubled][vul[dealer % 2]][suit] * over_trick;
            }
            else
            {
                int down = (level + 7) - _result.resTable[decoded_suit][dealer];
                cur_score = -UNDER_TRICKS_CHART[doubled][vul[dealer % 2]][down - 1];
            }
        }
    }
    int cur_score_NS = cur_score;
    if (dealer % 2 == 1)
    {
        cur_score_NS *= -1;
    }
    printf("cur_score_NS: %d\n", cur_score_NS);
    printf("best_score_NS: %d\n", best_score_NS);
    int diff = (cur_score_NS - best_score_NS) / 10;
    printf("diff: %d\n", diff);
    sign = 1;
    if (diff < 0)
    {
        sign = -1;
        diff = -diff;
    }
    if (diff > 400)
        diff = 400;
    printf("%d, %d", diff, sign);
    if (view % 2 == 0)
    {
        res.imp_loss = sign * IMP_CHART[diff];
    }
    else
    {
        res.imp_loss = sign * IMP_CHART[diff] * -1;
    }
    return res;
}

int main()
{
    int vul[2] = {1, 1};
    printf("imp loss: %d", test("N:AKQJT.AKQJT.A.K2 32.2.65432.AQJT9 7654.6543.87.876 98.987.KQJT9.543", vul, 4, 2, 0, 0, 0).imp_loss);
    return 0;
}