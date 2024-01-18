#ifndef __randonn_generator_H__
#define __randonn_generator_H__

#include <random>

class Randonn_generator
{
private:
    std::random_device seed;
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distributionDouble;
    std::uniform_int_distribution<int> distributionInt;    
    void initialize();

public:
    Randonn_generator();
    ~Randonn_generator();

    // random number generator
    // void setSeed(std::random_device seed);
    double randomDouble(double min, double max);
    int randomInt(int min, int max);

};

#endif // __randonn_generator_H__