#include "randonn_generator.h"

/* ============================================
*  Randonn_generator
*  Represents a random generator
* ============================================
*/

// ============================================
// =============== Constructors ===============
Randonn_generator::Randonn_generator()
{
    initialize();
}

// ========================================
// ============== Destructor ==============
Randonn_generator::~Randonn_generator()
{
    // std::cout<<"Random generator destroyed"<<std::endl;
}

// /// @brief Sets the seed of the random generator
// /// @param seed The seed of the random generator
// void Randonn_generator::setSeed(std::random_device seed)
// {
//     this->seed = seed;
// }

/// @brief Initializes the random generator by creating a random device and a default random engine
void Randonn_generator::initialize()
{
    generator = std::default_random_engine(seed());
}

/// @brief Generates a random double
/// @param min The minimum value of the random double
/// @param max The maximum value of the random double
/// @return A random double
double Randonn_generator::randomDouble(double min, double max)
{
    distributionDouble = std::uniform_real_distribution<double>(min, max);
    return distributionDouble(generator);
}

/// @brief Generates a random integer
/// @param min The minimum value of the random integer
/// @param max The maximum value of the random integer
/// @return A random integer
int Randonn_generator::randomInt(int min, int max)
{
    distributionInt = std::uniform_int_distribution<int>(min, max);
    return distributionInt(generator);
}