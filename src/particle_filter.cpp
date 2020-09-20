/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::discrete_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  num_particles = 150;  // TODO: Set the number of particles

  std::default_random_engine gen;

  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);


  for (int i = 0; i < num_particles; i++) {

    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1;

    particles.push_back(p);
    weights.push_back(1);

  }
  is_initialized = true ;

}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;

  for (int i = 0; i < num_particles; i++) {
    double x_f ;
    double y_f ;
    double theta_f ;

    if (yaw_rate == 0){
      x_f = particles[i].x + (velocity*delta_t*cos(particles[i].theta));
      y_f = particles[i].y + (velocity*delta_t*sin(particles[i].theta));
      theta_f = particles[i].theta;
    }
    else{
      x_f = particles[i].x + (velocity/yaw_rate)*(sin(particles[i].theta+(yaw_rate*delta_t)) - sin(particles[i].theta));
      y_f = particles[i].y + (velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta+(yaw_rate*delta_t)));
      theta_f = particles[i].theta + yaw_rate*delta_t;
    }

    std::normal_distribution<double> noise_x(x_f, std_pos[0]);
    std::normal_distribution<double> noise_y(y_f, std_pos[1]);
    std::normal_distribution<double> noise_theta(theta_f, std_pos[2]);

    particles[i].x = noise_x(gen);
    particles[i].y = noise_y(gen);
    particles[i].theta = noise_theta(gen);

  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations, double sensor_range) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */

  for(unsigned int i =0; i< observations.size(); i++){
    double min_dist = sensor_range *sensor_range;
    int min_id = 0;
    for(unsigned int j =0; j< predicted.size(); j++){
      double dist_ = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if(dist_ < min_dist){
        min_dist = dist_;
        min_id = predicted[j].id;
      }
    }
    observations[i].id = min_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   *   NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double sf = 0.0; //scaling factor
  for(int i =0; i< num_particles; i++){

    std::vector<LandmarkObs> transformed_obs;
    LandmarkObs obs;

    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;

    particles[i].weight = 1;

    for(unsigned int j =0; j< observations.size(); j++){
      LandmarkObs trans_obs;
      obs = observations[j];
      trans_obs.x = particles[i].x + (obs.x *cos(particles[i].theta)) - (obs.y * sin(particles[i].theta));
      trans_obs.y = particles[i].y + (obs.x *sin(particles[i].theta)) + (obs.y * cos(particles[i].theta));
      transformed_obs.push_back(trans_obs);
    }
    // find landmarks in the sensor range
    std::vector<LandmarkObs> predicted;

    for(unsigned int j =0; j< map_landmarks.landmark_list.size(); j++){
      double dist_ = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
      if(dist_ <= sensor_range){
        obs.x = map_landmarks.landmark_list[j].x_f;
        obs.y = map_landmarks.landmark_list[j].y_f;
        obs.id = map_landmarks.landmark_list[j].id_i;
        predicted.push_back(obs);
      }
    }
    //data association
    dataAssociation(predicted, transformed_obs, sensor_range);

    for(unsigned int j=0; j< transformed_obs.size(); j++){

      double means_x = transformed_obs[j].x;
      double means_y = transformed_obs[j].y;
      if(transformed_obs[j].id != 0){
        Map::single_landmark_s lm = map_landmarks.landmark_list.at(transformed_obs[j].id-1);
        long double gauss_norm = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);
        // calculate exponent and calculate weight using normalization terms and exponent
        long double exponent = gauss_norm * exp(-1*((pow((means_x-lm.x_f),2)/(2.0*pow(std_landmark[0],2))) + (pow((means_y-lm.y_f),2)/(2.0*pow(std_landmark[1],2)))));
        if(exponent > 0){
          particles[i].weight *= exponent;
        }
        associations.push_back(transformed_obs[j].id);
        sense_x.push_back(transformed_obs[j].x);
        sense_y.push_back(transformed_obs[j].y);
      }
    }
    SetAssociations(particles[i], associations, sense_x, sense_y );
    sf = sf + particles[i].weight;

  }

  for(int i =0; i< num_particles; i++){
    particles[i].weight = particles[i].weight/sf;
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::default_random_engine gen;
  std::discrete_distribution<int> distribution(weights.begin(), weights.end());

  std::vector<Particle> resample_particles;

  for(int i =0; i< num_particles; i++){
    resample_particles.push_back(particles[distribution(gen)]);
  }
  particles = resample_particles;

}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}